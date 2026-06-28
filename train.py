import json
import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import brentq
from scipy.special import ndtr
from sklearn.ensemble import IsolationForest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.risk.utils.mud_cake_risk import (
    AutoEncoder,
    DEFAULT_RANDOM_SEED,
    GEO_FEATURES,
    SORT_CANDIDATE_COLUMNS,
    TBM_FEATURES,
    build_multi_ring_sequences,
    calibrated_combined_risk,
    feature_masked_matrix,
    flat_feature_weights,
    normalize_with_feature_mask,
    sort_ring_dataframe,
    smooth_values,
    stratum_key,
    trend_vector,
)
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "app" / "risk" / "models"
STRATUM_MODEL_ROOT = MODEL_DIR / "mud_cake_by_stratum"
ALIGNED_DATA_ROOT = PROJECT_ROOT / "app" / "risk" / "aligned_by_stratum"
EXPECTED_STRATA = [str(i) for i in range(6)]
KDE_CALIBRATION_SCORES_FILENAME = "mud_cake_kde_calibration_scores.npy"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _ring_key(value: Any) -> int:
    return int(float(value))


def _normalize_ring_column(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Ring_Index": "Ring.No",
        "Ring_No": "Ring.No",
        "RING": "Ring.No",
        "ring": "Ring.No",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "Ring.No" not in df.columns:
        raise ValueError("缺少环号字段 Ring.No/Ring_Index/Ring_No")
    df["Ring.No"] = pd.to_numeric(df["Ring.No"], errors="coerce")
    df = df.dropna(subset=["Ring.No"]).copy()
    df["Ring.No"] = df["Ring.No"].astype(int)
    return df


def _normalize_tbm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_ring_column(df)
    missing = [feature for feature in TBM_FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"掘进参数缺少字段: {', '.join(missing)}")
    for feature in TBM_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna(subset=TBM_FEATURES, how="all").copy()
    return df


def _normalize_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_ring_column(df)
    if "Cluster_Label" not in df.columns:
        raise ValueError("地层数据缺少字段: Cluster_Label")
    missing = [feature for feature in GEO_FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"地层数据缺少字段: {', '.join(missing)}")
    keep_columns = ["Ring.No", "Cluster_Label"] + GEO_FEATURES
    df = df[keep_columns].copy()
    for feature in [col for col in df.columns if col != "Ring.No"]:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    missing_geo_cells = int(df[GEO_FEATURES].isna().sum().sum())
    if missing_geo_cells:
        print(f"地质参数缺失值已按 0 填充 | 缺失单元格数 {missing_geo_cells}", flush=True)
    df[GEO_FEATURES] = df[GEO_FEATURES].fillna(0.0)
    df = df.groupby("Ring.No", as_index=False).first()
    return df


def _candidate_files(folder: Path) -> Tuple[List[Path], List[Path]]:
    tables = [
        path for path in folder.iterdir()
        if path.is_file()
        and not path.name.startswith("~$")
        and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
    ]
    tbm_files: List[Path] = []
    geo_files: List[Path] = []
    for path in tables:
        try:
            columns = set(_read_table(path).head(0).columns)
        except Exception:
            continue
        if "Ring_Index" in columns or "Cluster_Label" in columns:
            geo_files.append(path)
        else:
            tbm_files.append(path)
    return tbm_files, geo_files


def _fuse_folder(folder: Path) -> List[pd.DataFrame]:
    tbm_files, geo_files = _candidate_files(folder)
    fused_frames: List[pd.DataFrame] = []
    for tbm_path in tbm_files:
        try:
            tbm_df = _normalize_tbm_columns(_read_table(tbm_path))
        except Exception as exc:
            print(f"跳过掘进文件 {tbm_path.name}: {exc}")
            continue
        for geo_path in geo_files:
            try:
                geo_df = _normalize_geo_columns(_read_table(geo_path))
            except Exception as exc:
                print(f"跳过地层文件 {geo_path.name}: {exc}")
                continue
            common_rings = sorted(set(tbm_df["Ring.No"]).intersection(set(geo_df["Ring.No"])))
            if not common_rings:
                continue
            tbm_common = tbm_df[tbm_df["Ring.No"].isin(common_rings)].copy()
            geo_common = geo_df[geo_df["Ring.No"].isin(common_rings)].copy()
            fused_df = tbm_common.merge(geo_common, on="Ring.No", how="inner")
            if fused_df.empty:
                continue
            fused_df["data_source"] = f"{folder.name}:{tbm_path.stem}+{geo_path.stem}"
            fused_frames.append(fused_df)
            print(
                f"融合完成 {fused_df['data_source'].iloc[0]} | "
                f"共同环号 {len(common_rings)} | 记录 {len(fused_df)}"
            )
    return fused_frames


def load_fused_training_data(data_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for folder in data_dirs:
        if folder.exists():
            frames.extend(_fuse_folder(folder))
    if not frames:
        raise ValueError("没有找到可用于共同环号 inner join 的掘进参数与地层数据")
    fused_df = pd.concat(frames, ignore_index=True)
    feature_columns = [col for col in TBM_FEATURES + GEO_FEATURES if col in fused_df.columns]
    fused_df = fused_df.dropna(subset=feature_columns, how="all").copy()
    return fused_df


def _split_source_rings(rings: List[int]) -> Tuple[List[int], List[int], List[int]]:
    n = len(rings)
    train_end = max(1, int(n * 0.6))
    val_end = max(train_end, int(n * 0.8))
    if n >= 3 and val_end == train_end:
        val_end = min(n - 1, train_end + 1)
    return rings[:train_end], rings[train_end:val_end], rings[val_end:]


def available_strata(data: pd.DataFrame) -> List[str]:
    if "Cluster_Label" not in data.columns:
        raise ValueError("缺少地层类别字段 Cluster_Label，无法按地层训练结泥饼模型")
    labels = pd.to_numeric(data["Cluster_Label"], errors="coerce").dropna().unique().tolist()
    return [stratum_key(label) for label in sorted(labels)]


def validate_expected_strata(data: pd.DataFrame) -> List[str]:
    strata = available_strata(data)
    missing = [label for label in EXPECTED_STRATA if label not in strata]
    if missing:
        raise ValueError(
            f"训练数据必须覆盖六类土 Cluster_Label=0..5，当前缺少: {', '.join(missing)}；"
            f"已检测到: {strata}"
        )
    extra = [label for label in strata if label not in EXPECTED_STRATA]
    if extra:
        print(f"检测到额外地层 {extra}，本次只按 0..5 训练六个模型。", flush=True)
    return EXPECTED_STRATA.copy()


def filter_by_stratum(data: pd.DataFrame, stratum_label: Any) -> pd.DataFrame:
    if "Cluster_Label" not in data.columns:
        raise ValueError("缺少地层类别字段 Cluster_Label，无法筛选地层训练数据")
    target = str(stratum_label)
    labels = pd.to_numeric(data["Cluster_Label"], errors="coerce")
    selected = data[labels.map(stratum_key) == target].copy()
    if selected.empty:
        raise ValueError(f"地层 {target} 没有可用训练数据")
    return selected


def _usable_feature_columns(data: pd.DataFrame) -> List[str]:
    required_features = TBM_FEATURES + GEO_FEATURES
    missing = [feature for feature in required_features if feature not in data.columns]
    if missing:
        raise ValueError(f"训练数据缺少模型输入字段: {', '.join(missing)}")
    return list(required_features)


def _feature_coverage(data: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, float]]:
    total = max(1, int(len(data)))
    coverage: Dict[str, Dict[str, float]] = {}
    for feature in features:
        if feature not in data.columns:
            valid_count = 0
        else:
            valid_count = int(pd.to_numeric(data[feature], errors="coerce").notna().sum())
        coverage[feature] = {
            "valid_count": valid_count,
            "valid_ratio": float(valid_count / total),
        }
    return coverage


def discover_aligned_data_dirs(root: Path = ALIGNED_DATA_ROOT) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"未找到对齐后的分地层数据目录: {root}")
    data_dirs = [path for path in sorted(root.glob("cluster_*")) if path.is_dir()]
    if not data_dirs:
        raise ValueError(f"目录 {root} 下未找到 cluster_* 子目录")
    return data_dirs


class MudCakeTrainer:
    def __init__(self, model_dir: Path = None, stratum_label: Any = None):
        self.autoencoder = None
        self.isolation_forest = None
        self.robust_scalers: Dict[str, List[Dict[str, float]]] = {}
        self.model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.stratum_label = str(stratum_label) if stratum_label is not None else None
        self.all_features: List[str] = []
        self.feature_dim = 0
        self.split_info: Dict[str, Any] = {}
        self.training_history: Dict[str, List[float]] = {}
        self.feature_coverage: Dict[str, Dict[str, float]] = {}
        self.kde_calibration_scores: Optional[np.ndarray] = None
        self.config = {
            "sequence_length": 6,
            "feature_weights": {
                "Thrust.Spd": 1.5,
                "Thrust.Force": 1.5,
                "CH.Tot.ContactForce": 1.5,
                "CH.Spd": 0.7,
                "CH.Torque": 0.7,
                "含水率": 1.1,
                "孔隙比": 1.1,
                "塑性指数": 1.1,
                "液性指数": 1.1,
                "内摩擦角": 1.1,
                "黏聚力": 1.1,
                "容重（重度）": 1.1,
                "饱和单轴抗压强度标准值": 1.1,
                "完整性指数": 1.1,
                "泊松比": 1.1,
                "重型圆锥动力触探锤击数N": 1.1,
            },
            "risk_fusion": {"ae_weight": 0.35, "if_weight": 0.65},
            "data_processing": {
                "random_seed": DEFAULT_RANDOM_SEED,
                "rings_per_sequence": 6,
                "window_points": 48,
                "window_stride": 8,
                "sequence_stride": 1,
                "max_train_sequences": 80000,
                "max_val_sequences": 20000,
                "max_test_sequences": 30000,
                "clip_zscore": 3.5,
                "smoothing_window": 5,
            },
            "autoencoder_params": {
                "hidden_size": 48,
                "dropout": 0.1,
                "learning_rate": 0.0005,
                "batch_size": 64,
                "epochs": 25,
                "patience": 10,
                "min_delta": 1e-4,
                "l2": 1e-5,
            },
            "isolation_forest_params": {
                "contamination": 0.01,
                "random_state": DEFAULT_RANDOM_SEED,
                "n_estimators": 200,
                "max_fit_samples": 50000,
                "max_samples_per_tree": 2048,
                "use_interactions": False,
                "n_jobs": -1,
            },
            "aggregation": {
                "final_ring_mode": "median",
                "quantile_p": 0.75,
                "topk_k": 3,
            },
            "risk_calibration": {
                "enabled": True,
                "method": "kde",
                "kde_kernel": "gaussian",
                "kde_low_cdf": 0.95,
                "kde_medium_cdf": 0.995,
                "kde_high_cdf": 0.999,
                "scale_low_quantile": 0.50,
                "scale_high_quantile": 0.999,
                "include_test_normal": True,
            },
            "reporting": {
                "score_batch_size": 1024,
                "scatter_sample_size": 12000,
            },
        }

    def _epoch_logger(self):
        trainer = self

        class EpochLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                optimizer = getattr(self.model, "optimizer", None)
                lr_value = ""
                if optimizer is not None:
                    try:
                        lr = optimizer.learning_rate
                        lr_value = f" lr={float(tf.keras.backend.get_value(lr)):.8f}"
                    except Exception:
                        lr_value = ""
                loss = logs.get("loss")
                val_loss = logs.get("val_loss")
                loss_text = f"{float(loss):.6f}" if loss is not None else "nan"
                val_text = f"{float(val_loss):.6f}" if val_loss is not None else "nan"
                print(
                    f"[cluster {trainer.stratum_label}] epoch={epoch + 1} "
                    f"loss={loss_text} val_loss={val_text}{lr_value}",
                    flush=True,
                )

        return EpochLogger()

    def load_training_data(self, data_dirs: Iterable[Path]) -> pd.DataFrame:
        data = load_fused_training_data(data_dirs)
        if self.stratum_label is not None:
            data = filter_by_stratum(data, self.stratum_label)
        self.all_features = _usable_feature_columns(data)
        self.feature_coverage = _feature_coverage(data, self.all_features)
        self._log_feature_coverage()
        self.feature_dim = len(self.all_features) * int(self.config["data_processing"]["window_points"])
        print(f"训练数据加载完成 | 记录 {len(data)} | 特征 {len(self.all_features)} | 特征维度 {self.feature_dim}")
        return data

    def prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.stratum_label is not None:
            data = filter_by_stratum(data, self.stratum_label)
        self.all_features = _usable_feature_columns(data)
        self.feature_coverage = _feature_coverage(data, self.all_features)
        self._log_feature_coverage()
        self.feature_dim = len(self.all_features) * int(self.config["data_processing"]["window_points"])
        print(
            f"训练数据加载完成 | 地层 {self.stratum_label or 'all'} | "
            f"记录 {len(data)} | 特征 {len(self.all_features)} | 特征维度 {self.feature_dim}"
        )
        return data

    def _log_feature_coverage(self) -> None:
        print(f"特征覆盖率 | 地层 {self.stratum_label or 'all'}", flush=True)
        for feature in self.all_features:
            item = self.feature_coverage.get(feature, {})
            print(
                f"  - {feature}: valid={int(item.get('valid_count', 0))} "
                f"ratio={float(item.get('valid_ratio', 0.0)):.2%}",
                flush=True,
            )

    def _fit_global_window_scalers(self, data: pd.DataFrame, train_rings_by_source: Dict[str, List[int]]) -> None:
        window_points = int(self.config["data_processing"]["window_points"])
        window_stride = int(self.config["data_processing"]["window_stride"])
        dim = len(self.all_features) * window_points
        values_per_dim: List[List[float]] = [[] for _ in range(dim)]
        for source, source_data in data.groupby("data_source"):
            train_rings = set(train_rings_by_source.get(source, []))
            for ring, ring_df in source_data[source_data["Ring.No"].isin(train_rings)].groupby("Ring.No"):
                ring_df = sort_ring_dataframe(ring_df, sort_candidates=SORT_CANDIDATE_COLUMNS)
                if len(ring_df) < window_points:
                    continue
                values, mask = feature_masked_matrix(ring_df, self.all_features)
                values = self._smooth(values)
                for j in range(len(self.all_features)):
                    base = j * window_points
                    feature_windows = np.lib.stride_tricks.sliding_window_view(values[:, j], window_points)[::window_stride]
                    mask_windows = np.lib.stride_tricks.sliding_window_view(mask[:, j], window_points)[::window_stride]
                    for k in range(window_points):
                        valid_values = feature_windows[:, k][mask_windows[:, k]]
                        if valid_values.size:
                            values_per_dim[base + k].extend(valid_values.astype(float).tolist())
        self.robust_scalers["global_window"] = []
        for values in values_per_dim:
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                self.robust_scalers["global_window"].append({"median": 0.0, "mad": 1.0})
                continue
            median = float(np.median(arr))
            mad_raw = float(np.median(np.abs(arr - median)))
            self.robust_scalers["global_window"].append({"median": median, "mad": 1.4826 * mad_raw if mad_raw > 0 else 1.0})

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        smoothing_window = int(self.config["data_processing"].get("smoothing_window", 0) or 0)
        return smooth_values(values, smoothing_window)

    def _normalize_with_feature_mask(self, data: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
        scalers = self.robust_scalers.get("global_window")
        if not scalers or len(scalers) != data.shape[1]:
            raise ValueError("缺少 global_window 鲁棒尺度")
        zmax = float(self.config["data_processing"].get("clip_zscore", 0) or 0)
        return normalize_with_feature_mask(data, feature_mask, scalers, zmax)

    def _build_sequences_for_rings(self, source_data: pd.DataFrame, rings: List[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
        seq_len = int(self.config["sequence_length"])
        window_points = int(self.config["data_processing"]["window_points"])
        window_stride = int(self.config["data_processing"]["window_stride"])
        rings_per_seq = int(self.config["data_processing"]["rings_per_sequence"])
        smoothing_window = int(self.config["data_processing"].get("smoothing_window", 0) or 0)
        sequence_stride = int(self.config["data_processing"].get("sequence_stride", 1) or 1)
        sequences, masks, _, _ = build_multi_ring_sequences(
            source_data,
            features=self.all_features,
            normalize_fn=self._normalize_with_feature_mask,
            sequence_length=seq_len,
            window_points=window_points,
            window_stride=window_stride,
            rings_per_sequence=rings_per_seq,
            smoothing_window=smoothing_window,
            ring_column="Ring.No",
            timestamp_field="time",
            sort_candidates=SORT_CANDIDATE_COLUMNS,
            rings=rings,
            sequence_stride=sequence_stride,
        )
        return list(zip(sequences, masks))

    @staticmethod
    def _limit_sequences(sequences: List[tuple], limit: int, seed_offset: int = 0) -> List[tuple]:
        if not limit or limit <= 0 or len(sequences) <= limit:
            return sequences
        rng = np.random.default_rng(DEFAULT_RANDOM_SEED + seed_offset)
        indices = np.sort(rng.choice(len(sequences), size=limit, replace=False))
        return [sequences[int(i)] for i in indices]

    @staticmethod
    def _split_sequences_temporally(sequences: List[tuple]) -> Tuple[List[tuple], List[tuple], List[tuple]]:
        n = len(sequences)
        if n <= 0:
            return [], [], []
        if n == 1:
            return sequences, [], []
        if n == 2:
            return sequences[:1], sequences[1:], []
        train_end = max(1, int(n * 0.6))
        val_end = max(train_end + 1, int(n * 0.8))
        if val_end >= n:
            val_end = n - 1
        return sequences[:train_end], sequences[train_end:val_end], sequences[val_end:]

    def create_ring_block_sequences(self, data: pd.DataFrame) -> Tuple[List[tuple], List[tuple], List[tuple]]:
        train_sequences: List[tuple] = []
        val_sequences: List[tuple] = []
        test_sequences: List[tuple] = []
        train_rings_by_source: Dict[str, List[int]] = {}
        split_info: Dict[str, Any] = {}
        for source, source_data in data.groupby("data_source"):
            rings = sorted({_ring_key(ring) for ring in source_data["Ring.No"].dropna().unique()})
            train_rings, val_rings, test_rings = _split_source_rings(rings)
            train_rings_by_source[source] = train_rings
            split_info[source] = {
                "train_rings": train_rings,
                "val_rings": val_rings,
                "test_rings": test_rings,
                "ring_count": len(rings),
                "split_mode": "ring_level",
            }
            print(
                f"Source {source} | rings train/val/test = "
                f"{len(train_rings)}/{len(val_rings)}/{len(test_rings)}",
                flush=True,
            )
        self._fit_global_window_scalers(data, train_rings_by_source)
        print("Robust scalers fitted; building sequences...", flush=True)
        for source, source_data in data.groupby("data_source"):
            info = split_info[source]
            source_train_sequences = self._build_sequences_for_rings(source_data, info["train_rings"])
            source_val_sequences = self._build_sequences_for_rings(source_data, info["val_rings"])
            source_test_sequences = self._build_sequences_for_rings(source_data, info["test_rings"])
            if (
                (not source_val_sequences or not source_test_sequences)
                and len(info["train_rings"]) + len(info["val_rings"]) + len(info["test_rings"])
                >= int(self.config["data_processing"]["rings_per_sequence"])
            ):
                all_rings = info["train_rings"] + info["val_rings"] + info["test_rings"]
                all_sequences = self._build_sequences_for_rings(source_data, all_rings)
                if len(all_sequences) >= 3:
                    source_train_sequences, source_val_sequences, source_test_sequences = self._split_sequences_temporally(all_sequences)
                    info["split_mode"] = "sequence_level_fallback"
                    info["fallback_sequence_count"] = len(all_sequences)
                    print(
                        f"Source {source} | fallback to temporal sequence split: "
                        f"train/val/test={len(source_train_sequences)}/{len(source_val_sequences)}/{len(source_test_sequences)}",
                        flush=True,
                    )
            train_sequences.extend(source_train_sequences)
            val_sequences.extend(source_val_sequences)
            test_sequences.extend(source_test_sequences)
            print(
                f"Built sequences for {source}: "
                f"train={len(train_sequences)} val={len(val_sequences)} test={len(test_sequences)}",
                flush=True,
            )
        processing_cfg = self.config.get("data_processing", {})
        raw_counts = {
            "train": len(train_sequences),
            "val": len(val_sequences),
            "test": len(test_sequences),
        }
        train_sequences = self._limit_sequences(train_sequences, int(processing_cfg.get("max_train_sequences", 0) or 0), 11)
        val_sequences = self._limit_sequences(val_sequences, int(processing_cfg.get("max_val_sequences", 0) or 0), 17)
        test_sequences = self._limit_sequences(test_sequences, int(processing_cfg.get("max_test_sequences", 0) or 0), 23)
        self.split_info = {
            "sources": split_info,
            "raw_sequence_counts": raw_counts,
            "sequence_counts": {
                "train": len(train_sequences),
                "val": len(val_sequences),
                "test": len(test_sequences),
            },
            "features": self.all_features,
            "geo_features_used_as_model_input": [f for f in GEO_FEATURES if f in self.all_features],
        }
        print(
            f"序列构建完成 | raw train={raw_counts['train']} val={raw_counts['val']} test={raw_counts['test']} | "
            f"used train={len(train_sequences)} val={len(val_sequences)} test={len(test_sequences)}",
            flush=True,
        )
        return train_sequences, val_sequences, test_sequences

    def train_autoencoder(self, train_sequences: List[tuple], val_sequences: List[tuple]) -> None:
        if not train_sequences:
            raise ValueError("没有有效训练序列")
        params = self.config["autoencoder_params"]
        self.autoencoder = AutoEncoder(
            input_size=self.feature_dim,
            hidden_size=int(params["hidden_size"]),
            dropout=float(params["dropout"]),
            l2=float(params.get("l2", 0.0)),
            track_loss=True,
        )
        self.autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(params["learning_rate"])),
            weighted_metrics=[],
        )
        batch_size = int(params["batch_size"])
        train_ds = self._sequence_dataset(train_sequences, batch_size, shuffle=True)
        val_ds = self._sequence_dataset(val_sequences, batch_size) if val_sequences else None
        monitor_metric = "val_loss" if val_ds is not None else "loss"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=int(params.get("patience", 8)),
                min_delta=float(params.get("min_delta", 0.0)),
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.5,
                patience=max(2, int(params.get("patience", 8)) // 2),
                min_lr=1e-5,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_dir / "mud_cake_autoencoder.best.weights.h5",
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=True,
            ),
            self._epoch_logger(),
        ]
        print(
            f"AutoEncoder training config | monitor={monitor_metric} "
            f"epochs={int(params['epochs'])} batch_size={batch_size} "
            f"patience={int(params.get('patience', 8))} min_delta={float(params.get('min_delta', 0.0))}",
            flush=True,
        )
        fit_kwargs = {
            "epochs": int(params["epochs"]),
            "verbose": 2,
            "steps_per_epoch": int(np.ceil(len(train_sequences) / batch_size)),
            "callbacks": callbacks,
        }
        if val_ds is not None:
            fit_kwargs.update({
                "validation_data": val_ds,
                "validation_steps": int(np.ceil(len(val_sequences) / batch_size)),
            })
        history = self.autoencoder.fit(train_ds, **fit_kwargs)
        best_weights = self.model_dir / "mud_cake_autoencoder.best.weights.h5"
        if best_weights.exists():
            self.autoencoder.load_weights(best_weights)
            best_weights.unlink()
        self.training_history = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }

    def _sequence_dataset(self, sequences: List[tuple], batch_size: int, shuffle: bool = False):
        seq_len = int(self.config["sequence_length"])
        feat_dim = int(self.feature_dim)
        weights = self._flat_feature_weights()
        x = np.asarray([seq for seq, _ in sequences], dtype=np.float32).reshape((-1, seq_len, feat_dim))
        masks = np.asarray([mask for _, mask in sequences], dtype=np.float32).reshape((-1, seq_len, feat_dim))
        weighted_masks = masks * weights.reshape(1, 1, feat_dim)
        dataset = tf.data.Dataset.from_tensor_slices((x, x, weighted_masks))
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=min(len(sequences), 10000),
                seed=DEFAULT_RANDOM_SEED,
                reshuffle_each_iteration=True,
            )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def _flat_feature_weights(self) -> np.ndarray:
        window_points = int(self.config["data_processing"]["window_points"])
        weight_map = self.config.get("feature_weights", {})
        return flat_feature_weights(self.all_features, weight_map, window_points)

    def _trend_vector(self, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return trend_vector(
            seq,
            mask,
            self.all_features,
            self.config.get("feature_weights", {}),
            int(self.config["data_processing"]["window_points"]),
            bool(self.config["isolation_forest_params"].get("use_interactions", True)),
        )

    def _trend_matrix(self, sequences: List[tuple]) -> np.ndarray:
        use_interactions = bool(self.config["isolation_forest_params"].get("use_interactions", True))
        if use_interactions:
            return np.array([self._trend_vector(seq, mask) for seq, mask in sequences], dtype=np.float32)

        x = np.array([seq for seq, _ in sequences], dtype=np.float32)
        masks = np.array([mask for _, mask in sequences], dtype=np.float32)
        n, seq_len, _ = x.shape
        feature_count = len(self.all_features)
        window_points = int(self.config["data_processing"]["window_points"])
        x = x.reshape(n, seq_len, feature_count, window_points)
        masks = masks.reshape(n, seq_len, feature_count, window_points)
        valid = masks > 0
        counts = valid.sum(axis=-1).astype(np.float32)

        t = np.arange(window_points, dtype=np.float32).reshape(1, 1, 1, window_points)
        sum_x = np.sum(t * valid, axis=-1)
        sum_y = np.sum(x * valid, axis=-1)
        sum_xx = np.sum((t ** 2) * valid, axis=-1)
        sum_xy = np.sum(t * x * valid, axis=-1)
        denom = counts * sum_xx - sum_x ** 2
        slopes = np.divide(
            counts * sum_xy - sum_x * sum_y,
            denom,
            out=np.zeros_like(sum_y, dtype=np.float32),
            where=(counts >= 3) & (np.abs(denom) > 1e-12),
        )

        first_idx = np.argmax(valid, axis=-1)
        last_idx = window_points - 1 - np.argmax(valid[..., ::-1], axis=-1)
        gather_shape = first_idx.shape + (1,)
        first_y = np.take_along_axis(x, first_idx.reshape(gather_shape), axis=-1).squeeze(-1)
        last_y = np.take_along_axis(x, last_idx.reshape(gather_shape), axis=-1).squeeze(-1)
        deltas = np.where(counts >= 3, last_y - first_y, 0.0)

        means = np.divide(sum_y, counts, out=np.zeros_like(sum_y, dtype=np.float32), where=counts > 0)
        variances = np.divide(
            np.sum(((x - means[..., None]) ** 2) * valid, axis=-1),
            counts,
            out=np.zeros_like(sum_y, dtype=np.float32),
            where=counts > 0,
        )
        stds = np.where(counts >= 3, np.sqrt(np.maximum(variances, 0.0)), 0.0)
        curvatures = np.zeros_like(stds, dtype=np.float32)

        weights = np.array(
            [float(self.config.get("feature_weights", {}).get(feature, 1.0)) for feature in self.all_features],
            dtype=np.float32,
        ).reshape(1, 1, feature_count)
        stacked = np.stack(
            [slopes * weights, deltas * weights, curvatures, stds * weights],
            axis=-1,
        )
        return stacked.reshape(n, seq_len * feature_count * 4).astype(np.float32)

    @staticmethod
    def _evenly_sample_sequences(sequences: List[tuple], max_count: int) -> List[tuple]:
        if max_count <= 0 or len(sequences) <= max_count:
            return sequences
        indices = np.linspace(0, len(sequences) - 1, max_count, dtype=int)
        return [sequences[i] for i in indices]

    def train_isolation_forest(self, train_sequences: List[tuple]) -> None:
        params = self.config["isolation_forest_params"]
        max_fit = int(params.get("max_fit_samples", 50000))
        indices = np.linspace(0, len(train_sequences) - 1, max_fit, dtype=int) if len(train_sequences) > max_fit else np.arange(len(train_sequences))
        print(f"Building IsolationForest trend vectors: {len(indices)} sequences...", flush=True)
        x_train = self._trend_matrix([train_sequences[i] for i in indices])
        max_samples = min(int(params.get("max_samples_per_tree", 8192)), len(x_train))
        print(
            f"Fitting IsolationForest: samples={len(x_train)} dim={x_train.shape[1]} "
            f"trees={int(params['n_estimators'])} max_samples_per_tree={max_samples}...",
            flush=True,
        )
        self.isolation_forest = IsolationForest(
            contamination=float(params["contamination"]),
            random_state=int(params["random_state"]),
            n_estimators=int(params["n_estimators"]),
            max_samples=max_samples,
            n_jobs=int(params.get("n_jobs", -1)),
        )
        self.isolation_forest.fit(x_train)

    def _raw_sequence_scores(
        self,
        sequences: List[tuple],
        calculate_combined: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not sequences:
            return np.array([]), np.array([]), np.array([])
        batch_size = int(self.config.get("reporting", {}).get("score_batch_size", 512) or 512)
        flat_weights = self._flat_feature_weights().reshape(1, 1, -1)
        ae_parts: List[np.ndarray] = []
        if_parts: List[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            x = np.array([seq for seq, _ in batch], dtype=np.float32)
            masks = np.array([mask for _, mask in batch], dtype=np.float32)
            reconstructed = self.autoencoder(x, training=False).numpy()
            ae_parts.append(np.mean(((x - reconstructed) ** 2) * (masks * flat_weights), axis=(1, 2)))
            trend_vectors = self._trend_matrix(batch)
            if_parts.append(-self.isolation_forest.decision_function(trend_vectors))
        ae_error = np.concatenate(ae_parts) if ae_parts else np.array([])
        if_anomaly = np.concatenate(if_parts) if if_parts else np.array([])
        combined = (
            self._calibrated_combined_score(ae_error, if_anomaly)
            if calculate_combined
            else np.array([])
        )
        return ae_error, if_anomaly, combined

    def _calibrated_combined_score(self, ae_error: np.ndarray, if_anomaly: np.ndarray) -> np.ndarray:
        calibration = self.config.get("risk_calibration", {})
        if calibration.get("enabled") is not True or calibration.get("method") != "kde":
            raise ValueError("Risk calibration must be enabled and use KDE")
        return calibrated_combined_risk(ae_error, if_anomaly, self.config)

    @staticmethod
    def _silverman_bandwidth(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            raise ValueError("KDE threshold calibration requires at least two finite scores")
        std = float(np.std(values, ddof=1))
        iqr = float(np.subtract(*np.percentile(values, [75, 25])))
        robust_sigma = min(std, iqr / 1.349) if iqr > 0 else std
        if robust_sigma <= 1e-12:
            raise ValueError("KDE scores have no usable dispersion; bandwidth cannot be data-derived")
        bandwidth = 0.9 * robust_sigma * (values.size ** (-1 / 5))
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Silverman KDE bandwidth is invalid")
        return float(bandwidth)

    @staticmethod
    def _kde_cdf_thresholds(scores: np.ndarray, calibration: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size < 20:
            raise ValueError("KDE threshold calibration requires at least 20 finite scores")
        cdf_points = {
            "low": float(calibration.get("kde_low_cdf", 0.95)),
            "medium": float(calibration.get("kde_medium_cdf", 0.995)),
            "high": float(calibration.get("kde_high_cdf", 0.999)),
        }
        if not (0.0 < cdf_points["low"] < cdf_points["medium"] < cdf_points["high"] < 1.0):
            raise ValueError("KDE CDF levels must satisfy 0 < low < medium < high < 1")

        kernel = str(calibration.get("kde_kernel", "gaussian") or "gaussian")
        if kernel != "gaussian":
            raise ValueError("Strict KDE calibration currently supports only the gaussian kernel")
        bandwidth = MudCakeTrainer._silverman_bandwidth(scores)

        def raw_cdf(x: float) -> float:
            return float(np.mean(ndtr((x - scores) / bandwidth)))

        support_low, support_high = 0.0, 1.0
        cdf_low, cdf_high = raw_cdf(support_low), raw_cdf(support_high)
        normalization = cdf_high - cdf_low
        if not np.isfinite(normalization) or normalization <= 1e-12:
            raise ValueError("Bounded KDE CDF normalization failed")

        def bounded_cdf(x: float) -> float:
            return (raw_cdf(x) - cdf_low) / normalization

        thresholds = {
            name: float(
                brentq(
                    lambda x, target=prob: bounded_cdf(x) - target,
                    support_low,
                    support_high,
                    xtol=1e-12,
                    rtol=1e-12,
                )
            )
            for name, prob in cdf_points.items()
        }
        if not (thresholds["low"] < thresholds["medium"] < thresholds["high"]):
            raise ValueError("KDE inverse CDF did not produce strictly ordered thresholds")
        score_bytes = np.ascontiguousarray(scores.astype("<f8")).tobytes()
        metadata = {
            "method": "kde",
            "kernel": kernel,
            "cdf_model": "gaussian_kde_conditioned_on_[0,1]",
            "bandwidth_method": "silverman_robust",
            "bandwidth": float(bandwidth),
            "cdf_levels": cdf_points,
            "support": [support_low, support_high],
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_sha256": hashlib.sha256(score_bytes).hexdigest(),
            "threshold_derivation": "inverse of bounded Gaussian KDE CDF; no threshold override",
        }
        return thresholds, metadata

    def calibrate_risk_thresholds(
        self,
        train_sequences: List[tuple],
        val_sequences: List[tuple],
        test_sequences: Optional[List[tuple]] = None,
    ) -> None:
        calibration = dict(self.config.get("risk_calibration", {}))
        if calibration.get("enabled") is not True or calibration.get("method") != "kde":
            raise ValueError("Risk thresholds must be calibrated with enabled KDE")
        calibration_sequences = list(train_sequences)
        if val_sequences:
            calibration_sequences.extend(val_sequences)
        include_test = calibration.get("include_test_normal") is True
        if include_test and test_sequences:
            calibration_sequences.extend(test_sequences)
        split_names = "train+validation+test" if include_test else "train+validation"
        print(f"Calibrating KDE on all {len(calibration_sequences)} {split_names} sequences...", flush=True)
        ae_error, if_anomaly, _ = self._raw_sequence_scores(
            calibration_sequences,
            calculate_combined=False,
        )
        if ae_error.size == 0 or if_anomaly.size == 0:
            raise ValueError("No scores are available for KDE threshold calibration")
        low_quantile = float(calibration.get("scale_low_quantile", 0.50))
        high_quantile = float(calibration.get("scale_high_quantile", 0.999))
        calibration["ae_error_bounds"] = {
            "low": float(np.quantile(ae_error, low_quantile)),
            "high": float(np.quantile(ae_error, high_quantile)),
        }
        calibration["if_anomaly_bounds"] = {
            "low": float(np.quantile(if_anomaly, low_quantile)),
            "high": float(np.quantile(if_anomaly, high_quantile)),
        }
        self.config["risk_calibration"] = calibration
        _, _, combined = self._raw_sequence_scores(calibration_sequences)
        thresholds, kde_metadata = self._kde_cdf_thresholds(combined, calibration)
        calibration["method"] = "kde"
        calibration["risk_thresholds"] = thresholds
        calibration["kde_threshold_metadata"] = kde_metadata
        calibration["calibration_sample_count"] = int(combined.size)
        calibration["calibration_splits"] = {
            "train": int(len(train_sequences)),
            "validation": int(len(val_sequences)),
            "test": int(len(test_sequences)) if include_test and test_sequences else 0,
        }
        self.kde_calibration_scores = np.asarray(combined, dtype="<f8")
        self.config["risk_calibration"] = calibration

    def _risk_level_counts(self, scores: np.ndarray) -> Dict[str, int]:
        thresholds = self._strict_kde_thresholds()
        low, medium, high = thresholds["low"], thresholds["medium"], thresholds["high"]
        return {
            "no_risk": int(np.sum(scores < low)),
            "low": int(np.sum((scores >= low) & (scores < medium))),
            "medium": int(np.sum((scores >= medium) & (scores < high))),
            "high": int(np.sum(scores >= high)),
        }

    def _strict_kde_thresholds(self) -> Dict[str, float]:
        calibration = self.config.get("risk_calibration", {})
        if calibration.get("enabled") is not True or calibration.get("method") != "kde":
            raise ValueError("Missing strict KDE risk calibration")
        thresholds = calibration.get("risk_thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != {"low", "medium", "high"}:
            raise ValueError("KDE thresholds must contain exactly low/medium/high")
        values = {name: float(thresholds[name]) for name in ("low", "medium", "high")}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("KDE thresholds contain non-finite values")
        if not (0.0 <= values["low"] < values["medium"] < values["high"] <= 1.0):
            raise ValueError("KDE thresholds must be strictly ordered within [0, 1]")
        metadata = calibration.get("kde_threshold_metadata", {})
        if metadata.get("threshold_derivation") != "inverse of bounded Gaussian KDE CDF; no threshold override":
            raise ValueError("KDE threshold provenance is missing or invalid")
        return values

    def _score_frame(self, split_name: str, sequences: List[tuple]) -> pd.DataFrame:
        ae_error, if_anomaly, combined = self._raw_sequence_scores(sequences)
        if combined.size == 0:
            return pd.DataFrame(columns=["split", "ae_error", "if_anomaly", "risk_score"])
        return pd.DataFrame({
            "split": split_name,
            "ae_error": ae_error,
            "if_anomaly": if_anomaly,
            "risk_score": combined,
        })

    def _save_training_report(
        self,
        train_sequences: List[tuple],
        val_sequences: List[tuple],
        test_sequences: List[tuple],
    ) -> Dict[str, Any]:
        report_dir = self.model_dir / "training_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        frames = [
            self._score_frame("train", train_sequences),
            self._score_frame("val", val_sequences),
            self._score_frame("test", test_sequences),
        ]
        scores_df = pd.concat(frames, ignore_index=True)
        scores_path = report_dir / "sequence_scores.csv"
        scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")

        thresholds = self._strict_kde_thresholds()
        calibration = self.config.get("risk_calibration", {})
        low, medium, high = thresholds["low"], thresholds["medium"], thresholds["high"]
        test_scores = scores_df[scores_df["split"] == "test"]["risk_score"].to_numpy(dtype=float)
        counts = self._risk_level_counts(test_scores)
        total = max(1, int(test_scores.size))
        metrics = {
            "stratum_label": self.stratum_label,
            "feature_count": int(len(self.all_features)),
            "feature_dim": int(self.feature_dim),
            "sequence_length": int(self.config["sequence_length"]),
            "rings_per_sequence": int(self.config["data_processing"]["rings_per_sequence"]),
            "window_points": int(self.config["data_processing"]["window_points"]),
            "window_stride": int(self.config["data_processing"]["window_stride"]),
            "sequence_stride": int(self.config["data_processing"].get("sequence_stride", 1) or 1),
            "train_sequences": int(len(train_sequences)),
            "val_sequences": int(len(val_sequences)),
            "test_sequences": int(len(test_sequences)),
            "test_no_risk_rate": counts["no_risk"] / total,
            "test_low_rate": counts["low"] / total,
            "test_medium_rate": counts["medium"] / total,
            "test_high_rate": counts["high"] / total,
            "test_medium_high_rate": (counts["medium"] + counts["high"]) / total,
            "test_high_rate_only": counts["high"] / total,
            "threshold_low": low,
            "threshold_medium": medium,
            "threshold_high": high,
            "threshold_method": calibration["method"],
            "kde_threshold_metadata": calibration.get("kde_threshold_metadata", {}),
        }
        metrics_path = report_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        self._plot_loss_curve(report_dir / "loss_curve.png")
        self._plot_risk_distribution(scores_df, report_dir / "risk_distribution.png", low, medium, high)
        self._plot_ae_if_scatter(scores_df, report_dir / "ae_if_scatter.png")
        self._plot_test_risk_levels(counts, report_dir / "test_risk_levels.png")
        self._plot_report_dashboard(scores_df, counts, metrics, report_dir / "training_dashboard.png", low, medium, high)
        return {
            "report_dir": str(report_dir),
            "scores_csv": str(scores_path),
            "metrics_json": str(metrics_path),
            "metrics": metrics,
            "figures": [
                str(report_dir / "loss_curve.png"),
                str(report_dir / "risk_distribution.png"),
                str(report_dir / "ae_if_scatter.png"),
                str(report_dir / "test_risk_levels.png"),
                str(report_dir / "training_dashboard.png"),
            ],
        }

    def _plot_loss_curve(self, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
        loss = self.training_history.get("loss", [])
        val_loss = self.training_history.get("val_loss", [])
        if loss:
            ax.plot(range(1, len(loss) + 1), loss, label="Train loss", linewidth=2.4, color="#2563eb")
        if val_loss:
            ax.plot(range(1, len(val_loss) + 1), val_loss, label="Validation loss", linewidth=2.4, color="#f97316")
        ax.set_title(f"AutoEncoder Learning Curve | Cluster {self.stratum_label}", fontsize=14, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Masked reconstruction loss")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_risk_distribution(self, scores_df: pd.DataFrame, output_path: Path, low: float, medium: float, high: float) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
        colors = {"train": "#2563eb", "val": "#22c55e", "test": "#f97316"}
        for split in ["train", "val", "test"]:
            values = scores_df[scores_df["split"] == split]["risk_score"].dropna().to_numpy(dtype=float)
            if values.size:
                ax.hist(values, bins=60, alpha=0.35, density=True, label=split, color=colors[split])
        for value, label, color in [(low, "low", "#facc15"), (medium, "medium", "#fb923c"), (high, "high", "#ef4444")]:
            ax.axvline(value, color=color, linestyle="--", linewidth=2, label=f"{label} threshold")
        ax.set_title(f"Normal-Condition Risk Score Distribution | Cluster {self.stratum_label}", fontsize=14, weight="bold")
        ax.set_xlabel("Calibrated risk score")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_ae_if_scatter(self, scores_df: pd.DataFrame, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=180)
        sample_size = int(self.config.get("reporting", {}).get("scatter_sample_size", 12000) or 12000)
        plot_df = scores_df.sample(min(len(scores_df), sample_size), random_state=DEFAULT_RANDOM_SEED) if len(scores_df) else scores_df
        colors = {"train": "#2563eb", "val": "#22c55e", "test": "#f97316"}
        for split in ["train", "val", "test"]:
            part = plot_df[plot_df["split"] == split]
            if not part.empty:
                ax.scatter(part["ae_error"], part["if_anomaly"], s=8, alpha=0.35, label=split, color=colors[split])
        ax.set_title(f"AE Error vs IF Anomaly | Cluster {self.stratum_label}", fontsize=14, weight="bold")
        ax.set_xlabel("AutoEncoder reconstruction error")
        ax.set_ylabel("IsolationForest anomaly score")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_test_risk_levels(self, counts: Dict[str, int], output_path: Path) -> None:
        labels = ["no_risk", "low", "medium", "high"]
        values = [counts.get(label, 0) for label in labels]
        colors = ["#22c55e", "#facc15", "#fb923c", "#ef4444"]
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=180)
        bars = ax.bar(labels, values, color=colors, width=0.62)
        total = max(1, sum(values))
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value}\n{value / total:.2%}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title(f"Normal Test Set Risk Levels | Cluster {self.stratum_label}", fontsize=14, weight="bold")
        ax.set_ylabel("Sequence count")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_report_dashboard(
        self,
        scores_df: pd.DataFrame,
        counts: Dict[str, int],
        metrics: Dict[str, Any],
        output_path: Path,
        low: float,
        medium: float,
        high: float,
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
        ax = axes[0, 0]
        loss = self.training_history.get("loss", [])
        val_loss = self.training_history.get("val_loss", [])
        if loss:
            ax.plot(range(1, len(loss) + 1), loss, label="Train", color="#2563eb", linewidth=2)
        if val_loss:
            ax.plot(range(1, len(val_loss) + 1), val_loss, label="Validation", color="#f97316", linewidth=2)
        ax.set_title("Learning curve", weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

        ax = axes[0, 1]
        for split, color in [("train", "#2563eb"), ("val", "#22c55e"), ("test", "#f97316")]:
            values = scores_df[scores_df["split"] == split]["risk_score"].dropna().to_numpy(dtype=float)
            if values.size:
                ax.hist(values, bins=50, alpha=0.35, density=True, label=split, color=color)
        for value, color in [(low, "#facc15"), (medium, "#fb923c"), (high, "#ef4444")]:
            ax.axvline(value, color=color, linestyle="--", linewidth=1.8)
        ax.set_title("Risk distribution with thresholds", weight="bold")
        ax.set_xlabel("Risk score")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

        ax = axes[1, 0]
        labels = ["no_risk", "low", "medium", "high"]
        values = [counts.get(label, 0) for label in labels]
        colors = ["#22c55e", "#facc15", "#fb923c", "#ef4444"]
        ax.bar(labels, values, color=colors)
        ax.set_title("Normal test set risk levels", weight="bold")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)

        ax = axes[1, 1]
        ax.axis("off")
        text = (
            f"Cluster: {self.stratum_label}\n"
            f"Features: {metrics['feature_count']} | Dim: {metrics['feature_dim']}\n"
            f"Sequence: {metrics['rings_per_sequence']} rings x {metrics['window_points']} points\n"
            f"Train sequences: {metrics['train_sequences']}\n"
            f"Validation sequences: {metrics['val_sequences']}\n"
            f"Test sequences: {metrics['test_sequences']}\n\n"
            f"Test no-risk rate: {metrics['test_no_risk_rate']:.2%}\n"
            f"Test medium+high false alarm rate: {metrics['test_medium_high_rate']:.2%}\n"
            f"Test high false alarm rate: {metrics['test_high_rate_only']:.2%}\n\n"
            f"Low threshold: {low:.4f}\n"
            f"Medium threshold: {medium:.4f}\n"
            f"High threshold: {high:.4f}\n"
        )
        ax.text(0.03, 0.95, text, va="top", ha="left", fontsize=13, linespacing=1.55)
        fig.suptitle(f"Mud-Cake Normal-Condition Model Report | Cluster {self.stratum_label}", fontsize=17, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def train(self, data_dirs: Iterable[Path]) -> Dict[str, Any]:
        data = self.load_training_data(data_dirs)
        return self.train_from_dataframe(data)

    def train_from_dataframe(self, data: pd.DataFrame) -> Dict[str, Any]:
        data = self.prepare_training_data(data)
        train_sequences, val_sequences, test_sequences = self.create_ring_block_sequences(data)
        self.train_autoencoder(train_sequences, val_sequences)
        print("AutoEncoder training finished; fitting IsolationForest...", flush=True)
        self.train_isolation_forest(train_sequences)
        print("IsolationForest finished; calibrating risk thresholds...", flush=True)
        self.calibrate_risk_thresholds(train_sequences, val_sequences, test_sequences)
        print("Risk calibration finished; saving model...", flush=True)
        self.save_models()
        print("Model saved; generating training report figures...", flush=True)
        report = self._save_training_report(train_sequences, val_sequences, test_sequences)
        print(f"Training report saved: {report['report_dir']}", flush=True)
        return {
            "status": "success",
            "data_size": len(data),
            "sequence_count": len(train_sequences) + len(val_sequences) + len(test_sequences),
            "feature_count": self.feature_dim,
            "features": self.all_features,
            "stratum_label": self.stratum_label,
            "model_dir": str(self.model_dir),
            "report": report,
        }

    def save_models(self) -> None:
        self._strict_kde_thresholds()
        if self.kde_calibration_scores is None or self.kde_calibration_scores.size < 20:
            raise ValueError("KDE calibration scores are missing; refusing to save model")
        if self.autoencoder is not None:
            self.autoencoder.save_weights(self.model_dir / "mud_cake_autoencoder.weights.h5")
        if self.isolation_forest is not None:
            joblib.dump(self.isolation_forest, self.model_dir / "mud_cake_isolation_forest.pkl")
        joblib.dump({"robust": self.robust_scalers}, self.model_dir / "mud_cake_scalers.pkl")
        np.save(
            self.model_dir / KDE_CALIBRATION_SCORES_FILENAME,
            np.asarray(self.kde_calibration_scores, dtype="<f8"),
        )
        model_info = {
            "learning_type": "unsupervised_normal_only",
            "training_mode": "stratum_specific" if self.stratum_label is not None else "unified",
            "stratum_label": self.stratum_label,
            "config": self.config,
            "features": self.all_features,
            "feature_dim": self.feature_dim,
            "input_feature_counts": {
                "tbm_features": len([f for f in TBM_FEATURES if f in self.all_features]),
                "geo_features": len([f for f in GEO_FEATURES if f in self.all_features]),
            },
            "feature_coverage": self.feature_coverage,
            "feature_groups": {
                "tbm_features": [f for f in TBM_FEATURES if f in self.all_features],
                "engineered_features": [],
                "geo_features": [f for f in GEO_FEATURES if f in self.all_features],
            },
            "split_info": self.split_info,
            "training_history": self.training_history,
        }
        with open(self.model_dir / "mud_cake_model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f"模型保存完成: {self.model_dir}")


def validate_saved_kde_model(model_dir: Path, expected_label: str) -> Dict[str, Any]:
    required = [
        model_dir / "mud_cake_model_info.json",
        model_dir / "mud_cake_autoencoder.weights.h5",
        model_dir / "mud_cake_isolation_forest.pkl",
        model_dir / "mud_cake_scalers.pkl",
        model_dir / KDE_CALIBRATION_SCORES_FILENAME,
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Cluster {expected_label} model artifacts are incomplete: {missing}")

    with open(model_dir / "mud_cake_model_info.json", "r", encoding="utf-8") as f:
        model_info = json.load(f)
    if str(model_info.get("stratum_label")) != str(expected_label):
        raise ValueError(f"Cluster {expected_label} model label does not match its directory")

    calibration = model_info.get("config", {}).get("risk_calibration", {})
    if calibration.get("enabled") is not True or calibration.get("method") != "kde":
        raise ValueError(f"Cluster {expected_label} does not contain strict KDE calibration")
    saved_thresholds = calibration.get("risk_thresholds", {})
    if set(saved_thresholds) != {"low", "medium", "high"}:
        raise ValueError(f"Cluster {expected_label} KDE thresholds are incomplete")

    scores = np.load(model_dir / KDE_CALIBRATION_SCORES_FILENAME, allow_pickle=False)
    scores = np.asarray(scores, dtype="<f8").reshape(-1)
    if scores.size != int(calibration.get("calibration_sample_count", -1)):
        raise ValueError(f"Cluster {expected_label} KDE score count does not match model metadata")
    score_hash = hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest()
    metadata = calibration.get("kde_threshold_metadata", {})
    if score_hash != metadata.get("score_sha256"):
        raise ValueError(f"Cluster {expected_label} KDE calibration score hash mismatch")
    recomputed, _ = MudCakeTrainer._kde_cdf_thresholds(scores, calibration)
    for name in ("low", "medium", "high"):
        if not np.isclose(float(saved_thresholds[name]), recomputed[name], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Cluster {expected_label} threshold {name} is not the KDE result: "
                f"saved={saved_thresholds[name]}, recomputed={recomputed[name]}"
            )
    return {
        "label": str(expected_label),
        "thresholds": {name: float(saved_thresholds[name]) for name in ("low", "medium", "high")},
        "score_count": int(scores.size),
        "score_sha256": score_hash,
        "bandwidth": float(metadata["bandwidth"]),
        "cdf_levels": metadata["cdf_levels"],
    }


def train_stratum_models(data_dirs: Iterable[Path]) -> Dict[str, Any]:
    data_dirs = list(data_dirs)
    print("训练数据目录:")
    for folder in data_dirs:
        print(f"- {folder}")
    data = load_fused_training_data(data_dirs)
    strata = validate_expected_strata(data)
    print(f"检测到完整六类地层，按固定顺序训练: {strata}")
    STRATUM_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = MODEL_DIR / "mud_cake_strata_manifest.json"
    manifest_path.unlink(missing_ok=True)
    results = {}
    manifest = {
        "learning_type": "unsupervised_normal_only",
        "training_mode": "stratum_specific",
        "selector_feature": "Cluster_Label",
        "expected_strata": EXPECTED_STRATA,
        "strata": {},
    }
    for label in strata:
        model_dir = STRATUM_MODEL_ROOT / f"cluster_{label}"
        print(f"\n开始训练地层 Cluster_Label={label} 的结泥饼模型 -> {model_dir}")
        trainer = MudCakeTrainer(model_dir=model_dir, stratum_label=label)
        result = trainer.train_from_dataframe(data)
        kde_audit = validate_saved_kde_model(model_dir, label)
        results[label] = result
        manifest["strata"][label] = {
            "model_dir": str(model_dir.relative_to(MODEL_DIR)),
            "features": result["features"],
            "feature_count": result["feature_count"],
            "sequence_count": result["sequence_count"],
            "data_size": result["data_size"],
            "report": result["report"],
            "kde_audit": kde_audit,
        }
    if set(manifest["strata"]) != set(EXPECTED_STRATA):
        raise RuntimeError("Six-stratum training did not produce exactly Cluster_Label=0..5")
    for label in EXPECTED_STRATA:
        validate_saved_kde_model(STRATUM_MODEL_ROOT / f"cluster_{label}", label)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {"status": "success", "training_mode": "stratum_specific", "strata": results}


def summarize_stratum_training_data(data_dirs: Iterable[Path]) -> Dict[str, Any]:
    data_dirs = list(data_dirs)
    data = load_fused_training_data(data_dirs)
    strata = validate_expected_strata(data)
    summary: Dict[str, Any] = {}
    for label in strata:
        subset = filter_by_stratum(data, label)
        features = _usable_feature_columns(subset)
        rings = sorted({_ring_key(ring) for ring in subset["Ring.No"].dropna().unique()})
        per_ring = subset.groupby("Ring.No").size()
        summary[label] = {
            "rows": int(len(subset)),
            "rings": int(len(rings)),
            "ring_min": int(min(rings)) if rings else None,
            "ring_max": int(max(rings)) if rings else None,
            "points_per_ring_min": int(per_ring.min()) if len(per_ring) else 0,
            "points_per_ring_median": float(per_ring.median()) if len(per_ring) else 0.0,
            "points_per_ring_max": int(per_ring.max()) if len(per_ring) else 0,
            "features": features,
            "feature_dim": len(features) * 48,
        }
    return {"status": "success", "data_dirs": [str(path) for path in data_dirs], "strata": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="按地层训练结泥饼风险预警模型")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ALIGNED_DATA_ROOT,
        help="对齐后的分地层数据根目录，默认 aligned_by_stratum",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        dest="data_dirs",
        help="显式指定一个训练数据目录，可重复传入；不传时自动读取 data-root/cluster_*",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查和汇总训练数据，不执行模型训练",
    )
    args = parser.parse_args()

    np.random.seed(DEFAULT_RANDOM_SEED)
    tf.random.set_seed(DEFAULT_RANDOM_SEED)
    data_dirs = args.data_dirs if args.data_dirs else discover_aligned_data_dirs(args.data_root)
    if args.dry_run:
        summary = summarize_stratum_training_data(data_dirs)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    result = train_stratum_models(data_dirs)
    print("按地层训练完成")
    for label, item in result["strata"].items():
        print(
            f"Cluster_Label={label} | 样本数: {item['data_size']} | "
            f"序列数: {item['sequence_count']} | 特征维度: {item['feature_count']}"
        )


if __name__ == "__main__":
    main()

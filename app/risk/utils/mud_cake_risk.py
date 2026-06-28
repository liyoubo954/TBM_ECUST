import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

DEFAULT_RANDOM_SEED = 42
RISK_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MUD_CAKE_MODEL_INFO_FILENAME = "mud_cake_model_info.json"
MUD_CAKE_AUTOENCODER_FILENAME = "mud_cake_autoencoder.weights.h5"
MUD_CAKE_ISOLATION_FOREST_FILENAME = "mud_cake_isolation_forest.pkl"
MUD_CAKE_SCALERS_FILENAME = "mud_cake_scalers.pkl"

TBM_FEATURES = [
    "Thrust.Spd",
    "CH.Spd",
    "CH.Torque",
    "Thrust.Force",     
    "CH.Tot.ContactForce",
]

GEO_FEATURES = [
    "含水率",
    "容重（重度）",
    "孔隙比",
    "塑性指数",
    "液性指数",
    "内摩擦角",
    "黏聚力",
    "饱和单轴抗压强度标准值",
    "完整性指数",
    "泊松比",
    "重型圆锥动力触探锤击数N",
]

SORT_CANDIDATE_COLUMNS = [
    "time",
    "ts(Asia/Shanghai)",
    "timestamp",
    "TIME",
    "Timestamp",
    "time_stamp",
    "采样时间",
    "DATETIME",
    "Date",
    "date",
    "时间",
    "POINT_NO",
    "ts",
]


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _safe_ratio_value(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator) or abs(denominator) <= 1e-6:
        return np.nan
    return numerator / abs(denominator)


def add_engineered_features_to_row(values: Dict[str, Any]) -> Dict[str, Any]:
    speed = abs(_safe_float(values.get("Thrust.Spd")))
    force = abs(_safe_float(values.get("Thrust.Force")))
    torque = abs(_safe_float(values.get("CH.Torque")))
    contact_force = abs(_safe_float(values.get("CH.Tot.ContactForce")))
    values["Torque.Abs"] = torque
    values["Force.Per.Speed"] = _safe_ratio_value(force, speed)
    values["Torque.Per.Speed"] = _safe_ratio_value(torque, speed)
    values["ContactForce.Per.Speed"] = _safe_ratio_value(contact_force, speed)
    values["Torque.Per.Force"] = _safe_ratio_value(torque, force)
    values["ContactForce.Per.Force"] = _safe_ratio_value(contact_force, force)
    return values


def stratum_key(value: Any) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return str(f).replace(".", "_")
    except Exception:
        return str(value).strip()


class AutoEncoder(tf.keras.Model):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        l2: float = 0.0,
        track_loss: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.loss_tracker = tf.keras.metrics.Mean(name="loss") if track_loss else None
        reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0.0 else None
        self.encoder = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            kernel_regularizer=reg,
            recurrent_regularizer=reg,
            bias_regularizer=None,
        )
        self.decoder = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            dropout=dropout,
            kernel_regularizer=reg,
            recurrent_regularizer=reg,
            bias_regularizer=None,
        )
        self.output_dense = tf.keras.layers.Dense(input_size, kernel_regularizer=reg)

    @property
    def metrics(self):
        return [self.loss_tracker] if self.loss_tracker is not None else []

    def call(self, x, training=False):
        encoded_seq, h, c = self.encoder(x, training=training)
        decoded_seq = self.decoder(encoded_seq, initial_state=[h, c], training=training)
        return self.output_dense(decoded_seq)

    def train_step(self, data):
        x, y, mask = _unpack_autoencoder_batch(data)
        with tf.GradientTape() as tape:
            loss = masked_reconstruction_loss(y, self(x, training=True), mask)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        if self.loss_tracker is None:
            return {"loss": loss}
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y, mask = _unpack_autoencoder_batch(data)
        loss = masked_reconstruction_loss(y, self(x, training=False), mask)
        if self.loss_tracker is None:
            return {"loss": loss}
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def _unpack_autoencoder_batch(data):
    if isinstance(data, (tuple, list)):
        if len(data) == 3:
            return data
        if len(data) == 2:
            x, y = data
            return x, y, tf.ones_like(y)
    return data, data, tf.ones_like(data)


def masked_reconstruction_loss(y_true, y_pred, mask):
    mask_f = tf.cast(mask, tf.float32)
    sq_err = tf.square(y_true - y_pred) * mask_f
    per_sample_sum = tf.reduce_sum(sq_err, axis=[1, 2])
    per_sample_count = tf.maximum(tf.reduce_sum(mask_f, axis=[1, 2]), 1.0)
    return tf.reduce_mean(per_sample_sum / per_sample_count)


def feature_masked_matrix(ring_df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(ring_df), len(features)), dtype=float)
    mask = np.zeros((len(ring_df), len(features)), dtype=bool)
    for j, feature in enumerate(features):
        series = pd.to_numeric(ring_df.get(feature), errors="coerce")
        vals = series.to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        if np.all(valid):
            values[:, j] = vals
            mask[:, j] = True
            continue
        for i in range(len(ring_df)):
            if valid[i]:
                values[i, j] = vals[i]
                mask[i, j] = True
            else:
                values[i, j] = values[i - 1, j] if i > 0 else 0.0
    return values, mask


def smooth_values(values: np.ndarray, smoothing_window: int) -> np.ndarray:
    if not smoothing_window or smoothing_window <= 1:
        return values
    smoothed = values.copy()
    for j in range(values.shape[1]):
        smoothed[:, j] = pd.Series(values[:, j]).rolling(
            window=smoothing_window,
            center=True,
            min_periods=1,
        ).median().to_numpy()
    return smoothed


def sort_ring_dataframe(
    ring_df: pd.DataFrame,
    sort_candidates: List[str] = None,
) -> pd.DataFrame:
    candidates = list(sort_candidates or SORT_CANDIDATE_COLUMNS)
    sort_col = next((column for column in candidates if column in ring_df.columns), None)
    if sort_col is None:
        return ring_df.reset_index(drop=True)
    ring_df = ring_df.copy()
    if sort_col != "POINT_NO":
        ring_df[sort_col] = pd.to_datetime(ring_df[sort_col], errors="coerce")
    return ring_df.sort_values(by=sort_col).reset_index(drop=True)


def build_ring_window_rows(
    ring_df: pd.DataFrame,
    features: List[str],
    window_points: int,
    window_stride: int,
    smoothing_window: int = 0,
    sort_candidates: List[str] = None,
    timestamp_field: str = "timestamp",
    allow_partial_window: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
    ring_df = sort_ring_dataframe(ring_df, sort_candidates=sort_candidates)
    if len(ring_df) < window_points:
        if not allow_partial_window or len(ring_df) <= 0:
            return [], [], []
        values, mask = feature_masked_matrix(ring_df, features)
        values = smooth_values(values, smoothing_window)
        packed_values = np.zeros((window_points, len(features)), dtype=float)
        packed_mask = np.zeros((window_points, len(features)), dtype=bool)
        point_count = min(len(ring_df), window_points)
        packed_values[-point_count:, :] = values[-point_count:, :]
        packed_mask[-point_count:, :] = mask[-point_count:, :]
        try:
            end_time = ring_df.iloc[-1].get(timestamp_field, "-")
        except Exception:
            end_time = "-"
        row = np.concatenate([packed_values[:, j] for j in range(len(features))]).astype(float)
        row_mask = np.concatenate([packed_mask[:, j] for j in range(len(features))]).astype(bool)
        return [row], [row_mask], [end_time]
    values, mask = feature_masked_matrix(ring_df, features)
    values = smooth_values(values, smoothing_window)
    rows: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    end_times: List[Any] = []
    for start in range(0, len(ring_df) - window_points + 1, window_stride):
        end = start + window_points
        rows.append(np.concatenate([values[start:end, j] for j in range(len(features))]).astype(float))
        masks.append(np.concatenate([mask[start:end, j] for j in range(len(features))]).astype(bool))
        try:
            end_times.append(ring_df.iloc[end - 1].get(timestamp_field, "-"))
        except Exception:
            end_times.append("-")
    return rows, masks, end_times


def build_multi_ring_sequences(
    df: pd.DataFrame,
    features: List[str],
    normalize_fn,
    sequence_length: int,
    window_points: int,
    window_stride: int,
    rings_per_sequence: int,
    smoothing_window: int = 0,
    ring_column: str = "RING",
    timestamp_field: str = "timestamp",
    sort_candidates: List[str] = None,
    rings: List[Any] = None,
    sequence_stride: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any], List[Any]]:
    def _ring_key(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return x

    candidate_rings = rings if rings is not None else df[ring_column].dropna().unique().tolist()
    ordered_rings = sorted(candidate_rings, key=_ring_key)
    latest_ring = ordered_rings[-1] if ordered_rings else None

    ring_window_cache: Dict[Any, Tuple[List[np.ndarray], List[np.ndarray], List[Any]]] = {}
    for ring in ordered_rings:
        ring_df = df[df[ring_column] == ring].copy()
        ring_rows, ring_masks, end_times = build_ring_window_rows(
            ring_df,
            features=features,
            window_points=window_points,
            window_stride=window_stride,
            smoothing_window=smoothing_window,
            sort_candidates=sort_candidates,
            timestamp_field=timestamp_field,
            allow_partial_window=(ring != latest_ring),
        )
        if ring_rows:
            ring_window_cache[ring] = (ring_rows, ring_masks, end_times)

    usable_rings = [ring for ring in ordered_rings if ring in ring_window_cache]
    if len(usable_rings) < rings_per_sequence:
        return [], [], [], []

    sequences: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    window_to_ring: List[Any] = []
    window_to_time: List[Any] = []

    stride = max(1, int(sequence_stride or 1))
    for start_idx in range(0, len(usable_rings) - rings_per_sequence + 1, stride):
        group = usable_rings[start_idx:start_idx + rings_per_sequence]
        ring_rows_by_ring: List[List[np.ndarray]] = []
        ring_masks_by_ring: List[List[np.ndarray]] = []
        ring_times_by_ring: List[List[Any]] = []
        for ring in group:
            ring_rows, ring_masks, end_times = ring_window_cache[ring]
            ring_rows_by_ring.append(ring_rows)
            ring_masks_by_ring.append(ring_masks)
            ring_times_by_ring.append(end_times)

        if group[-1] == latest_ring:
            aligned_window_count = len(ring_rows_by_ring[-1])
        else:
            aligned_window_count = min(len(ring_rows) for ring_rows in ring_rows_by_ring)
        if aligned_window_count <= 0:
            continue
        for window_idx in range(aligned_window_count):
            feature_matrix = np.vstack([
                ring_rows[min(window_idx, len(ring_rows) - 1)]
                for ring_rows in ring_rows_by_ring
            ])
            feature_mask = np.vstack([
                ring_masks[min(window_idx, len(ring_masks) - 1)]
                for ring_masks in ring_masks_by_ring
            ])
            normalized = normalize_fn(feature_matrix, feature_mask)
            if normalized.shape[0] != sequence_length:
                continue
            sequences.append(normalized.astype(np.float32))
            masks.append(feature_mask.astype(bool))
            window_to_ring.append(group[-1])
            window_to_time.append(ring_times_by_ring[-1][window_idx])

    return sequences, masks, window_to_ring, window_to_time


def normalize_with_feature_mask(
    data: np.ndarray,
    feature_mask: np.ndarray,
    scalers: List[Dict[str, float]],
    clip_zscore: float = 0.0,
) -> np.ndarray:
    if not scalers or len(scalers) != data.shape[1]:
        raise ValueError("missing or mismatched global_window scalers")
    normalized = data.copy()
    for i, scaler in enumerate(scalers):
        col_mask = feature_mask[:, i]
        median = float(scaler.get("median", 0.0))
        mad = float(scaler.get("mad", 1.0)) or 1.0
        transformed = (data[:, i] - median) / mad
        if clip_zscore > 0:
            transformed = np.clip(transformed, -clip_zscore, clip_zscore)
        normalized[col_mask, i] = transformed[col_mask]
        normalized[~col_mask, i] = 0.0
    return normalized


def minmax(a: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(a)), float(np.max(a))
    if mx - mn < 1e-8:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def scale_by_bounds(values: np.ndarray, bounds: Dict[str, Any]) -> np.ndarray:
    low = float(bounds.get("low", 0.0)) if isinstance(bounds, dict) else 0.0
    high = float(bounds.get("high", 1.0)) if isinstance(bounds, dict) else 1.0
    denom = high - low
    if denom <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - low) / denom, 0.0, 1.0)


def calibrated_combined_risk(
    ae_error: np.ndarray,
    if_anomaly: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    calibration = config.get("risk_calibration", {}) if isinstance(config, dict) else {}
    if calibration.get("enabled") and calibration.get("ae_error_bounds") and calibration.get("if_anomaly_bounds"):
        norm_err = scale_by_bounds(ae_error, calibration.get("ae_error_bounds", {}))
        norm_if = scale_by_bounds(if_anomaly, calibration.get("if_anomaly_bounds", {}))
    else:
        norm_err = minmax(ae_error)
        norm_if = minmax(if_anomaly)
    rf = config.get("risk_fusion", {}) if isinstance(config, dict) else {}
    ae_w = float(rf.get("ae_weight", 0.5))
    if_w = float(rf.get("if_weight", 0.5))
    total = ae_w + if_w
    if total <= 0:
        ae_w, if_w, total = 0.5, 0.5, 1.0
    return (ae_w / total) * norm_err + (if_w / total) * norm_if


def risk_thresholds(config: Dict[str, Any]) -> Dict[str, float]:
    calibration = config.get("risk_calibration", {}) if isinstance(config, dict) else {}
    if calibration.get("enabled") is not True or calibration.get("method") != "kde":
        raise RuntimeError("结泥饼模型必须使用训练阶段保存的 KDE 风险阈值")
    thresholds = calibration.get("risk_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {"low", "medium", "high"}:
        raise RuntimeError("KDE 风险阈值必须且只能包含 low/medium/high")
    values = {name: float(thresholds[name]) for name in ("low", "medium", "high")}
    if not all(np.isfinite(value) for value in values.values()):
        raise RuntimeError("KDE 风险阈值包含非有限数值")
    if not (0.0 <= values["low"] < values["medium"] < values["high"] <= 1.0):
        raise RuntimeError("KDE 风险阈值必须在 [0,1] 内严格递增")
    metadata = calibration.get("kde_threshold_metadata", {})
    if metadata.get("threshold_derivation") != "inverse of bounded Gaussian KDE CDF; no threshold override":
        raise RuntimeError("KDE 风险阈值来源校验失败")
    return values


def risk_level_from_score(score: float, config: Dict[str, Any]) -> str:
    thresholds = risk_thresholds(config)
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["medium"]:
        return "medium"
    if score >= thresholds["low"]:
        return "low"
    return "no_risk"


def load_autoencoder_weights(model: tf.keras.Model, weights_path: Path) -> None:
    """Load current or legacy HDF5 weights without changing trained values."""
    try:
        model.load_weights(weights_path)
        return
    except ValueError as current_format_error:
        try:
            import h5py
            from keras.src.legacy.saving import legacy_h5_format

            with h5py.File(weights_path, "r") as h5_file:
                if "layer_names" not in h5_file.attrs:
                    raise current_format_error
                legacy_h5_format.load_weights_from_hdf5_group(h5_file, model)
        except Exception:
            raise current_format_error


def flat_feature_weights(features: List[str], weight_map: Dict[str, float], window_points: int) -> np.ndarray:
    weights = np.array([float(weight_map.get(feature, 1.0)) for feature in features], dtype=np.float32)
    return np.repeat(weights, window_points).reshape(1, -1)


def trend_vector(
    seq: np.ndarray,
    mask: np.ndarray,
    features: List[str],
    weight_map: Dict[str, float],
    window_points: int,
    use_interactions: bool = True,
) -> np.ndarray:
    feature_weights = np.array([float(weight_map.get(feature, 1.0)) for feature in features], dtype=np.float32)
    out: List[float] = []
    for row, mask_row in zip(seq, mask):
        slopes: Dict[str, float] = {}
        deltas: Dict[str, float] = {}
        for j, feature in enumerate(features):
            base = j * window_points
            seg = row[base:base + window_points]
            seg_mask = mask_row[base:base + window_points]
            valid_idx = np.where(seg_mask)[0]
            weight = float(feature_weights[j])
            if valid_idx.size >= 3:
                x = valid_idx.astype(float)
                y = seg[valid_idx].astype(float)
                x_mean = float(np.mean(x))
                y_mean = float(np.mean(y))
                denom = float(np.sum((x - x_mean) ** 2))
                slope = (float(np.sum((x - x_mean) * (y - y_mean))) / denom) * weight if denom > 1e-12 else 0.0
                delta = float(y[-1] - y[0]) * weight
                curvature = 0.0
                std = float(np.std(y)) * weight
            else:
                slope = delta = curvature = std = 0.0
            slopes[feature] = slope
            deltas[feature] = delta
            out.extend([slope, delta, curvature, std])
        if use_interactions:
            for a, fa in enumerate(features):
                for b in range(a + 1, len(features)):
                    fb = features[b]
                    pair_weight = float(feature_weights[a] * feature_weights[b])
                    out.extend([
                        pair_weight * slopes.get(fa, 0.0) * slopes.get(fb, 0.0),
                        pair_weight * deltas.get(fa, 0.0) * deltas.get(fb, 0.0),
                    ])
    return np.array(out, dtype=float)


UNIFIED_REQUIRED_PARAMS = {
    "Thrust.Spd": "推进速度",
    "CH.Spd": "刀盘转速",
    "CH.Torque": "刀盘扭矩",
    "Thrust.Force": "推力",
    "CH.Tot.ContactForce": "刀盘总挤压力",
}

RISK_SPEC = {
    "name": "结泥饼风险",
    "risk_type_label": "结泥饼",
    "full_risk_type": "结泥饼风险",
    "output_key": "mud_cake_risk",
    "fault_cause": "渣土在刀盘面板或土仓内板结硬化，形成阻碍掘进的泥饼，导致掘进效率降低",
    "potential_risk": "结泥饼预警",
    "output_fields": TBM_FEATURES,
    "fields": TBM_FEATURES,
    "map": {
        "Thrust.Spd": "推进速度",
        "CH.Spd": "刀盘转速",
        "CH.Torque": "刀盘扭矩",
        "Thrust.Force": "推力",
        "CH.Tot.ContactForce": "刀盘总挤压力",
        "Cluster_Label": "地层类别",
    },
    "units": {
        "刀盘转速": "rpm",
        "刀盘扭矩": "kNm",
        "推进速度": "mm/min",
        "推力": "kN",
        "刀盘总挤压力": "kN",
        "地层类别": "",
    },
    "score_points": [(0.0, 4.0), (0.3, 3.5), (0.65, 2.0), (0.9, 0.5), (1.0, 0.0)],
    "probability_thresholds": (0.65, 0.3),
    "fault_reason_analysis": {
        "无风险Ⅰ": "刀盘扭矩与刀盘转速匹配良好，推力与推进速度稳定，贯入度无异常波动。渣土含水率与改良剂比例处于工艺目标，仓壁与面板未见黏附或板结痕迹，排泥连续顺畅，不具备泥饼形成的物理条件。",
        "低风险Ⅱ": "观察到刀盘扭矩轻微抬升、推进速度小幅下降，推力与贯入度边界化波动，反映土体塑性增大与局部摩阻上升。面板可能出现初始黏附点，渣粒分布偏粗或含水率偏离最佳区间，短时排泥不均。若该状态持续，将促使黏附向板结演化，需要关注改良剂与水分的微调空间。",
        "中风险Ⅲ": "刀盘扭矩与推力持续偏高，推进速度显著下降且波动加大，贯入度呈非线性起伏，指示面板与土仓内已有黏附/板结积聚。排泥阻力增大、管路压降上升、回流概率增加，渣土流变性恶化。成因多与高粘性土、改良不足或含水率偏低/偏高叠加导致黏结力提升有关，若不干预泥饼将沿面板扩展并破坏掘进稳定性与效率。",
        "高风险Ⅳ": "面板与仓内泥饼大面积形成，刀盘扭矩异常抬升且伴随振动，推进速度显著受限或间断，推力异常维持，排泥系统出现明显阻塞。进一步将导致驱动负荷过高、温升增大、密封与轴承受污染风险上升，存在设备损伤与渗水次生风险，应在可控前提下尽快处置以恢复切削与排泥通道。",
    },
    "measures": {
        "无风险Ⅰ": {
            "measures": ["维持状态：保持当前参数", "正常作业：按标准流程进行"],
            "reason": "推进速度与贯入度匹配良好，扭矩与推进力保持稳定，土体含水率适宜，渣土流动性良好，无结泥饼风险。",
        },
        "低风险Ⅱ": {
            "measures": ["维持参数：保持当前掘进参数", "监控刀盘扭矩：关注扭矩变化趋势"],
            "reason": "刀盘扭矩略有波动，推进速度轻微下降，贯入度变化不大，土体粘性略有增加，但渣土排出仍然顺畅，设备负荷在安全范围内。",
        },
        "中风险Ⅲ": {
            "measures": ["监控刀盘扭矩：每5分钟记录一次刀盘扭矩值", "增加添加剂：将添加剂浓度提高20-30%"],
            "reason": "扭矩明显上升，推进速度下降，贯入度异常，推进力增大，土体粘性增大，渣土含水率降低，出现结泥饼初期征兆。",
        },
        "高风险Ⅳ": {
            "measures": ["立即停机：停止掘进作业", "清理泥饼：使用高压水枪冲洗"],
            "reason": "扭矩急剧上升至警戒值，推进速度显著下降，贯入度严重异常，推进力达到极限，渣土呈干硬状态且排出困难，设备出现卡滞现象。",
        },
    },
}


class UnsupervisedMudCakeDetector:

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.device = self._select_device()
        self.autoencoder = None
        self.isolation_forest = None
        self.scalers = {}
        self.robust_scalers = {}
        self.model_info = {}
        self.training_data_df = None
        self.model_dir = RISK_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            'sequence_length': 6,
            'feature_selection': {
                'min_valid_ratio': 0.05,
                'min_variance': 1e-6,
            },
            'feature_weights': {
                'Thrust.Spd': 1.5,
                'CH.Tot.ContactForce': 1.5,
                'Thrust.Force': 1.5,
                'CH.Spd': 0.7,
                'CH.Torque': 0.7,
            },
            'risk_fusion': {
                'ae_weight': 0.35,
                'if_weight': 0.65,
            },
            'data_processing': {
                'sampling_ratio': 0.8,
                'random_seed': 42,
                'val_ratio': 0.2,
                'rings_per_sequence': 6,
                'window_points': 48,
                'window_stride': 8,
                'clip_zscore': 3.5,
            },
            'autoencoder_params': {
                'hidden_size': 16,
                'num_layers': 1,
                'dropout': 0.2,
                'learning_rate': 0.0005,
                'batch_size': 32,
                'epochs': 30,
                'patience': 8,
                'l2': 1e-5,
            },
            'isolation_forest_params': {
                'contamination': 0.04,
                'random_state': 42,
                'n_estimators': 100,
                'max_fit_samples': 50000,
                'use_interactions': False
            },
        }

    def _select_device(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return 'GPU'
            except Exception as e:
                return 'CPU'
        else:
            return 'CPU'


    def _normalize_with_feature_mask(self, data: np.ndarray, feature_mask: np.ndarray, data_source: str) -> np.ndarray:
        d = data.shape[1]
        zmax = float(self.config.get('data_processing', {}).get('clip_zscore', 0) or 0)
        # 优先使用鲁棒尺度（形如 self.robust_scalers['global_window'][i] = {'median': m, 'mad': s}）
        if data_source in self.robust_scalers and isinstance(self.robust_scalers[data_source], list) and len(self.robust_scalers[data_source]) == d:
            return normalize_with_feature_mask(data, feature_mask, self.robust_scalers[data_source], zmax)
        # 回退到标准化器路径（兼容旧模型）
        if data_source not in self.scalers or not isinstance(self.scalers[data_source], list) or len(self.scalers[data_source]) != d:
            self.scalers[data_source] = [StandardScaler() for _ in range(d)]
            for i in range(d):
                col_mask = feature_mask[:, i]
                values = data[col_mask, i]
                if values.size > 0:
                    self.scalers[data_source][i].fit(values.reshape(-1, 1))
                else:
                    self.scalers[data_source][i].fit(np.zeros((10, 1)))
        normalized = data.copy()
        for i in range(d):
            col_mask = feature_mask[:, i]
            if np.any(col_mask):
                col = data[:, i:i + 1]
                transformed = self.scalers[data_source][i].transform(col)[:, 0]
                if zmax > 0:
                    transformed = np.clip(transformed, -zmax, zmax)
                normalized[col_mask, i] = transformed[col_mask]
                normalized[~col_mask, i] = 0.0
        return normalized
    # 废弃：训练与推理不再使用基线/中心化，相关方法移除

    

    

    def load_models(self) -> bool:
        try:
            model_info_path = self.model_dir / MUD_CAKE_MODEL_INFO_FILENAME
            weights_path = self.model_dir / MUD_CAKE_AUTOENCODER_FILENAME
            isolation_forest_path = self.model_dir / MUD_CAKE_ISOLATION_FOREST_FILENAME
            scalers_path = self.model_dir / MUD_CAKE_SCALERS_FILENAME
            required_paths = [
                model_info_path,
                weights_path,
                isolation_forest_path,
                scalers_path,
            ]
            if any(not path.exists() for path in required_paths):
                return False
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            self.model_info = model_info
            self.config = model_info['config']
            # 模型加载时立即校验训练产物中的 KDE 阈值；禁止推理使用默认值或另一套阈值。
            risk_thresholds(self.config)
            self.all_features = model_info['features']
            self.feature_dim = model_info['feature_dim']
            ae_params = self.config['autoencoder_params']
            self.autoencoder = AutoEncoder(
                input_size=self.feature_dim,
                hidden_size=ae_params['hidden_size'],
                num_layers=ae_params['num_layers'],
                dropout=ae_params['dropout'],
                l2=float(ae_params.get('l2', 0.0)),
            )
            dummy = tf.zeros((1, int(self.config['sequence_length']), int(self.feature_dim)))
            _ = self.autoencoder(dummy, training=False)
            load_autoencoder_weights(self.autoencoder, weights_path)
            self.isolation_forest = joblib.load(isolation_forest_path)
            scalers_pack = joblib.load(scalers_path)
            if not isinstance(scalers_pack, dict):
                raise ValueError("mud_cake_scalers.pkl 格式错误，必须为 dict")
            self.robust_scalers = scalers_pack.get('robust', {})
            self.scalers = scalers_pack.get('standard', {})
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False


class MudCakeRiskCalculator:

    def __init__(self, model_dir: str = None):
        self.detector = UnsupervisedMudCakeDetector()
        if model_dir is not None:
            self.detector.model_dir = Path(model_dir)
        # 尝试加载模型；加载失败时仍允许实例化，调用方自行处理
        try:
            self.detector.load_models()
        except Exception:
            pass
        self.model_info = getattr(self.detector, "model_info", {})
        self.stratum_label = self.model_info.get("stratum_label")

    def _aggregate_values(self, values: List[float], mode: str, **kwargs) -> float:
        try:
            if not values:
                return 0.0
            arr = np.array(values, dtype=float)
            if mode == 'max':
                return float(np.max(arr))
            if mode == 'mean':
                return float(np.mean(arr))
            if mode == 'median':
                return float(np.median(arr))
            if mode == 'pquantile':
                q = float(kwargs.get('quantile_p', 0.95))
                q = max(0.0, min(1.0, q))
                return float(np.quantile(arr, q))
            if mode == 'topk_mean':
                k = int(kwargs.get('topk_k', 3) or 3)
                k = max(1, min(k, arr.size))
                topk = np.sort(arr)[-k:]
                return float(np.mean(topk))
            # 默认采用稳健的中位数，避免少数窗口尖峰导致误判
            return float(np.median(arr))
        except Exception:
            # 任何异常回退为均值
            try:
                return float(np.mean(np.array(values, dtype=float)))
            except Exception:
                return 0.0

    def calculate_ring_risk_sequence(self, sequence_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if not isinstance(sequence_points, list) or not sequence_points:
                return {'status': 'error', 'message': '输入序列为空'}

            # 提取特征与元信息
            all_feats = list(self.detector.all_features)
            seq_len = int(self.detector.config['sequence_length'])
            window_points = int(self.detector.config['data_processing']['window_points'])
            window_stride = int(self.detector.config['data_processing']['window_stride'])
            rings_per_seq = int(self.detector.config['data_processing']['rings_per_sequence'])
            sequence_stride = int(self.detector.config.get('data_processing', {}).get('sequence_stride', 1) or 1)
            smoothing_window = int(self.detector.config.get('data_processing', {}).get('smoothing_window', 0) or 0)

            # 转为 DataFrame
            rows = []
            for p in sequence_points:
                prop = p.get('property', {}) if isinstance(p, dict) else {}
                ring = p.get('ring', p.get('RING', 0))
                ts = p.get('timestamp', p.get('ts(Asia/Shanghai)', p.get('time', None)))
                row = {'RING': ring, 'timestamp': ts, 'data_source': 'unknown'}
                raw_feature_values = {}
                if isinstance(p, dict):
                    raw_feature_values.update({k: v for k, v in p.items() if k not in {'property'}})
                if isinstance(prop, dict):
                    raw_feature_values.update(prop)
                add_engineered_features_to_row(raw_feature_values)
                for f in all_feats:
                    v = raw_feature_values.get(f)
                    try:
                        row[f] = float(v) if v is not None and not pd.isna(v) else (0.0 if f in GEO_FEATURES else np.nan)
                    except Exception:
                        row[f] = 0.0 if f in GEO_FEATURES else np.nan
                rows.append(row)
            df = pd.DataFrame(rows)
            rings = sorted(df['RING'].dropna().unique(), key=lambda x: int(float(x)) if str(x).strip() else x)
            if len(rings) < rings_per_seq:
                return {'status': 'error', 'message': '可用环数量不足以构建序列'}
            sequences, masks, window_to_ring, window_to_time = build_multi_ring_sequences(
                df,
                features=all_feats,
                normalize_fn=lambda feature_matrix, feature_mask: self.detector._normalize_with_feature_mask(
                    feature_matrix, feature_mask, 'global_window'
                ),
                sequence_length=seq_len,
                window_points=window_points,
                window_stride=window_stride,
                rings_per_sequence=rings_per_seq,
                smoothing_window=smoothing_window,
                ring_column='RING',
                timestamp_field='timestamp',
                sort_candidates=SORT_CANDIDATE_COLUMNS,
                rings=rings,
                sequence_stride=sequence_stride,
            )

            if not sequences:
                return {'status': 'error', 'message': '无法构建有效序列'}

            X = np.array(sequences, dtype=np.float32)
            M = np.array(masks, dtype=np.float32)
            reconstructed = self.detector.autoencoder(X, training=False).numpy()

            # 加权掩码误差
            fw_cfg = self.detector.config.get('feature_weights', {})
            flat_w = flat_feature_weights(all_feats, fw_cfg, window_points)
            err = np.mean(((X - reconstructed) ** 2) * (M * flat_w.reshape(1, 1, -1)), axis=(1, 2))

            use_inter = bool(self.detector.config.get('isolation_forest_params', {}).get('use_interactions', True))
            trend_vectors = [
                trend_vector(s, m, all_feats, fw_cfg, window_points, use_inter)
                for s, m in zip(sequences, masks)
            ]
            scores = self.detector.isolation_forest.decision_function(trend_vectors)
            calibrated_risk = calibrated_combined_risk(err, -scores, self.detector.config)
            thresholds = risk_thresholds(self.detector.config)
            norm_err = minmax(err)
            norm_if = minmax(-scores)
            rf = self.detector.config.get('risk_fusion', {})
            ae_w = float(rf.get('ae_weight', 0.5))
            if_w = float(rf.get('if_weight', 0.5))
            ssum = ae_w + if_w
            if ssum <= 0:
                ae_w, if_w = 0.5, 0.5
                ssum = 1.0
            ae_w /= ssum
            if_w /= ssum
            combined = calibrated_risk
            # 趋势门控逻辑已删除：直接使用融合分作为风险值

            # 段落和最早触发均使用当前地层模型训练保存的 KDE 阈值。
            segments = []
            in_seg = False
            seg_start = 0
            earliest_index = None
            for i, v in enumerate(combined):
                if earliest_index is None and v >= thresholds["low"]:
                    earliest_index = i
                if v >= thresholds["high"] and not in_seg:
                    in_seg = True
                    seg_start = i
                elif v < thresholds["high"] and in_seg:
                    in_seg = False
                    segments.append({'start_index': seg_start, 'end_index': i - 1})
            if in_seg:
                segments.append({'start_index': seg_start, 'end_index': len(combined) - 1})

            # 每环综合风险（取该环所有窗口风险的稳健聚合）
            def _rk(x):
                try:
                    return int(x)
                except Exception:
                    try:
                        return int(float(x))
                    except Exception:
                        return x
            ring_set = sorted(set(window_to_ring), key=_rk)
            per_ring = []
            agg_cfg = self.detector.config.get('aggregation', {}) if hasattr(self.detector, 'config') else {}
            per_ring_mode = str(agg_cfg.get('per_ring_mode', 'median'))
            final_ring_mode = str(agg_cfg.get('final_ring_mode', per_ring_mode))
            topk_k = int(agg_cfg.get('topk_k', 3) or 3)
            quantile_p = float(agg_cfg.get('quantile_p', 0.95) or 0.95)
            for r in ring_set:
                idxs = [i for i, rr in enumerate(window_to_ring) if rr == r]
                if not idxs:
                    continue
                vals = [float(combined[i]) for i in idxs]
                ring_risk = self._aggregate_values(vals, per_ring_mode, topk_k=topk_k, quantile_p=quantile_p)
                level = risk_level_from_score(ring_risk, self.detector.config)
                try:
                    ring_int = int(r)
                except Exception:
                    try:
                        ring_int = int(float(r))
                    except Exception:
                        ring_int = r
                per_ring.append({'ring': ring_int, 'combined_risk': ring_risk, 'risk_level': level})

            # 最后一环结果：取“最后一环所有窗口”的稳健聚合（默认中位数），体现序列整体对末环的影响
            last_ring = ring_set[-1] if ring_set else None
            final_risk = 0.0
            final_level = 'no_risk'
            if last_ring is not None:
                # 仅在确有以最后一环为结束环的窗口时计算；否则视为序列与环映射异常
                last_indices = [i for i, rr in enumerate(window_to_ring) if (int(rr) if isinstance(rr, (int, float, str)) else rr) == (int(last_ring) if isinstance(last_ring, (int, float, str)) else last_ring)]
                if not last_indices:
                    raise RuntimeError("无法定位最后一环的窗口，序列与环映射异常")
                final_vals = [float(combined[i]) for i in last_indices]
                final_risk = self._aggregate_values(final_vals, final_ring_mode, topk_k=topk_k, quantile_p=quantile_p)
                final_level = risk_level_from_score(final_risk, self.detector.config)
            earliest_time = '-'
            earliest_ring = (int(last_ring) if last_ring is not None else None)
            if last_ring is not None:
                # 仅考虑最后一环的窗口；若转换异常则抛错，由上层捕获为评估失败
                for i, (ring_i, val_i) in enumerate(zip(window_to_ring, combined)):
                    if (int(ring_i) if isinstance(ring_i, (int, float, str)) else ring_i) == int(last_ring) and val_i >= thresholds["low"]:
                        tt = window_to_time[i] if i < len(window_to_time) else '-'
                        earliest_time = tt if tt else '-'
                        break

            return {
                'status': 'success',
                'combined_risk': final_risk,
                'risk_level': final_level,
                'segments': segments,
                'earliest_time': earliest_time,
                'earliest_ring': (int(last_ring) if last_ring is not None else None),
                'per_ring': per_ring,
            }
        except Exception as e:
            return {'status': 'error', 'message': f'序列评估失败: {str(e)}'}


class StratumMudCakeRiskCalculator:
    def __init__(self, model_root: str = None):
        self.model_root = Path(model_root) if model_root is not None else RISK_MODEL_DIR / "mud_cake_by_stratum"
        self.calculators: Dict[str, MudCakeRiskCalculator] = {}
        self.model_info: Dict[str, Any] = {
            "training_mode": "stratum_specific",
            "selector_feature": "Cluster_Label",
            "strata": {},
        }
        self.all_features: List[str] = []
        self.load_models()

    def load_models(self) -> bool:
        if not self.model_root.exists():
            return False
        tf.keras.backend.clear_session()
        for model_dir in sorted(self.model_root.glob("cluster_*")):
            if not model_dir.is_dir():
                continue
            label = model_dir.name.replace("cluster_", "", 1)
            required = [
                model_dir / MUD_CAKE_MODEL_INFO_FILENAME,
                model_dir / MUD_CAKE_AUTOENCODER_FILENAME,
                model_dir / MUD_CAKE_ISOLATION_FOREST_FILENAME,
                model_dir / MUD_CAKE_SCALERS_FILENAME,
            ]
            if any(not path.exists() for path in required):
                continue
            calc = MudCakeRiskCalculator(model_dir=str(model_dir))
            if calc.detector.autoencoder is None or calc.detector.isolation_forest is None:
                continue
            features = list(getattr(calc.detector, "all_features", []) or [])
            label = stratum_key(calc.stratum_label or label)
            self.calculators[label] = calc
            self.model_info["strata"][label] = {
                "model_dir": str(model_dir),
                "features": features,
                "feature_count": len(features),
            }
            for feature in features:
                if feature not in self.all_features:
                    self.all_features.append(feature)
        return bool(self.calculators)

    def available_strata(self) -> List[str]:
        return sorted(self.calculators.keys())

    def calculator_for_label(self, label: Any) -> Tuple[str, MudCakeRiskCalculator]:
        normalized = stratum_key(label)
        if not normalized:
            raise RuntimeError("结泥饼分地层模型推理缺少 Cluster_Label")
        calculator = self.calculators.get(normalized)
        if calculator is None:
            raise RuntimeError(f"未找到 Cluster_Label={normalized} 对应的结泥饼模型")
        return normalized, calculator

    def calculate_ring_risk_sequence_for_label(self, sequence_points: List[Dict[str, Any]], label: Any) -> Dict[str, Any]:
        normalized, calculator = self.calculator_for_label(label)
        result = calculator.calculate_ring_risk_sequence(sequence_points)
        if isinstance(result, dict):
            result["stratum_label"] = normalized
            result["training_mode"] = "stratum_specific"
        return result

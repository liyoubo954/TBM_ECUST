import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os
# 抑制TensorFlow的INFO级别日志输出（需在导入TensorFlow前设置）
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
import random


class AutoEncoder(tf.keras.Model):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.3, l2: float = 0.0):
        super().__init__()
        self.input_size = input_size
        reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0.0 else None
        self.encoder = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            return_state=True,
            dropout=dropout if num_layers > 1 else 0.0,
            kernel_regularizer=reg,
            recurrent_regularizer=reg,
            bias_regularizer=None,
        )
        self.decoder = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            dropout=dropout if num_layers > 1 else 0.0,
            kernel_regularizer=reg,
            recurrent_regularizer=reg,
            bias_regularizer=None,
        )
        self.output_dense = tf.keras.layers.Dense(input_size, kernel_regularizer=reg)

    def call(self, x, training=False):
        encoded_seq, h, c = self.encoder(x, training=training)
        decoded_seq = self.decoder(encoded_seq, initial_state=[h, c], training=training)
        reconstructed = self.output_dense(decoded_seq)
        return reconstructed

    def train_step(self, data):
        # 接受 (x, y, mask) 三元组；若未提供 mask 则退化为普通 MSE
        if isinstance(data, (tuple, list)):
            if len(data) == 3:
                x, y, mask = data
            elif len(data) == 2:
                x, y = data
                mask = tf.ones_like(y)
            else:
                x = data
                y = data
                mask = tf.ones_like(y)
        else:
            x = data
            y = data
            mask = tf.ones_like(y)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            mask_f = tf.cast(mask, tf.float32)
            sq_err = tf.square(y - y_pred) * mask_f
            per_sample_sum = tf.reduce_sum(sq_err, axis=[1, 2])
            per_sample_cnt = tf.reduce_sum(mask_f, axis=[1, 2])
            per_sample_cnt = tf.maximum(per_sample_cnt, 1.0)
            loss = tf.reduce_mean(per_sample_sum / per_sample_cnt)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        # 验证/评估阶段的掩码损失计算（不做反向传播）
        if isinstance(data, (tuple, list)):
            if len(data) == 3:
                x, y, mask = data
            elif len(data) == 2:
                x, y = data
                mask = tf.ones_like(y)
            else:
                x = data
                y = data
                mask = tf.ones_like(y)
        else:
            x = data
            y = data
            mask = tf.ones_like(y)

        y_pred = self(x, training=False)
        mask_f = tf.cast(mask, tf.float32)
        sq_err = tf.square(y - y_pred) * mask_f
        per_sample_sum = tf.reduce_sum(sq_err, axis=[1, 2])
        per_sample_cnt = tf.reduce_sum(mask_f, axis=[1, 2])
        per_sample_cnt = tf.maximum(per_sample_cnt, 1.0)
        loss = tf.reduce_mean(per_sample_sum / per_sample_cnt)
        return {"loss": loss}


class UnsupervisedMudCakeDetector:
    """
    无监督结泥饼风险检测器
    核心思想：学习正常工况的数据模式，偏离正常模式即为异常
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.device = self._select_device()
        self.autoencoder = None
        self.isolation_forest = None
        self.scalers = {}
        self.robust_scalers = {}
        self.training_data_df = None
        self.model_dir = Path('app/risk/models')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            'sequence_length': 6,
            'feature_selection': {
                'min_valid_ratio': 0.05,
                'min_variance': 1e-6,
            },
            'feature_weights': {
                'TJSD': 1.5,
                'DP_SS_ZTL': 1.5,
                'TJL': 1.5,
                'DP_SD': 0.7,
                'DP_ZJ': 0.7,
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
        """基于MAD的鲁棒归一化；若鲁棒尺度缺失则回退到StandardScaler。"""
        d = data.shape[1]
        zmax = float(self.config.get('data_processing', {}).get('clip_zscore', 0) or 0)
        # 优先使用鲁棒尺度（形如 self.robust_scalers['global_window'][i] = {'median': m, 'mad': s}）
        if data_source in self.robust_scalers and isinstance(self.robust_scalers[data_source], list) and len(self.robust_scalers[data_source]) == d:
            normalized = data.copy()
            for i in range(d):
                col_mask = feature_mask[:, i]
                med = float(self.robust_scalers[data_source][i].get('median', 0.0))
                mad = float(self.robust_scalers[data_source][i].get('mad', 1.0))
                scale = mad if mad > 1e-8 else 1.0
                col = data[:, i]
                transformed = (col - med) / scale
                if zmax > 0:
                    transformed = np.clip(transformed, -zmax, zmax)
                normalized[col_mask, i] = transformed[col_mask]
                normalized[~col_mask, i] = 0.0
            return normalized
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
            model_info_path = self.model_dir / 'mud_cake_model_info.json'
            if not model_info_path.exists():
                return False
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            self.config = model_info['config']
            self.all_features = model_info['features']
            self.feature_dim = model_info['feature_dim']
            self.autoencoder = AutoEncoder(input_size=self.feature_dim, hidden_size=self.config['autoencoder_params']['hidden_size'], num_layers=self.config['autoencoder_params']['num_layers'], dropout=self.config['autoencoder_params']['dropout'])
            dummy = tf.zeros((1, int(self.config['sequence_length']), int(self.feature_dim)))
            _ = self.autoencoder(dummy, training=False)
            self.autoencoder.load_weights(self.model_dir / 'mud_cake_autoencoder.h5')
            self.isolation_forest = joblib.load(self.model_dir / 'mud_cake_isolation_forest.pkl')
            try:
                scalers_pack = joblib.load(self.model_dir / 'mud_cake_scalers.pkl')
                if isinstance(scalers_pack, dict):
                    self.robust_scalers = scalers_pack.get('robust', {})
                    self.scalers = scalers_pack.get('standard', {})
                else:
                    # 兼容旧模型：仅标准化器
                    self.scalers = scalers_pack
            except Exception:
                self.robust_scalers = {}
                self.scalers = {}
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False


class MudCakeRiskCalculator:
    """
    兼容旧接口的风险计算器包装类。
    说明：为了解决外部模块（如 routes.py）在导入时依赖该类而导致的 ImportError，
    这里提供一个包装实现，内部复用 UnsupervisedMudCakeDetector 的模型与配置，
    并实现环对序列评估接口，以便 evaluate.py 和路由在导入阶段不再报错。
    """

    def __init__(self, model_type: str = 'unified', model_dir: str = None):
        self.model_type = model_type
        self.detector = UnsupervisedMudCakeDetector()
        if model_dir is not None:
            self.detector.model_dir = Path(model_dir)
        # 尝试加载模型；加载失败时仍允许实例化，调用方自行处理
        try:
            self.detector.load_models()
        except Exception:
            pass

    def _minmax(self, a: np.ndarray) -> np.ndarray:
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-8:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

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

    def calculate_ring_pair_risk_series(self, ring_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        兼容旧接口：将每环的聚合特征映射为“环内滑窗展平”形态，与训练完全一致。
        做法：把每环的特征值在窗口长度内重复，形成 len(all_features)*window_points 的向量，seq_len=1。
        """
        try:
            if not isinstance(ring_features, list) or not ring_features:
                return {'status': 'error', 'message': '输入环特征为空'}
            window_points = int(self.detector.config['data_processing']['window_points'])
            seq_len = int(self.detector.config['sequence_length'])
            all_feats = list(self.detector.all_features)
            # 构造单窗口序列（每环一个），并按环号排序
            records = []
            for item in ring_features:
                ring_id = item.get('ring', item.get('RING', 0))
                vec_vals = []
                vec_mask = []
                for f in all_feats:
                    v = item.get(f, np.nan)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        vec_vals.extend([0.0] * window_points)
                        vec_mask.extend([False] * window_points)
                    else:
                        val = float(v)
                        vec_vals.extend([val] * window_points)
                        vec_mask.extend([True] * window_points)
                records.append((ring_id, np.array(vec_vals, dtype=float), np.array(vec_mask, dtype=bool)))
            # 排序
            def _ring_key(x):
                try:
                    return int(x)
                except Exception:
                    try:
                        return int(float(x))
                    except Exception:
                        return x
            records.sort(key=lambda r: _ring_key(r[0]))
            if not records:
                return {'status': 'success', 'per_ring': [], 'segments': []}
            feature_matrix = np.vstack([r[1] for r in records])
            feature_mask = np.vstack([r[2] for r in records])
            normalized = self.detector._normalize_with_feature_mask(feature_matrix, feature_mask, 'global_window')
            sequences = []
            masks = []
            window_to_ring = []
            n = normalized.shape[0]
            for i in range(0, max(0, n - seq_len + 1)):
                sequences.append(normalized[i:i + seq_len])
                masks.append(feature_mask[i:i + seq_len])
                window_to_ring.append(records[i + seq_len - 1][0])
            if not sequences:
                return {'status': 'success', 'per_ring': [], 'segments': []}
            X = np.array(sequences, dtype=np.float32)
            M = np.array(masks, dtype=np.float32)
            reconstructed = self.detector.autoencoder(X, training=False).numpy()
            # 加权掩码：与训练一致按特征权重展开至窗口维度
            fw_cfg = self.detector.config.get('feature_weights', {})
            fw_map = {f: float(fw_cfg.get(f, 1.0)) for f in all_feats}
            flat_w = np.concatenate([np.full(window_points, fw_map[f], dtype=np.float32) for f in all_feats])
            err = np.mean(((X - reconstructed) ** 2) * (M * flat_w.reshape(1, 1, -1)), axis=(1, 2))
            flattened = [(s * m).flatten() for s, m in zip(sequences, masks)]
            scores = self.detector.isolation_forest.decision_function(flattened)
            norm_err = self._minmax(err)
            norm_if = self._minmax(-scores)
            # 与训练/评估保持一致：融合权重从配置读取，默认 0.5/0.5
            rf = self.detector.config.get('risk_fusion', {}) if hasattr(self.detector, 'config') else {}
            ae_w = float(rf.get('ae_weight', 0.5))
            if_w = float(rf.get('if_weight', 0.5))
            # 归一到 1，防止调用方只设置其中一个权重
            s = ae_w + if_w
            if s <= 0:
                ae_w, if_w = 0.5, 0.5
                s = 1.0
            ae_w /= s
            if_w /= s
            combined = ae_w * norm_err + if_w * norm_if
            # 高风险段（threshold=0.9）
            segments = []
            in_seg = False
            seg_start = 0
            for i, v in enumerate(combined):
                if v >= 0.9 and not in_seg:
                    in_seg = True
                    seg_start = i
                elif v < 0.9 and in_seg:
                    in_seg = False
                    segments.append({'start_index': seg_start, 'end_index': i - 1})
            if in_seg:
                segments.append({'start_index': seg_start, 'end_index': len(combined) - 1})
            # 每环综合风险：采用“全窗口稳健聚合”，避免少数窗口尖峰导致误判
            ring_set = sorted(set(window_to_ring), key=_ring_key)
            per_ring = []
            agg_cfg = self.detector.config.get('aggregation', {}) if hasattr(self.detector, 'config') else {}
            per_ring_mode = str(agg_cfg.get('per_ring_mode', 'median'))
            topk_k = int(agg_cfg.get('topk_k', 3) or 3)
            quantile_p = float(agg_cfg.get('quantile_p', 0.95) or 0.95)
            for r in ring_set:
                idxs = [i for i, rr in enumerate(window_to_ring) if rr == r]
                if not idxs:
                    continue
                vals = [float(combined[i]) for i in idxs]
                ring_risk = self._aggregate_values(vals, per_ring_mode, topk_k=topk_k, quantile_p=quantile_p)
                if ring_risk >= 0.9:
                    level = 'high'
                elif ring_risk >= 0.65:
                    level = 'medium'
                elif ring_risk >= 0.3:
                    level = 'low'
                else:
                    level = 'no_risk'
                # 尽量返回整数环号
                try:
                    ring_int = int(r)
                except Exception:
                    try:
                        ring_int = int(float(r))
                    except Exception:
                        ring_int = r
                per_ring.append({'ring': ring_int, 'combined_risk': ring_risk, 'risk_level': level})
            return {'status': 'success', 'per_ring': per_ring, 'segments': segments}
        except Exception as e:
            return {'status': 'error', 'message': f'环对评估失败: {str(e)}'}

    def calculate_ring_risk_sequence(self, sequence_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用连续多环的原始点数据构建滑动窗口与跨窗序列，评估各环风险；
        返回最后一环的综合风险及风险等级，并提供段落与最早触发信息。
        """
        try:
            if not isinstance(sequence_points, list) or not sequence_points:
                return {'status': 'error', 'message': '输入序列为空'}

            # 提取特征与元信息
            all_feats = list(self.detector.all_features)
            sort_candidates = ['ts(Asia/Shanghai)', 'timestamp', 'time', 'TIME', 'Timestamp', 'time_stamp', '采样时间', 'DATETIME', 'Date', 'date', '时间', 'POINT_NO', 'ts']
            seq_len = int(self.detector.config['sequence_length'])
            window_points = int(self.detector.config['data_processing']['window_points'])
            window_stride = int(self.detector.config['data_processing']['window_stride'])
            rings_per_seq = int(self.detector.config['data_processing']['rings_per_sequence'])

            # 转为 DataFrame
            rows = []
            for p in sequence_points:
                prop = p.get('property', {}) if isinstance(p, dict) else {}
                ring = p.get('ring', p.get('RING', 0))
                ts = p.get('timestamp', p.get('ts(Asia/Shanghai)', p.get('time', None)))
                row = {'RING': ring, 'timestamp': ts, 'data_source': 'unknown'}
                for f in all_feats:
                    v = prop.get(f, p.get(f, None))
                    try:
                        row[f] = float(v) if v is not None and not pd.isna(v) else np.nan
                    except Exception:
                        row[f] = np.nan
                rows.append(row)
            df = pd.DataFrame(rows)

            # 按环号排序与分组
            def _ring_key(x):
                try:
                    return int(x)
                except Exception:
                    try:
                        return int(float(x))
                    except Exception:
                        return x
            rings = sorted([r for r in df['RING'].dropna().unique()], key=_ring_key)
            if len(rings) < rings_per_seq:
                return {'status': 'error', 'message': '可用环数量不足以构建序列'}

            sequences = []
            masks = []
            window_to_ring = []
            window_to_time = []  # 每个窗口对应的结束时间戳
            window_ring_list = []
            ring_first_time = {}

            for s in range(0, len(rings) - rings_per_seq + 1):
                group = rings[s:s + rings_per_seq]
                group_rows = []
                group_masks = []
                # 记录每个窗口的所属环号
                for r in group:
                    ring_df = df[df['RING'] == r].copy()
                    sort_col = next((c for c in sort_candidates if c in ring_df.columns), None)
                    if sort_col is not None:
                        if sort_col != 'POINT_NO':
                            try:
                                ring_df[sort_col] = pd.to_datetime(ring_df[sort_col], errors='coerce')
                            except Exception:
                                pass
                        ring_df = ring_df.sort_values(by=sort_col)
                    ring_df = ring_df.reset_index(drop=True)
                    if r not in ring_first_time and len(ring_df) > 0:
                        ring_first_time[r] = ring_df.iloc[0].get('timestamp', '-')
                    n = len(ring_df)
                    if n < window_points:
                        continue
                    num_features = len(all_feats)
                    F = np.zeros((n, num_features), dtype=float)
                    FM = np.zeros((n, num_features), dtype=bool)
                    for j, feature in enumerate(all_feats):
                        vals = pd.to_numeric(ring_df[feature], errors='coerce').to_numpy()
                        mask = ~np.isnan(vals)
                        if not np.all(mask):
                            for i in range(n):
                                if mask[i]:
                                    F[i, j] = vals[i]
                                    FM[i, j] = True
                                else:
                                    F[i, j] = F[i - 1, j] if i > 0 else 0.0
                                    FM[i, j] = False
                        else:
                            F[:, j] = vals
                            FM[:, j] = True
                    sw = int(self.detector.config.get('data_processing', {}).get('smoothing_window', 0) or 0)
                    if sw and sw > 1:
                        for jj in range(num_features):
                            series = pd.Series(F[:, jj])
                            F[:, jj] = series.rolling(window=sw, center=True, min_periods=1).median().to_numpy()
                    for start in range(0, n - window_points + 1, window_stride):
                        end = start + window_points
                        window_vals = []
                        window_mask = []
                        for j in range(num_features):
                            seg = F[start:end, j]
                            seg_m = FM[start:end, j]
                            window_vals.extend(seg.tolist())
                            window_mask.extend(seg_m.tolist())
                        group_rows.append(np.array(window_vals, dtype=float))
                        group_masks.append(np.array(window_mask, dtype=bool))
                        window_ring_list.append(r)
                        # 记录窗口结束点的时间戳（更贴近预警时刻）
                        try:
                            ts_val = ring_df.iloc[end - 1].get('timestamp', '-')
                        except Exception:
                            ts_val = '-'
                        window_to_time.append(ts_val)
                if not group_rows:
                    continue
                feature_matrix = np.vstack(group_rows)
                feature_mask = np.vstack(group_masks)
                normalized = self.detector._normalize_with_feature_mask(feature_matrix, feature_mask, 'global_window')
                n_rows = normalized.shape[0]
                if n_rows - seq_len + 1 > 0:
                    for i in range(0, n_rows - seq_len + 1):
                        sequences.append(normalized[i:i + seq_len])
                        masks.append(feature_mask[i:i + seq_len])
                        window_to_ring.append(window_ring_list[i + seq_len - 1])

            if not sequences:
                return {'status': 'error', 'message': '无法构建有效序列'}

            X = np.array(sequences, dtype=np.float32)
            M = np.array(masks, dtype=np.float32)
            reconstructed = self.detector.autoencoder(X, training=False).numpy()

            # 加权掩码误差
            fw_cfg = self.detector.config.get('feature_weights', {})
            fw_map = {f: float(fw_cfg.get(f, 1.0)) for f in all_feats}
            flat_w = np.concatenate([np.full(window_points, fw_map[f], dtype=np.float32) for f in all_feats])
            err = np.mean(((X - reconstructed) ** 2) * (M * flat_w.reshape(1, 1, -1)), axis=(1, 2))

            # 趋势与交互特征
            num_features_local = len(all_feats)
            feat_weights = np.array([float(fw_cfg.get(f, 1.0)) for f in all_feats], dtype=np.float32)
            use_inter = bool(self.detector.config.get('isolation_forest_params', {}).get('use_interactions', True))

            def _trend_for_row(row: np.ndarray, mask_row: np.ndarray) -> np.ndarray:
                feats = []
                slopes = {}
                deltas = {}
                for j in range(num_features_local):
                    base = j * window_points
                    seg = row[base:base + window_points]
                    seg_m = mask_row[base:base + window_points]
                    idx = np.where(seg_m)[0]
                    w = float(feat_weights[j])
                    if idx.size >= 3:
                        x = idx.astype(float)
                        y = seg[idx].astype(float)
                        p1 = np.polyfit(x, y, deg=1)
                        p2 = np.polyfit(x, y, deg=2)
                        slope = float(p1[0]) * w
                        delta = float(y[-1] - y[0]) * w
                        curvature = float(p2[0]) * w
                        std = float(np.std(y)) * w
                    else:
                        slope = 0.0; delta = 0.0; curvature = 0.0; std = 0.0
                    slopes[all_feats[j]] = slope
                    deltas[all_feats[j]] = delta
                    feats.extend([slope, delta, curvature, std])
                if use_inter:
                    for a in range(num_features_local):
                        wa = float(feat_weights[a])
                        fa = all_feats[a]
                        for b in range(a + 1, num_features_local):
                            wb = float(feat_weights[b])
                            fb = all_feats[b]
                            w_pair = wa * wb
                            sa = float(slopes.get(fa, 0.0))
                            sb = float(slopes.get(fb, 0.0))
                            da = float(deltas.get(fa, 0.0))
                            db = float(deltas.get(fb, 0.0))
                            feats.extend([
                                w_pair * (sa * sb),
                                w_pair * (da * db),
                            ])
                return np.array(feats, dtype=float)

            trend_vectors = []
            for s, m in zip(sequences, masks):
                seq_trends = [_trend_for_row(s[i], m[i]) for i in range(s.shape[0])]
                trend_vectors.append(np.vstack(seq_trends).flatten())
            scores = self.detector.isolation_forest.decision_function(trend_vectors)

            # 融合
            def _minmax(a: np.ndarray) -> np.ndarray:
                mn, mx = float(np.min(a)), float(np.max(a))
                if mx - mn < 1e-8:
                    return np.zeros_like(a)
                return (a - mn) / (mx - mn)
            norm_err = _minmax(err)
            norm_if = _minmax(-scores)
            rf = self.detector.config.get('risk_fusion', {})
            ae_w = float(rf.get('ae_weight', 0.5))
            if_w = float(rf.get('if_weight', 0.5))
            ssum = ae_w + if_w
            if ssum <= 0:
                ae_w, if_w = 0.5, 0.5
                ssum = 1.0
            ae_w /= ssum
            if_w /= ssum
            combined = ae_w * norm_err + if_w * norm_if
            # 趋势门控逻辑已删除：直接使用融合分作为风险值

            # 段落（高风险≥0.9）与最早触发（中风险≥0.65）
            segments = []
            in_seg = False
            seg_start = 0
            earliest_index = None
            for i, v in enumerate(combined):
                if earliest_index is None and v >= 0.65:
                    earliest_index = i
                if v >= 0.9 and not in_seg:
                    in_seg = True
                    seg_start = i
                elif v < 0.9 and in_seg:
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
                if ring_risk >= 0.9:
                    level = 'high'
                elif ring_risk >= 0.65:
                    level = 'medium'
                elif ring_risk >= 0.3:
                    level = 'low'
                else:
                    level = 'no_risk'
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
                if final_risk >= 0.9:
                    final_level = 'high'
                elif final_risk >= 0.65:
                    final_level = 'medium'
                elif final_risk >= 0.3:
                    final_level = 'low'
                else:
                    final_level = 'no_risk'
            earliest_time = '-'
            earliest_ring = (int(last_ring) if last_ring is not None else None)
            if last_ring is not None:
                # 仅考虑最后一环的窗口；若转换异常则抛错，由上层捕获为评估失败
                for i, (ring_i, val_i) in enumerate(zip(window_to_ring, combined)):
                    if (int(ring_i) if isinstance(ring_i, (int, float, str)) else ring_i) == int(last_ring) and val_i >= 0.65:
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


def calculate_mud_cake_risk(data: Dict[str, Any], calculator: MudCakeRiskCalculator = None) -> Dict[str, Any]:
    """
    单点评估（兼容旧接口）。
    仅使用训练好的模型进行评估；模型不可用或评估失败时直接报错，不启用备用模式。
    """
    try:
        # 严格依赖模型：模型不可用则直接报错
        if not (calculator and hasattr(calculator, 'detector') and calculator.detector is not None and 
                calculator.detector.autoencoder is not None and calculator.detector.isolation_forest is not None):
            raise RuntimeError("结泥饼模型未加载或不可用，不启用备用模式")

        # 使用“环对序列评估”接口：将单点映射为一环的聚合特征，按窗口长度重复，保持与训练一致
        all_feats = list(getattr(calculator.detector, 'all_features', []))
        if not all_feats:
            raise RuntimeError("模型特征集合不可用，无法执行评估")
        ring_id = data.get('RING', data.get('ring', 0))
        item = {'ring': ring_id}
        for f in all_feats:
            v = data.get(f, 0.0)
            try:
                item[f] = float(v) if v is not None and not pd.isna(v) else 0.0
            except Exception:
                item[f] = 0.0
        pair_res = calculator.calculate_ring_pair_risk_series([item])
        if not (isinstance(pair_res, dict) and pair_res.get('status') == 'success'):
            msg = pair_res.get('message', '模型评估失败') if isinstance(pair_res, dict) else '模型评估失败'
            raise RuntimeError(f"结泥饼模型评估失败: {msg}")
        per_ring = pair_res.get('per_ring', []) or []
        if not per_ring:
            raise RuntimeError("模型评估未返回有效环级结果")
        ring_risk = float(per_ring[0].get('combined_risk', 0.0))
        level = per_ring[0].get('risk_level', 'no_risk')
        return {'status': 'success', 'probability': ring_risk, 'risk_level': level}
    except Exception as e:
        raise

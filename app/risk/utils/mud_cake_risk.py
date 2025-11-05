import numpy as np
import pandas as pd
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, List, Set
import json
import joblib
import logging

# 配置logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class AutoEncoder(tf.keras.Model):
    """动态特征适配的LSTM自编码器（Keras）"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, original_input_size=None):
        super().__init__()
        # 与训练脚本保持一致的属性
        self.model_input_size = int(input_size)
        self.original_input_size = int(original_input_size or input_size)

        # 与训练脚本一致：编码器返回序列和状态，解码器返回序列
        self.encoder = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            return_state=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 保持输出层命名为 output_dense，便于权重映射
        self.output_dense = tf.keras.layers.Dense(self.model_input_size)

    def call(self, x, training=False):
        # x shape: (batch, time, features)
        input_dim = tf.shape(x)[-1]

        # 与训练脚本一致的维度适配：按静态特征维度进行截断/填充
        if x.shape[-1] != self.model_input_size:
            if x.shape[-1] > self.model_input_size:
                x = x[:, :, :self.model_input_size]
            else:
                pad_dim = self.model_input_size - x.shape[-1]
                padding = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], pad_dim), dtype=x.dtype)
                x = tf.concat([x, padding], axis=-1)

        # 编码：获取序列和最终隐状态
        encoded_seq, h, c = self.encoder(x, training=training)
        # 解码：使用编码器隐状态作为初始状态
        decoded_seq = self.decoder(encoded_seq, initial_state=[h, c], training=training)
        # 输出重构
        reconstructed = self.output_dense(decoded_seq)

        # 输出维度适配到原始输入维度
        if self.original_input_size != self.model_input_size:
            if self.original_input_size > self.model_input_size:
                pad_dim = self.original_input_size - self.model_input_size
                padding = tf.zeros((tf.shape(reconstructed)[0], tf.shape(reconstructed)[1], pad_dim), dtype=reconstructed.dtype)
                reconstructed = tf.concat([reconstructed, padding], axis=-1)
            else:
                reconstructed = reconstructed[:, :, :self.original_input_size]

        return reconstructed


class MudCakeRiskCalculator:

    # 风险阈值定义为类变量
    RISK_THRESHOLDS = {
        'no_risk': 0.0,  # 0-0.3: 无风险
        'low': 0.3,  # 0.3-0.524: 低风险
        'medium': 0.524,  # 0.524-0.9: 中风险
        'high': 0.9  # 0.9-1.0: 高风险
    }

    def __init__(self, model_type='unified', model_dir=None):
        self.model_type = model_type
        # TensorFlow 自动选择设备；记录设备信息供调试
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        self.models_loaded = False

        # 模型组件初始化为None
        self.scaler = None
        self.autoencoder = None
        self.isolation_forest = None
        self.reconstruction_threshold = float(os.environ.get("MUD_RECON_THRESHOLD", "0.18"))
        self.isolation_threshold = float(os.environ.get("MUD_ISO_THRESHOLD", "-0.25"))
        self.rec_weight = float(os.environ.get("MUD_REC_WEIGHT", "0.5"))
        self.iso_weight = float(os.environ.get("MUD_ISO_WEIGHT", "0.5"))
        # 环境可控的风险分档阈值（与训练分布对齐）
        self.risk_thresholds = {
            'no_risk': 0.0,
            'low': float(os.environ.get("MUD_LOW_BOUND", str(self.RISK_THRESHOLDS['low']))),
            'medium': float(os.environ.get("MUD_MEDIUM_BOUND", str(self.RISK_THRESHOLDS['medium']))),
            'high': float(os.environ.get("MUD_HIGH_BOUND", str(self.RISK_THRESHOLDS['high'])))
        }
        # 是否忽略 RING/state 两列（默认不忽略，按训练使用）
        self.ignore_ring_state = os.environ.get("MUD_IGNORE_RING_STATE", "false").lower() in ("1", "true", "yes")
        self.available_features = []
        # 允许特征（仅作为兜底；实际以train.py保存的all_features为准）
        self.ALLOWED_FEATURES = ['TJSD', 'TJL', 'DP_SD', 'DP_ZJ', 'DP_SS_ZTL']
        self.feature_dim = 0
        self.model_info = {}

        # 设置模型目录
        if model_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(os.path.dirname(current_dir), 'models')
        else:
            self.model_dir = model_dir

    def _safe_load_pickle(self, file_path):
        """安全加载pickle文件，处理损坏的文件"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            try:
                return joblib.load(file_path)
            except Exception as e2:
                raise

    def _safe_tf_load(self, file_path, custom_objects=None):
        """安全加载TensorFlow/Keras模型文件（SavedModel 或 HDF5）"""
        try:
            # 优先按 Keras SavedModel/H5 加载
            model = tf.keras.models.load_model(file_path, compile=False, custom_objects=custom_objects or {})
            return model
        except Exception:
            # 尝试 SavedModel 目录加载为 Keras 模型
            try:
                model = tf.keras.models.load_model(file_path, compile=False, custom_objects=custom_objects or {})
                return model
            except Exception:
                # 回退到原始SavedModel（非Keras）
                try:
                    model = tf.saved_model.load(file_path)
                    return model
                except Exception:
                    raise

    def _safe_tf_load_weights(self, model_instance, file_path):
        """安全加载Keras权重（H5），用于子类模型"""
        try:
            model_instance.load_weights(file_path)
            return model_instance
        except Exception:
            raise
    def _load_models(self):
        try:
            model_info_path = os.path.join(self.model_dir, 'mud_cake_model_info.json')
            if not os.path.exists(model_info_path):
                raise FileNotFoundError(f"Mud cake model info file not found: {model_info_path}")
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                self.model_info = model_info
                training_features = model_info.get('training_features', {})
                # 以训练保存的特征列表为标准，不再限制为5个允许特征
                available = training_features.get('available_features', model_info.get('all_features', []))
                self.available_features = available if available else model_info.get('all_features', [])
                if not self.available_features:
                    # 兜底使用允许特征
                    self.available_features = list(self.ALLOWED_FEATURES)
                self.feature_dim = len(self.available_features)
                scaler_path = os.path.join(self.model_dir, 'mud_cake_scalers.pkl')
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Mud cake scaler file not found: {scaler_path}")
                scalers_dict = self._safe_load_pickle(scaler_path)
                if isinstance(scalers_dict, dict):
                    if 'Other_Projects' in scalers_dict:
                        self.scaler = scalers_dict['Other_Projects']
                    else:
                        self.scaler = list(scalers_dict.values())[0]
                else:
                    self.scaler = scalers_dict
                # 3. 加载结泥饼自编码器模型（TensorFlow/Keras）
                autoencoder_path_h5 = os.path.join(self.model_dir, 'mud_cake_autoencoder.h5')
                autoencoder_path_dir = os.path.join(self.model_dir, 'mud_cake_autoencoder')

                # 从配置获取参数
                autoencoder_params = model_info['config']['autoencoder_params']
                original_input_size = autoencoder_params.get('input_size', len(self.ALLOWED_FEATURES))

                # 优先加载权重（H5）到子类模型
                if os.path.exists(autoencoder_path_h5):
                    try:
                        # 优先尝试按“完整Keras模型”加载（H5保存的全模型）
                        loaded = self._safe_tf_load(autoencoder_path_h5, custom_objects={'AutoEncoder': AutoEncoder})
                        setattr(loaded, 'model_input_size', int(original_input_size))
                        setattr(loaded, 'original_input_size', int(self.feature_dim))
                        self.autoencoder = loaded
                        logger.info("Loaded mud cake autoencoder (H5 full model) successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load H5 full model; trying weights: {e}")
                        try:
                            ae = AutoEncoder(
                                input_size=original_input_size,
                                hidden_size=autoencoder_params['hidden_size'],
                                num_layers=autoencoder_params['num_layers'],
                                dropout=autoencoder_params['dropout'],
                                original_input_size=self.feature_dim
                            )
                            # 显式构建模型图以初始化变量（使用已保存的序列长度）
                            try:
                                seq_len = int(self.model_info.get('config', {}).get('sequence_length', 25))
                                dummy = tf.zeros((1, seq_len, int(self.feature_dim)), dtype=tf.float32)
                                ae(dummy, training=False)
                            except Exception as build_e:
                                logger.warning(f"AutoEncoder build before weight load failed: {build_e}")
                            self.autoencoder = self._safe_tf_load_weights(ae, autoencoder_path_h5)
                            logger.info("Loaded mud cake autoencoder weights (H5) successfully")
                        except Exception as e2:
                            logger.warning(f"Failed to load H5 weights for autoencoder: {e2}")
                            self.autoencoder = None
                elif os.path.exists(autoencoder_path_dir):
                    try:
                        loaded = self._safe_tf_load(autoencoder_path_dir, custom_objects={'AutoEncoder': AutoEncoder})
                        setattr(loaded, 'model_input_size', int(original_input_size))
                        setattr(loaded, 'original_input_size', int(self.feature_dim))
                        self.autoencoder = loaded
                        logger.info("Loaded mud cake autoencoder (SavedModel) successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load SavedModel autoencoder: {e}")
                        self.autoencoder = None
                else:
                    # 构建一个形状对齐的占位模型（未加载权重时不用于计算，仍走风险兜底）
                    try:
                        self.autoencoder = AutoEncoder(
                            input_size=original_input_size,
                            hidden_size=autoencoder_params['hidden_size'],
                            num_layers=autoencoder_params['num_layers'],
                            dropout=autoencoder_params['dropout'],
                            original_input_size=self.feature_dim
                        )
                        logger.info("Built fallback Keras autoencoder instance (no weights)")
                    except Exception as e:
                        logger.warning(f"Failed to build fallback AutoEncoder: {e}")
                        self.autoencoder = None

                # 4. 加载结泥饼孤立森林模型
                isolation_path = os.path.join(self.model_dir, 'mud_cake_isolation_forest.pkl')
                if not os.path.exists(isolation_path):
                    raise FileNotFoundError(f"Mud cake isolation forest file not found: {isolation_path}")

                self.isolation_forest = self._safe_load_pickle(isolation_path)
                logger.info("Loaded mud cake isolation forest successfully")

                # 5. 设置结泥饼风险阈值（使用 __init__ 中配置）
                # 保持 __init__ 中通过环境变量设定的阈值，不在此处覆盖
                # 严格校验：所有核心组件必须可用
                if self.scaler is None or self.autoencoder is None or self.isolation_forest is None:
                    raise RuntimeError("Mud cake models incomplete: autoencoder/scaler/isolation not available")

                self.models_loaded = True
                logger.info("All mud cake trained models loaded successfully")

            except Exception as e:
                logger.error(f"Error loading mud cake trained models: {str(e)}")
                self.models_loaded = False
                raise RuntimeError(f"Failed to load mud cake trained models: {str(e)}")

        except Exception as e:
            logger.error(f"Error loading mud cake trained models: {str(e)}")
            self.models_loaded = False
            raise RuntimeError(f"Failed to load mud cake trained models: {str(e)}")

    def create_standard_features(self, data_point: Dict[str, Any], available_columns: Set[str] = None) -> Dict[
        str, float]:
        """按训练特征列表严格取值；缺失或无效置0"""
        features = {}

        # 训练特征顺序
        feature_list = self.available_features if self.available_features else self.model_info.get('all_features', [])
        prop = data_point.get('property', {}) if isinstance(data_point, dict) else {}

        for feature in feature_list:
            if feature == 'RING':
                val = data_point.get('ring', 0)
            elif feature == 'state':
                val = data_point.get('state', 0)
            else:
                val = prop.get(feature, 0)
            try:
                features[feature] = float(val) if not pd.isna(val) else 0.0
            except Exception:
                features[feature] = 0.0

        return features

    def _validate_input(self, data_sequence):
        """验证输入数据的有效性"""
        if not data_sequence:
            raise ValueError("数据序列不能为空")

        if not isinstance(data_sequence, list):
            raise ValueError("数据序列必须是列表格式")

        for i, data in enumerate(data_sequence):
            if not isinstance(data, dict):
                raise ValueError(f"数据点 {i} 必须是字典格式")

            if 'property' not in data:
                raise ValueError(f"数据点 {i} 缺少 'property' 字段")

    def _extract_features(self, data_sequence) -> np.ndarray:
        """按训练特征列表构造序列特征矩阵"""
        features = []
        feature_list = self.available_features if self.available_features else self.model_info.get('all_features', [])
        for data in data_sequence:
            feature_dict = self.create_standard_features(data)
            feature_row = []
            for f in feature_list:
                # 支持开关忽略 RING/state，否则按训练使用真实值
                if f in ('RING', 'state') and getattr(self, 'ignore_ring_state', False):
                    feature_row.append(0.0)
                else:
                    feature_row.append(feature_dict.get(f, 0.0))
            features.append(feature_row)
        return np.array(features)

    def _calculate_reconstruction_risk(self, features):
        """计算自编码器重构风险，支持任意特征数量的动态适配"""
        if not self.models_loaded or self.autoencoder is None:
            raise RuntimeError("模型未加载或自编码器不可用")

        try:
            # [batch, time, dim]
            features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
            features_tensor = tf.expand_dims(features_tensor, axis=0)

            # 获取当前特征维度和模型期望维度
            current_dim = int(features_tensor.shape[-1])
            model_input_size = int(getattr(self.autoencoder, 'model_input_size', current_dim))
            original_input_size = int(getattr(self.autoencoder, 'original_input_size', current_dim))

            logger.info(f"重构风险计算 - 当前特征维度: {current_dim}, 模型输入维度: {model_input_size}, 原始输入维度: {original_input_size}")

            # 动态调整输入特征维度
            if current_dim != model_input_size:
                if current_dim < model_input_size:
                    pad_dim = model_input_size - current_dim
                    padding = tf.zeros([1, tf.shape(features_tensor)[1], pad_dim], dtype=features_tensor.dtype)
                    features_tensor_adj = tf.concat([features_tensor, padding], axis=2)
                    reconstructed = self.autoencoder(features_tensor_adj, training=False)
                    reconstructed = reconstructed[:, :, :current_dim]
                else:
                    features_tensor_adj = features_tensor[:, :, :model_input_size]
                    reconstructed = self.autoencoder(features_tensor_adj, training=False)
                    if int(reconstructed.shape[-1]) > current_dim:
                        reconstructed = reconstructed[:, :, :current_dim]
            else:
                reconstructed = self.autoencoder(features_tensor, training=False)

            # 比较相同维度
            min_dim = min(int(features_tensor.shape[-1]), int(reconstructed.shape[-1]))
            features_tensor_compare = features_tensor[:, :, :min_dim]
            reconstructed_compare = reconstructed[:, :, :min_dim]

            # 计算重构误差（MSE）
            errors = tf.reduce_mean(tf.square(reconstructed_compare - features_tensor_compare), axis=[1, 2])
            avg_error = float(tf.reduce_mean(errors).numpy())
            # 使用序列内的鲁棒阈值（median + 3*MAD）进行数据驱动概率映射
            per_step_errors = tf.reduce_mean(tf.square(reconstructed_compare - features_tensor_compare), axis=2)
            errors_np = per_step_errors.numpy().reshape(-1)
            baseline = float(np.median(errors_np)) if errors_np.size > 0 else avg_error
            mad = float(np.median(np.abs(errors_np - baseline))) if errors_np.size > 0 else 0.0
            dynamic_threshold = baseline + 3.0 * mad
            if dynamic_threshold <= 1e-8:
                dynamic_threshold = max(self.reconstruction_threshold, 1e-3)
            risk_prob = float(np.mean(errors_np > dynamic_threshold)) if errors_np.size > 0 else min(1.0, avg_error / max(self.reconstruction_threshold, 1e-3))
            return risk_prob, avg_error
        except Exception as e:
            logger.error(f"重构风险计算错误: {str(e)}")
            raise

    def _calculate_isolation_risk(self, features):
        """计算孤立森林异常风险（序列展平 T*D）"""
        if not self.models_loaded or self.isolation_forest is None:
            raise RuntimeError("模型未加载或孤立森林不可用")

        try:
            flat = features.reshape(1, -1)
            if hasattr(self.isolation_forest, 'n_features_in_'):
                expected = self.isolation_forest.n_features_in_
            elif hasattr(self.isolation_forest, 'n_features_'):
                expected = self.isolation_forest.n_features_
            else:
                seq_len = int(self.model_info.get('config', {}).get('sequence_length', features.shape[0]))
                expected = seq_len * features.shape[1]

            actual = flat.shape[1]
            if actual != expected:
                if actual < expected:
                    padding = np.zeros((1, expected - actual))
                    flat = np.hstack([flat, padding])
                else:
                    flat = flat[:, :expected]

            try:
                score = self.isolation_forest.decision_function(flat)
                avg = float(np.mean(score))
            except Exception:
                score = self.isolation_forest.score_samples(flat)
                avg = float(np.mean(score))

            # 使用Logistic映射将孤立森林分数转换为概率（正值更正常，负值更异常）
            iso_scale = float(os.environ.get("MUD_ISO_SCALE", "5.0"))
            risk_prob = float(1.0 / (1.0 + np.exp(iso_scale * avg)))
            risk_prob = max(0.0, min(1.0, risk_prob))
            return risk_prob, avg
        except Exception as e:
            logger.error(f"孤立森林风险计算错误: {str(e)}")
            raise

    def calculate_risk(self, data_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算结泥饼风险，支持动态特征适配和缺失字段处理"""
        try:
            self._validate_input(data_sequence)

            if not self.models_loaded:
                logger.info("Loading unified model with missing flag method")
                self._load_models()

            # 提取特征并记录可用字段
            features = self._extract_features(data_sequence)
            logger.debug(f"提取到特征矩阵: 形状={features.shape}")
            
            # 对齐训练序列长度
            seq_len = int(self.model_info.get('config', {}).get('sequence_length', features.shape[0]))
            if features.shape[0] != seq_len:
                cur = features.shape[0]
                if cur > seq_len:
                    features = features[-seq_len:]
                else:
                    missing = seq_len - cur
                    if cur == 0:
                        pad = np.zeros((missing, features.shape[1]))
                        features = np.vstack([features, pad]) if features.size else pad
                    else:
                        fill_method = getattr(self, 'seq_fill_method', 'repeat_window')
                        if fill_method == 'repeat_last':
                            pad = np.tile(features[-1], (missing, 1))
                            features = np.vstack([features, pad])
                        elif fill_method == 'zero_pad':
                            pad = np.zeros((missing, features.shape[1]))
                            features = np.vstack([features, pad])
                        else:  # repeat_window
                            reps = int(np.ceil(missing / cur))
                            tiled = np.tile(features, (reps, 1))[:missing]
                            features = np.vstack([features, tiled])
                    logger.info(f"对齐序列长度: {cur} -> {seq_len}, 填充方式: {getattr(self, 'seq_fill_method', 'repeat_window')}")
            
            # 标准化：优先整矩阵变换，确保特征顺序与训练一致
            if self.scaler is not None:
                try:
                    original_features = features.copy()
                    # 如果scaler是针对多特征训练的（常见情形），直接对整矩阵进行变换
                    if hasattr(self.scaler, 'transform'):
                        # 判断维度是否匹配（n_features 或 mean_ 长度）
                        n_in = getattr(self.scaler, 'n_features_in_', None)
                        mean_len = len(getattr(self.scaler, 'mean_', [])) if hasattr(self.scaler, 'mean_') else None
                        if n_in == features.shape[1] or mean_len == features.shape[1] or (n_in is None and mean_len is None):
                            features = self.scaler.transform(features)
                        else:
                            # 尝试列级标注的scaler映射（dict[str->StandardScaler]）
                            normalized = np.zeros_like(features)
                            feature_list = self.available_features if self.available_features else self.model_info.get('all_features', [])
                            for j, fname in enumerate(feature_list):
                                col = features[:, j:j+1]
                                scaler_j = None
                                if isinstance(self.scaler, dict):
                                    scaler_j = self.scaler.get(fname) or self.scaler.get(str(j))
                                if scaler_j is None:
                                    normalized[:, j] = col[:, 0]
                                    continue
                                try:
                                    normalized[:, j:j+1] = scaler_j.transform(col)
                                except Exception:
                                    normalized[:, j] = col[:, 0]
                            features = normalized
                    else:
                        # 无 transform 能力，保留原始数值
                        original_features = features.copy()
                except Exception as e:
                    logger.error(f"标准化特征时出错: {e}")
                    logger.warning("标准化失败，使用原始特征继续计算")
                    original_features = features.copy()
            else:
                # 没有标准化器，保存原始特征
                original_features = features.copy()

            # 计算重构风险和孤立森林风险
            rec_risk, rec_error = self._calculate_reconstruction_risk(features)
            iso_risk, iso_score = self._calculate_isolation_risk(features)
            # 使用类属性权重（支持环境变量覆盖）
            rec_weight = self.rec_weight
            iso_weight = self.iso_weight
            
            combined_risk = rec_weight * rec_risk + iso_weight * iso_risk
            combined_risk = min(combined_risk, 1.0)

            # 使用实例级阈值（支持环境变量）
            if combined_risk >= self.risk_thresholds['high']:
                risk_level = '高风险'
            elif combined_risk >= self.risk_thresholds['medium']:
                risk_level = '中风险'
            elif combined_risk >= self.risk_thresholds['low']:
                risk_level = '低风险'
            else:
                risk_level = '无风险'

            # 统计特征列有效性（避免未定义变量导致异常）
            try:
                total_cols = int(features.shape[1])
                non_zero_cols = int(np.sum(np.any(features != 0, axis=0)))
            except Exception:
                total_cols = int(self.feature_dim) if hasattr(self, 'feature_dim') else features.shape[1]
                non_zero_cols = total_cols
            return {
                'status': 'success',
                'combined_risk': combined_risk,
                'risk_level': risk_level,
                'components': {
                    'reconstruction': {
                        'risk': round(rec_risk, 4),
                        'error': round(rec_error, 6),
                        'threshold': self.reconstruction_threshold
                    },
                    'isolation': {
                        'risk': round(iso_risk, 4),
                        'score': round(iso_score, 6),
                        'threshold': self.isolation_threshold
                    }
                },
                'feature_info': {
                    'total_features': total_cols,
                    'effective_features': non_zero_cols,
                    'effective_ratio': round(non_zero_cols/total_cols, 2)
                },
                'data_points': len(data_sequence),
                'model_type': f"{self.model_type}_unified_missing_flag"
            }

        except ValueError as ve:
            logger.error(f"Input validation error: {str(ve)}")
            raise
        except Exception as e:
            # 降低该异常的日志等级，避免控制台噪声输出
            logger.debug(f"Calculation error: {str(e)}")
            raise
def calculate_mud_cake_risk(data_point, shield_id='unified', calculator=None):
    """计算结泥饼风险的接口函数 - 支持使用预加载的计算器"""
    try:
        if calculator is None:
            calculator = MudCakeRiskCalculator(model_type=shield_id)
        # 按训练特征列表构造属性字典（缺失置0）
        feature_list = calculator.available_features if calculator.available_features else calculator.model_info.get('all_features', [])
        property_dict = {}
        for field in feature_list:
            if field in ('RING', 'state'):
                continue
            try:
                value = data_point.get(field, 0)
                property_dict[field] = float(value) if value is not None and value != '' and not pd.isna(value) else 0.0
            except Exception:
                property_dict[field] = 0.0

        formatted_data_point = {
            'property': property_dict,
            'state': data_point.get('state', 'excavating'),
            'timestamp': data_point.get('ts(Asia/Shanghai)', ''),
            'ring': data_point.get('RING', 0)
        }

        # 使用训练配置的序列长度
        seq_len = int(calculator.model_info.get('config', {}).get('sequence_length', 25))
        data_sequence = [formatted_data_point] * seq_len
        result = calculator.calculate_risk(data_sequence)

        if result['status'] == 'success':
            return {
                'probability': round(result['combined_risk'], 2),
                'risk_level': result['risk_level'],
                'details': f"综合风险: {result['combined_risk']}"
            }
        else:
            raise RuntimeError(result.get('message', '模型评估失败'))

    except Exception as e:
        raise


def get_risk_level_by_probability(probability):
    """根据概率值确定风险级别（支持环境阈值）"""
    low = float(os.environ.get("MUD_LOW_BOUND", str(MudCakeRiskCalculator.RISK_THRESHOLDS['low'])))
    medium = float(os.environ.get("MUD_MEDIUM_BOUND", str(MudCakeRiskCalculator.RISK_THRESHOLDS['medium'])))
    high = float(os.environ.get("MUD_HIGH_BOUND", str(MudCakeRiskCalculator.RISK_THRESHOLDS['high'])))
    if probability >= high:
        return '高风险'
    elif probability >= medium:
        return '中风险'
    elif probability >= low:
        return '低风险'
    else:
        return '无风险'
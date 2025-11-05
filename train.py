import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import tensorflow as tf
import random


class AutoEncoder(tf.keras.Model):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.3,
                 original_input_size: int = None):
        super().__init__()
        # 设置模型期望的输入维度
        self.model_input_size = input_size
        # 设置实际数据的输入维度（如果提供）
        self.original_input_size = original_input_size or input_size

        # 编码器：返回序列和状态
        self.encoder = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            return_state=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 解码器：返回序列
        self.decoder = tf.keras.layers.LSTM(
            units=hidden_size,
            return_sequences=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 输出层：逐时序回归到 input_size
        self.output_dense = tf.keras.layers.Dense(self.model_input_size)

    def call(self, x, training=False):
        # x shape: (batch, time, features)

        # 如果输入维度不匹配，直接调整
        if x.shape[-1] != self.model_input_size:
            if x.shape[-1] > self.model_input_size:
                x = x[:, :, :self.model_input_size]
                tf.print("输入特征维度(", tf.shape(x)[-1], ")大于模型期望维度(", self.model_input_size, ")，已自动截断")
            else:
                pad_dim = self.model_input_size - x.shape[-1]
                padding = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], pad_dim), dtype=x.dtype)
                x = tf.concat([x, padding], axis=-1)
                tf.print("输入特征维度(", tf.shape(x)[-1], ")小于模型期望维度(", self.model_input_size, ")，已自动填充")

        # 编码
        encoded_seq, h, c = self.encoder(x, training=training)
        # 解码（使用编码器的最终隐状态作为初始状态）
        decoded_seq = self.decoder(encoded_seq, initial_state=[h, c], training=training)
        # 输出重构
        reconstructed = self.output_dense(decoded_seq)

        # 如果需要调整输出维度以匹配原始输入维度
        if self.original_input_size != self.model_input_size:
            if self.original_input_size > self.model_input_size:
                pad_dim = self.original_input_size - self.model_input_size
                padding = tf.zeros((tf.shape(reconstructed)[0], tf.shape(reconstructed)[1], pad_dim), dtype=reconstructed.dtype)
                reconstructed = tf.concat([reconstructed, padding], axis=-1)
            else:
                reconstructed = reconstructed[:, :, :self.original_input_size]

        return reconstructed


class DeepMudCakeDetector:
    def __init__(self):
        self.device = self._select_device()
        self.autoencoder = None
        self.isolation_forest = None
        self.project_scalers = {}  # 按数据源的标准化器
        self.model_dir = Path('app/risk/models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config = {
            'sequence_length': 75,
            'min_sequence_length': 10,
            'max_sequence_length': 80,
            'validation_split': 0.2,
            'sampling': {
                'enable_sampling': True,
                'sampling_ratio': 0.6,
                'sampling_method': 'random',
                'random_seed': 42,
                'preserve_time_order': False,
                'only_state_1': True
            },
            'autoencoder_params': {
                'hidden_size': 24,
                'num_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.0005,
                'learning_rate_901_1000': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'epochs_901_1000': 50,
                'loss_weight_901_1000': 4.0,
                'loss_weight_other': 3.0
            },
            'isolation_forest_params': {
                'contamination': 0.1,
                'random_state': 42,
                'n_estimators': 50
            }
        }

    def _select_device(self):
        """设备选择 - 检测TensorFlow GPU并报告"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # 启用按需显存增长，避免一次性占满显存
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
                details = tf.config.experimental.get_device_details(gpus[0])
                mem = details.get('device_memory', None)
                if mem is not None:
                    print(f"检测到GPU显存: {mem / (1024 ** 3):.1f} GB")
                print("使用GPU进行训练")
                return 'GPU'
            except Exception as e:
                print(f"GPU检测失败，使用CPU: {e}")
                return 'CPU'
        else:
            print("未检测到GPU，使用CPU进行训练")
            return 'CPU'

    def load_training_data(self, data_paths: List[str]) -> pd.DataFrame:
        """加载训练数据 - 只加载state=1的数据，动态忽略缺失字段，不需要填充数据"""
        print("开始加载训练数据...")
        combined_data = []
        all_potential_features = set()

        # 设置随机种子
        if self.config['sampling']['enable_sampling']:
            random.seed(self.config['sampling']['random_seed'])
            np.random.seed(self.config['sampling']['random_seed'])
            print(f"启用数据采样，采样比例: {self.config['sampling']['sampling_ratio']}")

        # 定义候选特征字段（不使用GRD和QDV温度，改用DP_SS_ZTL）
        all_possible_features = ['TJSD', 'TJL', 'DP_SD', 'DP_ZJ', 'DP_SS_ZTL']

        for data_path in data_paths:
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
                if not df.empty:
                    file_name = Path(data_path).stem
                    print(f"\n处理文件: {file_name}")

                    # 只保留state=1的数据（如果存在该列且配置启用）
                    if 'state' in df.columns and self.config['sampling']['only_state_1']:
                        original_len = len(df)
                        df = df[df['state'] == 1]
                        filtered_len = len(df)
                        print(f"文件 {file_name}: 过滤state=1数据，从{original_len}行减少到{filtered_len}行")
                        if df.empty:
                            print(f"警告: 文件 {file_name} 过滤state=1后没有数据，跳过此文件")
                            continue
                    df['data_source'] = file_name
                    if self.config['sampling']['enable_sampling']:
                        if '901-1000' in file_name:
                            print(f"文件 {file_name}: 使用全部数据进行学习（结泥饼发生概率高）")
                        else:
                            df = self._apply_sampling(df)

                    df['project_type'] = "TBM_Project"
                    df['file_size'] = len(df)

                    # 检查并记录该文件中存在的有效特征字段（有效率与方差过滤）
                    project_features = []
                    for feature in all_possible_features:
                        if feature in df.columns:
                            valid_ratio = df[feature].notna().sum() / len(df)
                            variance = df[feature].var() if valid_ratio > 0 else 0
                            if valid_ratio > 0.05 and variance > 1e-6:
                                project_features.append(feature)
                                all_potential_features.add(feature)

                    # 辅助特征（如存在则加入可用特征列表）
                    if 'RING' in df.columns:
                        project_features.append('RING')
                        all_potential_features.add('RING')
                    if 'state' in df.columns:
                        project_features.append('state')
                        all_potential_features.add('state')

                    df['available_features'] = [project_features] * len(df)
                    combined_data.append(df)
            except Exception as e:
                print(f"加载数据文件{data_path}时出错: {str(e)}")
                continue

        if not combined_data:
            raise ValueError("没有找到有效的训练数据文件")

        result = pd.concat(combined_data, ignore_index=True)
        self.all_features = sorted(list(all_potential_features))
        self.feature_dim = len(self.all_features)
        self.config['autoencoder_params']['input_size'] = self.feature_dim
        print(f"数据加载完成 | 合并行数: {len(result)} | 特征维度: {self.feature_dim} | 文件数: {len(data_paths)}")
        return result

    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """采样处理 - 901-1000来源保留全部，其他按比例采样"""
        file_name = df['data_source'].iloc[0] if 'data_source' in df.columns else "unknown"
        sampling_config = self.config['sampling']

        # 高风险数据源使用全量样本
        if '901-1000' in file_name:
            print(f"文件 {file_name}: 使用全部数据进行学习（结泥饼发生概率高）")
            return df

        # 其他数据源按比例采样
        ratio = float(sampling_config.get('sampling_ratio', 1.0))
        if ratio >= 1.0:
            print(f"数据源 {file_name}: 采样比例为1.0，使用全部数据，原始数据量 {len(df)}")
            return df

        target_samples = max(1, int(len(df) * ratio))
        method = sampling_config.get('sampling_method', 'random')
        print(f"数据源 {file_name}: 按比例采样 ratio={ratio} → 样本数 {target_samples}/{len(df)}，method={method}")
        try:
            if method == 'stratified':
                return self._stratified_sampling(df, target_samples)
            elif method == 'systematic':
                return self._systematic_sampling(df, target_samples)
            else:
                return self._random_sampling(df, target_samples)
        except Exception as e:
            print(f"采样失败，回退到随机采样: {e}")
            return self._random_sampling(df, target_samples)

    def _random_sampling(self, df: pd.DataFrame, target_samples: int) -> pd.DataFrame:
        """随机采样"""
        if self.config['sampling']['preserve_time_order']:
            sampled_indices = sorted(df.sample(n=target_samples).index)
            return df.loc[sampled_indices].reset_index(drop=True)
        else:
            return df.sample(n=target_samples).reset_index(drop=True)

    def _stratified_sampling(self, df: pd.DataFrame, target_samples: int) -> pd.DataFrame:
        """分层采样 - 基于state进行分层（如果可用）"""
        try:
            if 'state' in df.columns and df['state'].nunique() > 1:
                stratified_samples = []
                state_counts = df['state'].value_counts()
                for state, count in state_counts.items():
                    state_data = df[df['state'] == state]
                    state_target = max(1, int(target_samples * count / len(df)))
                    state_target = min(state_target, len(state_data))
                    sampled_state_data = state_data.sample(n=state_target)
                    stratified_samples.append(sampled_state_data)
                result = pd.concat(stratified_samples, ignore_index=True)
                if len(result) < target_samples:
                    remaining = target_samples - len(result)
                    remaining_data = df[~df.index.isin(result.index)]
                    if len(remaining_data) > 0:
                        additional = remaining_data.sample(n=min(remaining, len(remaining_data)))
                        result = pd.concat([result, additional], ignore_index=True)
                return result[:target_samples]
            else:
                return self._random_sampling(df, target_samples)
        except Exception as e:
            print(f"分层采样失败，回退到随机采样: {e}")
            return self._random_sampling(df, target_samples)

    def _systematic_sampling(self, df: pd.DataFrame, target_samples: int) -> pd.DataFrame:
        """系统采样 - 等间隔采样"""
        if target_samples >= len(df):
            return df
        interval = len(df) / target_samples
        indices = [int(i * interval) for i in range(target_samples)]
        return df.iloc[indices].reset_index(drop=True)

    def create_feature_sequences(self, data: pd.DataFrame) -> List[tuple]:
        """创建特征序列 - 返回(序列, 掩码)，动态忽略缺失字段"""
        sequences = []
        data_source_groups = data.groupby('data_source')
        data_source_info = {}
        for data_source, source_data in data_source_groups:
            data_source_info[data_source] = {'size': len(source_data), 'data': source_data}

        for data_source, info in data_source_info.items():
            source_data = info['data']
            time_col = 'ts(Asia/Shanghai)' if 'ts(Asia/Shanghai)' in source_data.columns else 'ts'
            if time_col in source_data.columns:
                source_data = source_data.sort_values(time_col).reset_index(drop=True)

            available_features = source_data['available_features'].iloc[0]
            feature_matrix = np.zeros((len(source_data), self.feature_dim))
            feature_mask = np.zeros((len(source_data), self.feature_dim), dtype=bool)

            for i, global_feature in enumerate(self.all_features):
                if global_feature in available_features and global_feature in source_data.columns:
                    feature_values = source_data[global_feature].values.astype(float)
                    valid_mask = ~np.isnan(feature_values)
                    feature_matrix[valid_mask, i] = feature_values[valid_mask]
                    feature_mask[valid_mask, i] = True

            # 标准化处理 - 按数据源共享尺度，仅对有效值进行变换
            normalized_data = self._normalize_with_feature_mask(feature_matrix, feature_mask, str(data_source))

            available_sequences = len(normalized_data) - self.config['sequence_length'] + 1
            if available_sequences <= 0:
                continue
            step_size = 2 if '901-1000' in str(data_source) else 4

            for i in range(0, available_sequences, step_size):
                seq = normalized_data[i:i + self.config['sequence_length']]
                mask_seq = feature_mask[i:i + self.config['sequence_length']]
                sequences.append((seq, mask_seq))

        print(f"序列构建完成 | 总序列数: {len(sequences)} | 序列长度: {self.config['sequence_length']} | 特征维度: {self.feature_dim}")
        return sequences

    def _normalize_with_feature_mask(self, data: np.ndarray, feature_mask: np.ndarray, project_type: str) -> np.ndarray:
        """使用特征掩码进行标准化处理（按数据源共享尺度）"""
        scaler_key = f"{project_type}"
        if scaler_key not in self.project_scalers:
            self.project_scalers[scaler_key] = StandardScaler()
            real_values = []
            for i in range(data.shape[1]):
                if np.any(feature_mask[:, i]):
                    col_data = data[:, i]
                    col_mask = feature_mask[:, i]
                    real_col_values = col_data[col_mask]
                    real_values.extend(real_col_values)
            if real_values:
                real_array = np.array(real_values).reshape(-1, 1)
                self.project_scalers[scaler_key].fit(real_array)
            else:
                self.project_scalers[scaler_key].fit(np.zeros((10, 1)))

        normalized_data = data.copy()
        for i in range(data.shape[1]):
            if np.any(feature_mask[:, i]):
                col_data = data[:, i:i + 1]
                col_mask = feature_mask[:, i]
                normalized_col = self.project_scalers[scaler_key].transform(col_data)
                normalized_data[col_mask, i] = normalized_col[col_mask, 0]
                normalized_data[~col_mask, i] = 0.0
        return normalized_data

    def train_autoencoder(self, sequences: List[tuple], has_901_1000_data: bool = False) -> None:
        """训练自编码器 - 使用掩码，仅计算有效特征损失（TensorFlow）"""
        if not sequences:
            raise ValueError("没有有效的训练序列")
        try:
            # 初始化模型
            self.autoencoder = AutoEncoder(
                input_size=self.feature_dim,
                hidden_size=self.config['autoencoder_params']['hidden_size'],
                num_layers=self.config['autoencoder_params']['num_layers'],
                dropout=self.config['autoencoder_params']['dropout']
            )

            print(f"使用所有 {len(sequences)} 个序列进行训练")
            data_sequences = []
            mask_sequences = []
            for seq, mask in sequences:
                data_sequences.append(seq)
                mask_sequences.append(mask)

            X_np = np.array(data_sequences, dtype=np.float32)
            masks_np = np.array(mask_sequences, dtype=np.float32)
            train_size = int(len(X_np) * (1 - self.config['validation_split']))
            train_X, val_X = X_np[:train_size], X_np[train_size:]
            train_masks, val_masks = masks_np[:train_size], masks_np[train_size:]

            batch_size = self.config['autoencoder_params']['batch_size']
            buffer_size = min(100000, len(train_X))
            train_ds = (
                tf.data.Dataset.from_tensor_slices((train_X, train_masks))
                .shuffle(buffer_size)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            val_ds = (
                tf.data.Dataset.from_tensor_slices((val_X, val_masks))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

            base_lr = self.config['autoencoder_params']['learning_rate']
            special_lr = self.config['autoencoder_params']['learning_rate_901_1000']
            base_epochs = self.config['autoencoder_params']['epochs']
            special_epochs = self.config['autoencoder_params']['epochs_901_1000']
            loss_weight_901_1000 = self.config['autoencoder_params']['loss_weight_901_1000']
            loss_weight_other = self.config['autoencoder_params']['loss_weight_other']

            current_lr = special_lr if has_901_1000_data else base_lr
            current_epochs = special_epochs if has_901_1000_data else base_epochs
            current_loss_weight = loss_weight_901_1000 if has_901_1000_data else loss_weight_other

            optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)

            print(f"开始训练自编码器，设备: {self.device}")
            print(f"训练数据: {len(train_X)}, 验证数据: {len(val_X)}")
            if has_901_1000_data:
                print(f"检测到901-1000.csv数据，使用特殊训练参数: 学习率={current_lr}, 训练轮数={current_epochs}, 损失权重={current_loss_weight}")
            else:
                print(f"使用标准训练参数: 学习率={current_lr}, 训练轮数={current_epochs}, 损失权重={current_loss_weight}")

            # 添加早停机制
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # 使用Keras的fit方法简化训练过程
            self.autoencoder.compile(optimizer=optimizer,
                                   loss=self._masked_mse_loss(current_loss_weight))

            history = self.autoencoder.fit(
                train_ds,
                validation_data=val_ds,
                epochs=current_epochs,
                callbacks=[early_stopping]
            )

            print("自编码器训练完成")
        except Exception as e:
            print(f"自编码器训练失败: {e}")
            raise

    def _masked_mse_loss(self, weight: float = 1.0):
        """创建带掩码的MSE损失函数"""
        def loss(y_true, y_pred):
            # y_true包含数据和掩码 (data, mask)
            data = y_true[0]
            mask = y_true[1]
            squared_error = tf.square(y_pred - data)
            masked_error = tf.reduce_mean(squared_error * mask)
            return masked_error * weight
        return loss

    def train_isolation_forest(self, sequences: List[tuple]) -> None:
        """训练单一孤立森林 - 使用掩码后的有效数据展开向量"""
        if not sequences:
            raise ValueError("没有有效的训练序列")
        try:
            flattened_sequences = []
            for seq_data, seq_mask in sequences:
                masked_seq = seq_data * seq_mask
                flattened_sequences.append(masked_seq.flatten())
            X = np.array(flattened_sequences)
            self.isolation_forest = IsolationForest(
                contamination=self.config['isolation_forest_params']['contamination'],
                random_state=self.config['isolation_forest_params']['random_state'],
                n_estimators=self.config['isolation_forest_params']['n_estimators']
            )
            self.isolation_forest.fit(X)
        except Exception as e:
            print(f"孤立森林训练失败: {e}")
            raise

    def train(self, data_paths: List[str]) -> Dict[str, Any]:
        """完整的训练流程"""
        try:
            # 加载数据
            data = self.load_training_data(data_paths)

            # 检测是否包含901-1000.csv数据
            has_901_1000_data = any('901-1000' in str(path) for path in data_paths)

            # 创建序列
            sequences = self.create_feature_sequences(data)

            # 训练模型 - 传递901-1000.csv检测结果
            self.train_autoencoder(sequences, has_901_1000_data)
            self.train_isolation_forest(sequences)

            # 保存模型
            self.save_models()

            return {
                'status': 'success',
                'message': '模型训练完成',
                'data_size': len(data),
                'sequence_count': len(sequences),
                'feature_count': self.feature_dim
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'训练失败: {str(e)}'
            }

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """预测异常 - 支持序列和掩码数据"""
        try:
            if self.autoencoder is None or self.isolation_forest is None:
                if not self.load_models():
                    return {'status': 'error', 'message': '模型未训练或加载失败'}

            sequences = self.create_feature_sequences(data)
            if not sequences:
                return {'status': 'error', 'message': '无法创建有效序列'}

            data_sequences = []
            mask_sequences = []
            for seq, mask in sequences:
                data_sequences.append(seq)
                mask_sequences.append(mask)

            X = np.array(data_sequences, dtype=np.float32)
            masks = np.array(mask_sequences, dtype=np.float32)
            reconstructed = self.autoencoder(X, training=False)
            masked_output = reconstructed.numpy() * masks
            masked_input = X * masks
            reconstruction_errors = np.mean((masked_input - masked_output) ** 2, axis=(1, 2))

            flattened_sequences = []
            for seq_data, seq_mask in sequences:
                masked_seq = seq_data * seq_mask
                flattened_sequences.append(masked_seq.flatten())
            isolation_scores = self.isolation_forest.decision_function(flattened_sequences)
            isolation_predictions = self.isolation_forest.predict(flattened_sequences)

            return {
                'status': 'success',
                'reconstruction_errors': reconstruction_errors.tolist(),
                'isolation_scores': isolation_scores.tolist(),
                'anomaly_predictions': (isolation_predictions == -1).tolist(),
                'sequence_count': len(sequences)
            }
        except Exception as e:
            return {'status': 'error', 'message': f'预测失败: {str(e)}'}

    def save_models(self) -> None:
        """保存结泥饼风险检测模型"""
        try:
            if self.autoencoder is not None:
                # 保存权重为 H5，便于子类模型可靠加载
                self.autoencoder.save_weights(self.model_dir / 'mud_cake_autoencoder.h5')

            if self.isolation_forest is not None:
                joblib.dump(self.isolation_forest, self.model_dir / 'mud_cake_isolation_forest.pkl')

            if self.project_scalers:
                joblib.dump(self.project_scalers, self.model_dir / 'mud_cake_scalers.pkl')

            model_info = {
                'config': self.config,
                'all_features': self.all_features,
                'feature_dim': self.feature_dim,
                'device': str(self.device),
                'model_type': 'mud_cake_risk_detection',
                'description': '结泥饼风险检测模型',
                'training_features': {
                    'available_features': self.all_features,
                    'feature_dim': self.feature_dim,
                    'feature_indices': {feature: idx for idx, feature in enumerate(self.all_features)}
                }
            }
            with open(self.model_dir / 'mud_cake_model_info.json', 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)

            # 清理不必要的元信息文件
            for fname in [
                'mud_cake_autoencoder_info.json',
                'mud_cake_isolation_forest_info.json',
                'mud_cake_scalers_info.json'
            ]:
                try:
                    p = self.model_dir / fname
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            print(f"结泥饼风险检测模型已保存到: {self.model_dir}")
        except Exception as e:
            print(f"保存结泥饼模型失败: {e}")
            raise

    def load_models(self) -> bool:
        """加载结泥饼风险检测模型"""
        try:
            model_info_path = self.model_dir / 'mud_cake_model_info.json'
            if not model_info_path.exists():
                print("结泥饼模型信息文件不存在")
                return False
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            self.config = model_info['config']
            self.all_features = model_info['all_features']
            self.feature_dim = model_info['feature_dim']

            autoencoder_path = self.model_dir / 'mud_cake_autoencoder.h5'
            if autoencoder_path.exists():
                # 重新实例化模型并加载权重
                self.autoencoder = AutoEncoder(
                    input_size=self.feature_dim,
                    hidden_size=self.config['autoencoder_params']['hidden_size'],
                    num_layers=self.config['autoencoder_params']['num_layers'],
                    dropout=self.config['autoencoder_params']['dropout']
                )
                self.autoencoder.load_weights(autoencoder_path)

            isolation_forest_path = self.model_dir / 'mud_cake_isolation_forest.pkl'
            if isolation_forest_path.exists():
                self.isolation_forest = joblib.load(isolation_forest_path)

            scalers_path = self.model_dir / 'mud_cake_scalers.pkl'
            if scalers_path.exists():
                self.project_scalers = joblib.load(scalers_path)

            print("结泥饼风险检测模型加载成功")
            return True
        except Exception as e:
            print(f"加载结泥饼模型失败: {e}")
            return False


def main():
    """主函数 - 训练模型"""
    try:
        # 设置随机种子
        np.random.seed(42)
        tf.random.set_seed(42)

        # 初始化检测器
        detector = DeepMudCakeDetector()

        # 数据文件路径 - 使用data目录下的所有CSV文件
        data_dir = Path('data')
        data_files = list(data_dir.glob('*.csv'))

        if not data_files:
            print("未找到数据文件")
            return

        print(f"找到 {len(data_files)} 个数据文件用于训练")

        # 过滤存在的文件
        existing_files = [str(f) for f in data_files if f.exists()]

        # 训练模型
        result = detector.train(existing_files)

        if result.get('status') == 'success':
            print("模型训练完成并保存成功")
        else:
            print(f"训练失败: {result.get('message', '未知错误')}")

    except Exception as e:
        print(f"训练过程出错: {e}")


if __name__ == "__main__":
    main()
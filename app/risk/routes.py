from flask import Blueprint, request, jsonify
from influxdb import InfluxDBClient
from config import Config
from app.risk.utils.clog_risk import calculate_clog_risk
from app.risk.utils.mdr_seal_risk import calculate_mdr_seal_risk
from app.risk.utils.tail_seal_risk import calculate_tail_seal_risk
from app.risk.utils.mud_cake_risk import MudCakeRiskCalculator, calculate_mud_cake_risk
import traceback
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import pymysql

CONSECUTIVE_TRIGGER_N = int(os.environ.get('CONSECUTIVE_TRIGGER_N', '3'))
MIN_RING_POINTS = int(os.environ.get('MIN_RING_POINTS', '25'))
MAX_LATEST_SCAN_RINGS = int(os.environ.get('MAX_LATEST_SCAN_RINGS', '200'))

# 创建InfluxDB客户端连接
client = InfluxDBClient(
    host=Config.INFLUXDB_HOST,
    port=Config.INFLUXDB_PORT,
    username=Config.INFLUXDB_USERNAME,
    password=Config.INFLUXDB_PASSWORD,
    database=Config.INFLUXDB_DATABASE,
    timeout=Config.INFLUXDB_TIMEOUT
)


class RiskAssessor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, 'models')
        self.mud_cake_calculator = None
        self._init_mud_cake_calculator()
        self.mud_cake_seq_len = self._load_mud_cake_sequence_length()

    def _init_mud_cake_calculator(self):
        required_files = [
            'mud_cake_autoencoder.h5',
            'mud_cake_isolation_forest.pkl',
            'mud_cake_model_info.json',
            'mud_cake_scalers.pkl'
        ]

        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(self.model_dir, file)):
                missing_files.append(file)

        if missing_files:
            self.mud_cake_calculator = None
        else:
            self.mud_cake_calculator = MudCakeRiskCalculator(model_type='unified', model_dir=self.model_dir)

    def _load_mud_cake_sequence_length(self):
        try:
            info_path = os.path.join(self.model_dir, 'mud_cake_model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                cfg = info.get('config', {})
                seq_len = int(cfg.get('sequence_length', 25))
                return seq_len if seq_len > 0 else 25
        except Exception:
            pass
        return 25

    def _format_mud_cake_point(self, point):
        property_dict = {}
        try:
            feature_list = []
            if self.mud_cake_calculator and getattr(self.mud_cake_calculator, 'model_info', None):
                tf = self.mud_cake_calculator.model_info.get('training_features', {})
                feature_list = tf.get('available_features', self.mud_cake_calculator.model_info.get('all_features', []))
            if not feature_list:
                feature_list = ['TJSD', 'TJL', 'DP_SD', 'DP_ZJ', 'DP_SS_ZTL']
        except Exception:
            feature_list = ['TJSD', 'TJL', 'DP_SD', 'DP_ZJ', 'DP_SS_ZTL']

        for field in feature_list:
            if field in ('RING', 'state'):
                continue
            val = point.get(field)
            try:
                if val is not None and val != '' and not pd.isna(val):
                    property_dict[field] = float(val)
                else:
                    property_dict[field] = 0.0
            except Exception:
                property_dict[field] = 0.0

        formatted = {
            'property': property_dict,
            'state': point.get('state', 1),
            'timestamp': point.get('ts(Asia/Shanghai)', point.get('time', '')),
            'ring': point.get('RING', 0)
        }
        return formatted

    def _build_mud_cake_sequence(self, ring_points, seq_len):
        if not isinstance(ring_points, list) or len(ring_points) == 0:
            return []
        try:
            points = [p for p in ring_points if isinstance(p, dict)]
            points.sort(key=lambda x: x.get('time'))
        except Exception:
            points = ring_points[-seq_len:]
        sliced = points[-seq_len:] if len(points) >= seq_len else points
        sequence = [self._format_mud_cake_point(p) for p in sliced]
        return sequence

    def assess_mud_cake_risk_sequence(self, ring_points):
        try:
            if self.mud_cake_calculator is None:
                raise Exception("结泥饼风险计算器不可用")
            sequence = self._build_mud_cake_sequence(ring_points, self.mud_cake_seq_len)
            if not sequence:
                raise Exception("当前环缺少有效数据用于序列评估")
            result = self.mud_cake_calculator.calculate_risk(sequence)
            probability = 0.0
            risk_level_direct = '无风险'
            if isinstance(result, dict) and result.get('status') == 'success':
                probability = float(result.get('combined_risk', 0.0))
                risk_level_direct = result.get('risk_level', '无风险')
            else:
                raise RuntimeError(f"结泥饼模型评估失败: {result.get('message', 'unknown error')}")
            mapped_score = self.map_probability_to_score(probability, "结泥饼风险")
            risk_level, measures, reason, potential_risk = self._get_risk_level_and_measures(mapped_score, "结泥饼风险")
            return {
                "risk_type": "结泥饼风险",
                "risk_level": risk_level,
                "risk_level_model": risk_level_direct,
                "risk_score": round(mapped_score, 2),
                "probability": round(probability, 2),
                "measures": measures,
                "reason": reason,
                "potential_risk": potential_risk,
            }
        except Exception as e:
            raise

    def _find_mud_cake_earliest_warning(self, ring_points):
        points = [p for p in ring_points if isinstance(p, dict)]
        points.sort(key=lambda x: x.get('time'))
        n = len(points)
        if n == 0:
            return None, None
        seq_len = self.mud_cake_seq_len
        start = 0 if n <= seq_len else seq_len - 1
        for i in range(start, n):
            window = points[max(0, i - seq_len + 1): i + 1]
            res = self.assess_mud_cake_risk_sequence(window)
            if res.get("risk_level") in ["中风险Ⅲ", "高风险Ⅳ"]:
                t = points[i].get('time') or '-'
                params = {
                    "DP_SD": points[i].get("DP_SD"),
                    "DP_ZJ": points[i].get("DP_ZJ"),
                    "TJSD": points[i].get("TJSD"),
                    "TJL": points[i].get("TJL"),
                    "DP_SS_ZTL": points[i].get("DP_SS_ZTL")
                }
                return t, params
        return None, None

    def _get_risk_level_and_measures(self, mapped_score, risk_type):
        if mapped_score <= 0.5:
            risk_level = "高风险Ⅳ"
        elif mapped_score <= 2:
            risk_level = "中风险Ⅲ"
        elif mapped_score <= 3.5:
            risk_level = "低风险Ⅱ"
        else:
            risk_level = "无风险Ⅰ"
        if risk_type == "结泥饼风险":
            potential_risk = "结泥饼预警" if risk_level != "无风险Ⅰ" else "-"
        elif risk_type == "滞排风险":
            potential_risk = "滞排预警" if risk_level != "无风险Ⅰ" else "-"
        elif risk_type == "主驱动密封失效风险":
            potential_risk = "主驱动密封失效预警" if risk_level != "无风险Ⅰ" else "-"
        elif risk_type == "盾尾密封失效风险":
            potential_risk = "盾尾密封失效预警" if risk_level != "无风险Ⅰ" else "-"
        else:
            potential_risk = "系统预警" if risk_level != "无风险Ⅰ" else "-"
        measures, reason = self._get_measures_and_reason(risk_level, risk_type)
        return risk_level, measures, reason, potential_risk

    def _get_measures_and_reason(self, risk_level, risk_type):
        """根据风险等级和类型获取措施和原因"""
        measures_reasons = {
            "结泥饼风险": {
                "无风险Ⅰ": {
                    "measures": ["维持状态：保持当前参数", "正常作业：按标准流程进行"],
                    "reason": "推进速度与贯入度匹配良好，扭矩与推进力保持稳定，土体含水率适宜，渣土流动性良好，无结泥饼风险。"
                },
                "低风险Ⅱ": {
                    "measures": ["维持参数：保持当前掘进参数", "监控刀盘扭矩：关注扭矩变化趋势"],
                    "reason": "扭矩略有波动，推进速度轻微下降，贯入度变化不大，土体粘性略有增加，但渣土排出仍然顺畅，设备负荷在安全范围内。"
                },
                "中风险Ⅲ": {
                    "measures": ["监控刀盘扭矩：每5分钟记录一次刀盘扭矩值", "增加添加剂：将添加剂浓度提高20-30%"],
                    "reason": "扭矩明显上升，推进速度下降，贯入度异常，推进力增大，土体粘性增大，渣土含水率降低，出现结泥饼初期征兆。"
                },
                "高风险Ⅳ": {
                    "measures": ["立即停机：停止掘进作业", "清理泥饼：使用高压水枪冲洗"],
                    "reason": "扭矩急剧上升至警戒值，推进速度显著下降，贯入度严重异常，推进力达到极限，渣土呈干硬状态且排出困难，设备出现卡滞现象。"
                }
            },
            "滞排风险": {
                "无风险Ⅰ": {
                    "measures": ["维持状态：保持当前掘进参数", "正常作业：按照标准流程进行"],
                    "reason": "开挖仓压力稳定，工作仓压力正常，排浆泵压力平稳，排泥系统运行平稳，渣土含水率适宜，排泥管道通畅，刀盘转速和刀盘扭矩保持稳定，渣土排出连续顺畅。"
                },
                "低风险Ⅱ": {
                    "measures": ["监控压力趋势：每15分钟记录一次压力", "调整掘进参数：适当降低推进速度"],
                    "reason": "开挖仓压力略有波动，工作仓压力轻微升高，排浆泵压力小幅变化，排泥速度略有下降，刀盘扭矩略有增加，渣土含水率略有变化，但排泥系统整体运行仍在安全范围内。"
                },
                "中风险Ⅲ": {
                    "measures": ["加强密封监控：将密封压力监测频率提高到每30分钟一次",
                                 "检查压力梯度：详细分析各道密封之间的压力梯度"],
                    "reason": "开挖仓压力不稳定，工作仓压力明显升高，排浆泵压力波动较大，排泥速度明显下降，刀盘扭矩波动明显，渣土含水率异常。"
                },
                "高风险Ⅳ": {
                    "measures": ["立即停机：停止掘进作业", "检查压力系统：全面检查气垫仓压力"],
                    "reason": "开挖仓压力异常，工作仓压力过高，排浆泵压力急剧变化，刀盘扭矩急剧上升，排泥系统出现明显阻塞，渣土几乎无法排出。"
                }
            },
            "主驱动密封失效风险": {
                "无风险Ⅰ": {
                    "measures": ["定期保养：执行密封系统保养", "正常监控：记录密封压力数据"],
                    "reason": "工作仓压力稳定，密封反馈压力正常，主驱动密封压力适宜，刀盘转速平稳，刀盘扭矩稳定，密封检测压力在安全范围内，密封系统运行参数均在理想范围内。"
                },
                "低风险Ⅱ": {
                    "measures": ["定期保养：执行密封系统保养", "正常监控：记录密封压力数据"],
                    "reason": "工作仓压力略有波动，密封反馈压力小幅变化，主驱动密封压力轻微下降，刀盘转速略有波动，刀盘扭矩小幅变化，密封检测压力轻微升高，但密封系统整体性能良好，无渗漏现象。"
                },
                "中风险Ⅲ": {
                    "measures": ["加强密封监控：将密封压力监测频率提高到每30分钟一次",
                                 "检查压力梯度：详细分析各道密封之间的压力梯度"],
                    "reason": "工作仓压力不稳定，密封反馈压力明显下降，主驱动密封压力异常，刀盘转速波动较大，刀盘扭矩不稳定，密封检测压力升高，出现间歇性压力波动。"
                },
                "高风险Ⅳ": {
                    "measures": ["立即停机：停止掘进作业", "全面检查密封压力：详细检查工作仓压力"],
                    "reason": "工作仓压力异常，密封反馈压力急剧下降，主驱动密封压力严重不足，刀盘转速异常，刀盘扭矩急剧变化，密封检测压力超出警戒值，密封腔出现明显渗漏。"
                }
            },
            "盾尾密封失效风险": {
                "无风险Ⅰ": {
                    "measures": ["正常监控：监控密封状态", "定期保养：检查密封分配器和管路"],
                    "reason": "密封压力稳定，泵出口压力正常，注脂量适宜，刀盘转速平稳，刀盘扭矩稳定，密封圈弹性良好，无渗水迹象，密封腔压力分布均匀，系统运行状态理想。"
                },
                "低风险Ⅱ": {
                    "measures": ["正常监控：监控密封状态", "定期保养：检查密封分配器和管路"],
                    "reason": "密封压力有轻微波动，泵出口压力略有变化，注脂量略有增加，刀盘转速略有波动，刀盘扭矩小幅变化，个别密封圈磨损轻微，但整体密封效果良好，无明显渗漏现象。"
                },
                "中风险Ⅲ": {
                    "measures": ["加强密封监控：每小时检查一次密封状态", "检查密封圈状态：评估密封圈磨损程度"],
                    "reason": "密封压力不稳定，泵出口压力波动较大，注脂量明显增加，刀盘转速波动较大，刀盘扭矩不稳定，密封腔压力波动频繁，部分密封圈磨损加剧。"
                },
                "高风险Ⅳ": {
                    "measures": ["立即停机：停止掘进作业", "检查密封系统状态：详细检查密封压力"],
                    "reason": "密封压力无法维持，泵出口压力异常，注脂量异常增加，刀盘转速异常，刀盘扭矩急剧变化，多处出现明显渗漏，密封圈严重磨损或损坏。"
                }
            }
        }

        default = {
            "measures": ["检查系统", "联系技术支持"],
            "reason": "系统运行状态未知，需要进一步检查。"
        }

        result = measures_reasons.get(risk_type, {}).get(risk_level, default)
        return result.get("measures", []), result.get("reason", "")

    def map_probability_to_score(self, probability, risk_type):
        if risk_type == "结泥饼风险":
            if probability >= 0.9:
                return 0.5 - (probability - 0.9) * 0.5 / 0.1
            elif probability >= 0.524:
                return 2 - (probability - 0.524) * 1.5 / (0.9 - 0.524)
            elif probability >= 0.3:
                return 3.5 - (probability - 0.3) * 1.5 / (0.524 - 0.3)
            else:
                return 4.0 - probability * 0.5 / 0.3
        elif risk_type == "滞排风险":
            if probability >= 0.9:
                return 0.5 - (probability - 0.9) * 0.5 / 0.1
            elif probability >= 0.7:
                return 2 - (probability - 0.7) * 1.5 / (0.9 - 0.7)
            elif probability >= 0.3:
                return 3.5 - (probability - 0.3) * 1.5 / (0.7 - 0.3)
            else:
                return 4.0 - probability * 0.5 / 0.3
        elif risk_type == "主驱动密封失效风险":
            if probability >= 0.95:
                return 0.5 - (probability - 0.95) * 0.5 / 0.05
            elif probability >= 0.5:
                return 2 - (probability - 0.5) * 1.5 / 0.45
            else:
                return 4.0 - probability * 2 / 0.5
        elif risk_type == "盾尾密封失效风险":
            if probability >= 0.9:
                return 0.5 - (probability - 0.9) * 0.5 / 0.1
            elif probability >= 0.5:
                return 2 - (probability - 0.5) * 1.5 / (0.9 - 0.5)
            elif probability >= 0.3:
                return 3.5 - (probability - 0.3) * 1.5 / (0.5 - 0.3)
            else:
                return 4.0 - probability * 0.5 / 0.3

    def assess_mud_cake_risk(self, data):
        try:
            if self.mud_cake_calculator is None:
                raise Exception("结泥饼风险计算器不可用")

            # 优先依据实际环号的序列进行评估
            ring_number = data.get('RING')
            ring_points = None
            if ring_number is not None:
                try:
                    ring_number = float(ring_number)
                    ring_points = query_ring_data(ring_number)
                except Exception:
                    ring_points = None

            if ring_points:
                # 使用序列评估，与 /getRiskLevel 保持一致
                seq_result = self.assess_mud_cake_risk_sequence(ring_points)
                # 取该环最后一个点作为关键参数来源（若不存在则回退到传入单点）
                last_point = ring_points[-1] if isinstance(ring_points, list) and ring_points else {}
                param_values = {
                    "DP_SD": last_point.get('DP_SD', data.get('DP_SD', 0)),
                    "DP_ZJ": last_point.get('DP_ZJ', data.get('DP_ZJ', 0)),
                    "TJSD": last_point.get('TJSD', data.get('TJSD', 0)),
                    "TJL": last_point.get('TJL', data.get('TJL', 0)),
                    "DP_SS_ZTL": last_point.get('DP_SS_ZTL', data.get('DP_SS_ZTL', 0))
                }
                return {
                    **seq_result,
                    "impact_parameters": ["刀盘转速", "刀盘扭矩", "总推力", "推进速度"],
                    "param_values": param_values,
                    "fault_cause": "渣土在刀盘面板或土仓内板结硬化，形成阻碍掘进的泥饼，导致掘进效率降低"
                }

            # 无环号或未能查询到环数据时，退回单点（仍由模型决定概率，不做人工限定）
            result = calculate_mud_cake_risk(data, calculator=self.mud_cake_calculator)
            probability = result.get('probability', 0.0)
            mapped_score = self.map_probability_to_score(probability, "结泥饼风险")
            risk_level, measures, reason, potential_risk = self._get_risk_level_and_measures(mapped_score, "结泥饼风险")
            return {
                "risk_type": "结泥饼风险",
                "risk_level": risk_level,
                "risk_score": round(mapped_score, 2),
                "probability": round(probability, 2),
                "measures": measures,
                "reason": reason,
                "potential_risk": potential_risk,
                "fault_cause": "渣土在刀盘面板或土仓内板结硬化，形成阻碍掘进的泥饼，导致掘进效率降低",
                "impact_parameters": ["刀盘转速", "刀盘扭矩", "总推力", "推进速度"],
                "param_values": {
                    "DP_SD": data.get('DP_SD', 0),
                    "DP_ZJ": data.get('DP_ZJ', 0),
                    "TJSD": data.get('TJSD', 0),
                    "TJL": data.get('TJL', 0),
                    "DP_SS_ZTL": data.get('DP_SS_ZTL', 0)
                }
            }
        except Exception as e:
            raise

    def assess_clog_risk(self, data):
        try:
            result = calculate_clog_risk(data)
            probability = result.get('probability', 0)
            mapped_score = self.map_probability_to_score(probability, "滞排风险")
            risk_level, measures, reason, potential_risk = self._get_risk_level_and_measures(mapped_score, "滞排风险")
            return {
                "risk_type": "滞排风险",
                "risk_level": risk_level,
                "risk_score": round(mapped_score, 2),
                "probability": round(probability, 3),
                "measures": measures,
                "reason": reason,
                "potential_risk": potential_risk,
                "fault_cause": "渣土改良不充分导致流动性不佳，最终在管道内发生滞排",
                "impact_parameters": ["开挖仓压力", "工作仓压力", "排浆泵P2.1进泥口压力检测"],
                "param_values": {
                    "GZC_PRS1": data.get('GZC_PRS1', 0),
                    "KWC_PRS4": data.get('KWC_PRS4', 0),
                    "PJB_JNK_PRS2.1": data.get('PJB_JNK_PRS2.1', 0)
                }
            }
        except Exception as e:
            raise

    def assess_mdr_seal_risk(self, data):
        try:
            result = calculate_mdr_seal_risk(data)
            probability = result.get('probability', 0)
            mapped_score = self.map_probability_to_score(probability, "主驱动密封失效风险")
            risk_level, measures, reason, potential_risk = self._get_risk_level_and_measures(mapped_score,
                                                                                             "主驱动密封失效风险")
            return {
                "risk_type": "主驱动密封失效风险",
                "risk_level": risk_level,
                "risk_score": round(mapped_score, 2),
                "probability": round(probability, 3),
                "measures": measures,
                "reason": reason,
                "potential_risk": potential_risk,
                "fault_cause": "主轴承密封系统因磨损老化或密封油脂压力不足，导致外部渣土或地下水浸入",
                "impact_parameters": ["工作仓压力", "油气密封反馈压力", "主驱动伸缩密封油脂压力", "齿轮油油气密封压力"],
                "param_values": {
                    "GZC_PRS1": data.get('GZC_PRS1', 0),
                    "YQMF_FK_PRS": data.get('YQMF_FK_PRS', 0),
                    "ZQD_SS_PRS1": data.get('ZQD_SS_PRS1', 0),
                    "CLY_YQ_PRS": data.get('CLY_YQ_PRS', 0),
                    "YQMF_XLJC_QPRS": data.get('YQMF_XLJC_QPRS', 0)
                }
            }
        except Exception as e:
            raise

    def assess_tail_seal_risk(self, data):
        """评估盾尾密封失效风险"""
        try:
            result = calculate_tail_seal_risk(data)
            probability = result.get('probability', 0)
            mapped_score = self.map_probability_to_score(probability, "盾尾密封失效风险")
            risk_level, measures, reason, potential_risk = self._get_risk_level_and_measures(mapped_score,
                                                                                             "盾尾密封失效风险")
            return {
                "risk_type": "盾尾密封失效风险",
                "risk_level": risk_level,
                "risk_score": round(mapped_score, 2),
                "probability": round(probability, 3),
                "measures": measures,
                "reason": reason,
                "potential_risk": potential_risk,
                "fault_cause": "盾尾密封刷密封件磨损、撕裂或者压损后，失去阻挡同步注浆浆液和地下水的能力",
                "impact_parameters": ["盾尾密封压力4.2", "盾尾密封压力4.4", "注浆压力1", "注浆压力2"],
                "param_values": {
                    "DW_PRS4.2": data.get('DW_PRS4.2', 0),
                    "DW_PRS4.4": data.get('DW_PRS4.4', 0),
                    "AYB_PRS1_1": data.get('AYB_PRS1_1', 0),
                    "AYB_PRS1_2": data.get('AYB_PRS1_2', 0)
                }
            }
        except Exception as e:
            raise

    def assess_all_risks(self, data_point):
        """评估所有风险（结泥饼在环序列中统一评估，不在单点评估）"""
        return [
            self.assess_clog_risk(data_point),
            self.assess_mdr_seal_risk(data_point),
            self.assess_tail_seal_risk(data_point)
        ]


risk_assessor = RiskAssessor()
bp = Blueprint('risk', __name__)


def _to_local_dt(time_str: str):
    """Parse InfluxDB UTC time string and convert to Asia/Shanghai local datetime.
    Supports formats with or without fractional seconds like 'YYYY-MM-DDTHH:MM:SSZ' or '...SS.sssZ'.
    """
    if not isinstance(time_str, str):
        return None
    dt = None
    try:
        dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            dt = None
    if dt is None:
        return None
    # Convert UTC ('Z') to Asia/Shanghai (+8)
    return dt + timedelta(hours=8)


def format_time_utc_to_shanghai(time_str: str) -> str:
    """Format UTC 'Z' time to 'YYYY/M/D H:MM:SS' in Asia/Shanghai (no leading zeros for Y/M/D/H)."""
    try:
        local_dt = _to_local_dt(time_str)
        if local_dt is None:
            return time_str
        return f"{local_dt.year}/{local_dt.month}/{local_dt.day} {local_dt.hour}:{local_dt.minute:02d}:{local_dt.second:02d}"
    except Exception:
        return time_str




def _round_value_2(v):
    try:
        if isinstance(v, (int, float)):
            return round(v, 2)
        if isinstance(v, str):
            return round(float(v), 2)
    except Exception:
        return v
    return v


def round_params(params):
    """将字典中的数值统一保留两位小数。"""
    if not isinstance(params, dict):
        return params
    return {k: _round_value_2(v) for k, v in params.items()}


def map_safety_to_level(safety_level: str) -> str:
    """从安全等级字符串中提取罗马数字等级，如 无风险Ⅰ -> Ⅰ。
    若无法识别则返回 '-'。"""
    try:
        if isinstance(safety_level, str):
            for sym in ("Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ"):
                if sym in safety_level:
                    return sym
    except Exception:
        pass
    return "-"


def _risk_priority(level: str) -> int:
    """将风险等级映射为优先级数值，便于比较。
    同时兼容含罗马数字的格式（如 中风险Ⅲ）和不含罗马数字的格式（如 中风险）。"""
    try:
        if isinstance(level, str):
            if "高风险Ⅳ" in level or "高风险" in level:
                return 4
            if "中风险Ⅲ" in level or "中风险" in level:
                return 3
            if "低风险Ⅱ" in level or "低风险" in level:
                return 2
            if "无风险Ⅰ" in level or "无风险" in level:
                return 1
    except Exception:
        pass
    return 0


def _aggregate_with_consecutive(lst, risk_type: str, n: int):
    """按连续触发策略汇总同一风险类型的环级结果。"""
    priorities = []
    probs = []
    for r in lst:
        try:
            priorities.append(_risk_priority(r.get('risk_level')))
            probs.append(r.get('probability', 0))
        except Exception:
            priorities.append(0)
            probs.append(0)

    def find_segment(min_priority):
        count = 0
        start = None
        for i, pri in enumerate(priorities):
            if pri >= min_priority:
                if count == 0:
                    start = i
                count += 1
                if count >= n:
                    return start, i
            else:
                count = 0
                start = None
        return None, None

    hi_start, _ = find_segment(4)
    if hi_start is not None:
        # 扩展为完整的连续高风险片段（优先级>=4）
        j = hi_start
        while j < len(priorities) and priorities[j] >= 4:
            j += 1
        hi_run_end = j - 1
        seg_items = [lst[i] for i in range(hi_start, hi_run_end + 1)]
        max_prob = max((item.get('probability', 0) for item in seg_items), default=0)
        chosen = max(seg_items, key=lambda r: r.get('probability', 0)) if seg_items else lst[-1]
        final_item = dict(chosen)
        final_item['risk_level'] = "高风险Ⅳ"
        final_item['probability'] = round(max_prob, 2)
        try:
            final_item['risk_score'] = round(risk_assessor.map_probability_to_score(max_prob, risk_type), 2)
        except Exception:
            pass
        return final_item

    mid_flags = [1 if pri == 3 else 0 for pri in priorities]
    count = 0
    mid_start = None
    for i, flag in enumerate(mid_flags):
        if flag:
            if count == 0:
                mid_start = i
            count += 1
            if count >= n:
                break
        else:
            count = 0
            mid_start = None

    if mid_start is not None and count >= n:
        j = mid_start
        while j < len(mid_flags) and mid_flags[j] == 1:
            j += 1
        mid_run_end = j - 1
        seg_items = [lst[i] for i in range(mid_start, mid_run_end + 1)]
        max_prob = max((item.get('probability', 0) for item in seg_items), default=0)
        chosen = max(seg_items, key=lambda r: r.get('probability', 0)) if seg_items else lst[-1]
        final_item = dict(chosen)
        final_item['risk_level'] = "中风险Ⅲ"
        final_item['probability'] = round(max_prob, 2)
        try:
            final_item['risk_score'] = round(risk_assessor.map_probability_to_score(max_prob, risk_type), 2)
        except Exception:
            pass
        return final_item

    # 低/无风险分支：不取平均，使用最大概率
    probs_all = [r.get('probability', 0) for r in lst]
    any_low = any(pri == 2 for pri in priorities)
    chosen = lst[-1]
    final_item = dict(chosen)

    if any_low:
        final_item['risk_level'] = "低风险Ⅱ"
        low_probs = [lst[i].get('probability', 0) for i, pri in enumerate(priorities) if pri == 2]
        max_low_prob = max(low_probs) if low_probs else (max(probs_all) if probs_all else 0)
        final_item['probability'] = round(max_low_prob, 2)
        try:
            final_item['risk_score'] = round(risk_assessor.map_probability_to_score(max_low_prob, risk_type), 2)
        except Exception:
            pass
        # 同步潜在预警文案
        if risk_type in ["滞排风险", "主驱动密封失效风险", "盾尾密封失效风险"]:
            pr_map = {
                "滞排风险": "滞排预警",
                "主驱动密封失效风险": "主驱动密封失效预警",
                "盾尾密封失效风险": "盾尾密封失效预警",
            }
            final_item['potential_risk'] = pr_map.get(risk_type, "-")
    else:
        final_item['risk_level'] = "无风险Ⅰ"
        no_probs = [lst[i].get('probability', 0) for i, pri in enumerate(priorities) if pri == 1]
        max_no_prob = max(no_probs) if no_probs else (max(probs_all) if probs_all else 0)
        final_item['probability'] = round(max_no_prob, 2)
        try:
            final_item['risk_score'] = round(risk_assessor.map_probability_to_score(max_no_prob, risk_type), 2)
        except Exception:
            pass
        final_item['potential_risk'] = "-"

    return final_item


def aggregate_ring_risk_results(original_ring_data):
    if not isinstance(original_ring_data, list):
        original_ring_data = [original_ring_data] if original_ring_data else []

    risk_by_type = {}
    errors = []
    for point in original_ring_data:
        if isinstance(point, dict):
            try:
                point_results = risk_assessor.assess_all_risks(point)
                for r in point_results:
                    rt = r.get('risk_type')
                    if rt and rt != "结泥饼风险":
                        risk_by_type.setdefault(rt, []).append(r)
            except Exception as e:
                # 跳过该时刻的无效数据点
                errors.append(str(e))
                continue

    final_results = []
    for rt, lst in risk_by_type.items():
        if not lst:
            continue
        final_item = _aggregate_with_consecutive(lst, rt, CONSECUTIVE_TRIGGER_N)
        final_results.append(final_item)

    # 结泥饼序列风险评估：若整环无效则记录错误
    try:
        seq_risk = risk_assessor.assess_mud_cake_risk_sequence(original_ring_data)
        if isinstance(seq_risk, dict):
            final_results.append(seq_risk)
    except Exception as e:
        errors.append(str(e))

    # 若所有风险类型均无有效数据结果，则抛错而不是返回固定值
    if not final_results:
        raise RuntimeError("当前环缺少有效数据用于风险评估")

    return final_results


def query_ring_data(ring_number):
    """查询指定环号的数据"""
    try:
        ring_formatted = f"{int(float(ring_number))}.00"
        query_exact = f"""
        SELECT * FROM "{Config.INFLUXDB_MEASUREMENT}" 
        WHERE "RING"='{ring_formatted}' 
        ORDER BY time ASC
        """
        query_numeric = f"""
        SELECT * FROM "{Config.INFLUXDB_MEASUREMENT}" 
        WHERE "RING"={float(ring_number)} 
        ORDER BY time ASC
        """
        ring_int_val = int(float(ring_number))
        query_range = f"""
        SELECT * FROM "{Config.INFLUXDB_MEASUREMENT}" 
        WHERE "RING" >= {ring_int_val} AND "RING" < {ring_int_val + 1} 
        ORDER BY time ASC
        """
        # 按顺序短路查询：一旦某种查询有结果，直接返回（保持输出一致）
        queries = [query_exact, query_numeric, query_range]
        for q in queries:
            try:
                res = client.query(q)
                points = list(res.get_points())
                if points:
                    seen = set()
                    unique_points = []
                    for p in points:
                        t = p.get('time')
                        if t not in seen:
                            seen.add(t)
                            unique_points.append(p)
                    unique_points.sort(key=lambda x: x.get('time'))
                    return unique_points
            except Exception:
                # 当前查询失败，继续尝试下一种查询
                pass

        return None

    except Exception as e:
        traceback.print_exc()
        return None


@bp.route('/getRiskLevel', methods=['POST'])
def get_risk_level():
    try:
        # 获取请求参数
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "请求数据为空"
            }), 400

        ring_number = data.get('RING')
        if ring_number is None:
            return jsonify({
                "status": "error",
                "message": "未提供环号参数"
            }), 400

        # 确保ring_number可以被处理，无论是整数还是字符串
        try:
            # 尝试转换为数值类型，以支持整数形式的RING参数
            ring_number = float(ring_number)
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": f"环号参数格式错误: {ring_number}"
            }), 400

        # 查询数据库获取环数据
        ring_data = query_ring_data(ring_number)
        if not ring_data:
            return jsonify({
                "status": "error",
                "message": f"无法获取环号{ring_number}的数据",
                "suggestion": "请检查环号是否正确或联系管理员"
            }), 404

        # 始终按“环的序列”进行评估：将单点统一包装为列表并使用序列评估与汇总
        original_ring_data = ring_data if isinstance(ring_data, list) else ([ring_data] if ring_data else [])
        risk_results = aggregate_ring_risk_results(original_ring_data)
        result = {
            "status": "success",
            "ring": int(ring_number),
            "mud_cake_risk": {},
            "clog_risk": {},
            "mdr_seal_risk": {},
            "tail_seal_risk": {}
        }
        # 规范化环数据为列表，便于后续关键参数填充与时间处理
        ring_data = ring_data if isinstance(ring_data, list) else ([ring_data] if ring_data else [])
        for risk in risk_results:
            risk_type = risk['risk_type']
            # 移除未使用的risk_info与参数值冗余处理

            # 根据风险类型存储结果
            if risk_type == "结泥饼风险":
                # 映射风险类型名称
                risk_type_mapped = "结泥饼"

                # 构建风险对象（直接使用已汇总的序列风险结果，避免重复推理）
                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": "通苏嘉甬",
                    "fault_measures": risk["measures"],
                    "fault_reason": risk["reason"],
                    "fault_cause": "渣土在刀盘面板或土仓内板结硬化，形成阻碍掘进的泥饼，导致掘进效率降低",
                    "impact_parameters": ["刀盘转速", "刀盘扭矩", "总推力", "推进速度"],
                    "safety_level": risk["risk_level"],
                    "risk_level": map_safety_to_level(risk["risk_level"]),
                    "risk_score": round(risk["risk_score"], 2),
                    "probability": round(risk["probability"], 2),
                    "potential_risk": risk["potential_risk"],
                    "warning_time": "-",
                    "warning_parameters": "-",
                }

                # 计算预警时间：按滑动窗口找到最早出现中/高风险的窗口末时刻
                earliest_time_raw, earliest_params = risk_assessor._find_mud_cake_earliest_warning(original_ring_data)
                if earliest_params:
                    earliest_params = round_params(earliest_params)
                risk_out["warning_time"] = (
                    format_time_utc_to_shanghai(earliest_time_raw)
                    if earliest_time_raw and earliest_time_raw != "-" else "-"
                )
                risk_out["warning_parameters"] = earliest_params if earliest_params else "-"

                result["mud_cake_risk"] = risk_out

            elif risk_type == "滞排风险":
                risk_type_mapped = "滞排"

                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": "通苏嘉甬",
                    "fault_measures": risk["measures"],
                    "fault_reason": risk["reason"],
                    "fault_cause": "渣土改良不充分导致流动性不佳，最终在管道内发生滞排",
                    "impact_parameters": ["开挖仓压力", "工作仓压力", "排浆泵P2.1进泥口压力检测"],
                    "safety_level": risk["risk_level"],
                    "risk_level": map_safety_to_level(risk["risk_level"]),
                    "risk_score": round(risk["risk_score"], 2),
                    "probability": round(risk["probability"], 2),
                    "potential_risk": risk["potential_risk"],
                    "warning_time": "-",
                    "warning_parameters": "-",
                }


                # 计算预警时间：要求“连续N次≥中风险”后触发，记录该段起点时刻
                earliest_time_raw = None
                earliest_params = None
                consec = 0
                start_time = None
                start_params = None
                for point in original_ring_data:
                    if isinstance(point, dict):
                        try:
                            risk_point = risk_assessor.assess_clog_risk(point)
                            level = risk_point.get("risk_level")
                            if level in ["中风险Ⅲ", "高风险Ⅳ"]:
                                if consec == 0:
                                    start_time = point.get("time") or "-"
                                    start_params = {
                                        "GZC_PRS1": point.get("GZC_PRS1"),
                                        "KWC_PRS4": point.get("KWC_PRS4"),
                                        "PJB_JNK_PRS2.1": point.get("PJB_JNK_PRS2.1")
                                    }
                                consec += 1
                                if consec >= CONSECUTIVE_TRIGGER_N:
                                    earliest_time_raw = start_time
                                    earliest_params = start_params
                                    break
                            else:
                                consec = 0
                                start_time = None
                                start_params = None
                        except Exception:
                            pass

                if earliest_params:
                    earliest_params = round_params(earliest_params)
                risk_out["warning_time"] = (
                    format_time_utc_to_shanghai(earliest_time_raw)
                    if earliest_time_raw and earliest_time_raw != "-" else "-"
                )
                risk_out["warning_parameters"] = earliest_params if earliest_params else "-"

                result["clog_risk"] = risk_out

            elif risk_type == "主驱动密封失效风险":
                risk_type_mapped = "主驱动密封失效"

                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": "通苏嘉甬",
                    "fault_measures": risk["measures"],
                    "fault_reason": risk["reason"],
                    "fault_cause": "主轴承密封系统因磨损老化或密封油脂压力不足，导致外部渣土或地下水浸入",
                    "impact_parameters": ["工作仓压力", "油气密封反馈压力", "主驱动伸缩密封油脂压力", "齿轮油油气密封压力"],
                    "safety_level": risk["risk_level"],
                    "risk_level": map_safety_to_level(risk["risk_level"]),
                    "risk_score": round(risk["risk_score"], 2),
                    "probability": round(risk["probability"], 2),
                    "potential_risk": risk["potential_risk"],
                    "warning_time": "-",
                    "warning_parameters": "-",
                }


                # 计算预警时间：要求“连续N次≥中风险”后触发，记录该段起点时刻
                earliest_time_raw = None
                earliest_params = None
                consec = 0
                start_time = None
                start_params = None
                for point in original_ring_data:
                    if isinstance(point, dict):
                        try:
                            risk_point = risk_assessor.assess_mdr_seal_risk(point)
                            level = risk_point.get("risk_level")
                            if level in ["中风险Ⅲ", "高风险Ⅳ"]:
                                if consec == 0:
                                    start_time = point.get("time") or "-"
                                    start_params = {
                                        "CLY_YQ_PRS": point.get("CLY_YQ_PRS"),
                                        "GZC_PRS1": point.get("GZC_PRS1"),
                                        "YQMF_FK_PRS": point.get("YQMF_FK_PRS"),
                                        "YQMF_XLJC_QPRS": point.get("YQMF_XLJC_QPRS"),
                                        "ZQD_SS_PRS1": point.get("ZQD_SS_PRS1")
                                    }
                                consec += 1
                                if consec >= CONSECUTIVE_TRIGGER_N:
                                    earliest_time_raw = start_time
                                    earliest_params = start_params
                                    break
                            else:
                                consec = 0
                                start_time = None
                                start_params = None
                        except Exception:
                            pass

                if earliest_params:
                    earliest_params = round_params(earliest_params)
                risk_out["warning_time"] = (
                    format_time_utc_to_shanghai(earliest_time_raw)
                    if earliest_time_raw and earliest_time_raw != "-" else "-"
                )
                risk_out["warning_parameters"] = earliest_params if earliest_params else "-"
                result["mdr_seal_risk"] = risk_out

            elif risk_type == "盾尾密封失效风险":
                risk_type_mapped = "盾尾密封失效"
                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": "通苏嘉甬",
                    "fault_measures": risk["measures"],
                    "fault_reason": risk["reason"],
                    "fault_cause": "盾尾密封刷密封件磨损、撕裂或者压损后，失去阻挡同步注浆浆液和地下水的能力",
                    "impact_parameters": ["盾尾密封压力4.2", "盾尾密封压力4.4", "注浆压力1", "注浆压力2"],
                    "safety_level": risk["risk_level"],
                    "risk_level": map_safety_to_level(risk["risk_level"]),
                    "risk_score": round(risk["risk_score"], 2),
                    "probability": round(risk["probability"], 2),
                    "potential_risk": risk["potential_risk"],
                    "warning_time": "-",
                    "warning_parameters": "-",
                }
                # 计算预警时间：要求“连续N次≥中风险”后触发，记录该段起点时刻
                earliest_time_raw = None
                earliest_params = None
                consec = 0
                start_time = None
                start_params = None
                for point in original_ring_data:
                    if isinstance(point, dict):
                        try:
                            risk_point = risk_assessor.assess_tail_seal_risk(point)
                            level = risk_point.get("risk_level")
                            if level in ["中风险Ⅲ", "高风险Ⅳ"]:
                                if consec == 0:
                                    start_time = point.get("time") or "-"
                                    start_params = {
                                        "AYB_PRS1_1": point.get("AYB_PRS1_1"),
                                        "AYB_PRS1_2": point.get("AYB_PRS1_2"),
                                        "DW_PRS4.2": point.get("DW_PRS4.2"),
                                        "DW_PRS4.4": point.get("DW_PRS4.4")
                                    }
                                consec += 1
                                if consec >= CONSECUTIVE_TRIGGER_N:
                                    earliest_time_raw = start_time
                                    earliest_params = start_params
                                    break
                            else:
                                consec = 0
                                start_time = None
                                start_params = None
                        except Exception:
                            pass
                if earliest_params:
                    earliest_params = round_params(earliest_params)
                risk_out["warning_time"] = (
                    format_time_utc_to_shanghai(earliest_time_raw)
                    if earliest_time_raw and earliest_time_raw != "-" else "-"
                )
                risk_out["warning_parameters"] = earliest_params if earliest_params else "-"
                result["tail_seal_risk"] = risk_out

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": f"获取数据时发生错误: {str(e)}"
        }), 500


def _parse_ring_number(ring_val):
    try:
        if ring_val is None:
            return None
        if isinstance(ring_val, (int, float)):
            return int(float(ring_val))
        if isinstance(ring_val, str):
            try:
                return int(float(ring_val))
            except Exception:
                return None
    except Exception:
        return None
    return None

# -------------------- MySQL helpers for latest ring --------------------

def _mysql_connect():
    try:
        return pymysql.connect(
            host="192.168.211.104",
            port=6446,
            user="root",
            password="7m@9X!zP2qA5LbNcRfTgYhJkM3nD4v6B",
            database="algorithm",
            charset="utf8",
            cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception as e:
        raise RuntimeError(f"MySQL连接失败: {e}")


def _parse_measures(val):
    try:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            s = val.strip()
            if s:
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        return obj
                except Exception:
                    return [i.strip() for i in s.split(",") if i.strip()]
        return []
    except Exception:
        return []


def _normalize_safety_level(level):
    try:
        if not level:
            return "无风险Ⅰ"
        s = str(level)
        if any(sym in s for sym in ("Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ")):
            return s
        if "高风险" in s:
            return "高风险Ⅳ"
        if "中风险" in s:
            return "中风险Ⅲ"
        if "低风险" in s:
            return "低风险Ⅱ"
        if "无风险" in s:
            return "无风险Ⅰ"
        return "无风险Ⅰ"
    except Exception:
        return "无风险Ⅰ"


def _fetch_latest_row(conn, table):
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT * FROM `{table}` ORDER BY `id` DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                return row
        except Exception:
            pass
        cur.execute(f"SELECT * FROM `{table}` ORDER BY `ring` DESC LIMIT 1")
        return cur.fetchone()


@bp.route('/getLatestRiskLevel', methods=['GET'])
def get_latest_risk_level():
    try:
        conn = _mysql_connect()
        tables = {
            "滞排": "clog_risk",
            "主驱动密封失效": "mdr_seal_risk",
            "结泥饼": "mud_cake_risk",
            "盾尾密封失效": "tail_seal_risk",
        }
        rows = {}
        latest_ring = None
        with conn:
            for k, t in tables.items():
                row = _fetch_latest_row(conn, t)
                if row:
                    rows[k] = row
                    r = row.get("ring")
                    if r is not None:
                        try:
                            rv = int(float(r))
                            latest_ring = rv if latest_ring is None else max(latest_ring, rv)
                        except Exception:
                            pass
        if latest_ring is None:
            return jsonify({"status": "error", "error": "MySQL未找到最新环数据"}), 404

        PROJECT_NAME = os.environ.get("PROJECT_NAME", "通苏嘉甬")

        def _get_any(row, keys):
            for key in keys:
                try:
                    val = row.get(key)
                    if val not in (None, "", "-"):
                        return val
                except Exception:
                    pass
            return None

        def _read_float(row, key, ndigits=2):
            try:
                return round(float(row.get(key, 0.0)), ndigits)
            except Exception:
                try:
                    val = row.get(key)
                    return round(float(val), ndigits) if val is not None else 0.0
                except Exception:
                    return 0.0

        def build_risk_out(row, latest_ring, meta):
            safety = _normalize_safety_level(row.get("risk_level"))
            return {
                "risk_type": meta["risk_type"],
                "ring": latest_ring,
                "project_name": PROJECT_NAME,
                "fault_measures": _parse_measures(_get_any(row, ["fault_measures", "measures"])),
                "fault_reason": _get_any(row, ["fault_reason", "reason"]) or meta["default_reason"],
                "fault_cause": meta["fault_cause"],
                "impact_parameters": meta["impact_parameters"],
                "safety_level": safety,
                "risk_level": map_safety_to_level(safety),
                "risk_score": _read_float(row, "risk_score", 2),
                "potential_risk": _get_any(row, ["potential_risk"]) or "-",
                "warning_time": _get_any(row, ["warning_time"]) or "-",
                "warning_parameters": _get_any(row, ["warning_parameters"]) or "-",
            }

        meta_map = {
            "滞排": {
                "result_key": "clog_risk",
                "risk_type": "滞排",
                "impact_parameters": ["开挖仓压力", "工作仓压力", "排浆泵P2.1进泥口压力检测"],
                "fault_cause": "渣土改良不充分导致流动性不佳，最终在管道内发生滞排",
                "default_reason": "开挖仓压力稳定，工作仓压力正常，排浆泵压力平稳，排泥系统运行平稳，渣土含水率适宜，排泥管道通畅，刀盘转速和刀盘扭矩保持稳定，渣土排出连续顺畅。",
            },
            "主驱动密封失效": {
                "result_key": "mdr_seal_risk",
                "risk_type": "主驱动密封失效",
                "impact_parameters": ["工作仓压力", "油气密封反馈压力", "主驱动伸缩密封油脂压力", "齿轮油油气密封压力"],
                "fault_cause": "主轴承密封系统因磨损老化或密封油脂压力不足，导致外部渣土或地下水浸入",
                "default_reason": "工作仓压力稳定，密封反馈压力正常，主驱动密封压力适宜，刀盘转速平稳，刀盘扭矩稳定，密封检测压力在安全范围内，密封系统运行参数均在理想范围内。",
            },
            "结泥饼": {
                "result_key": "mud_cake_risk",
                "risk_type": "结泥饼",
                "impact_parameters": ["刀盘转速", "刀盘扭矩", "总推力", "推进速度"],
                "fault_cause": "渣土在刀盘面板或土仓内板结硬化，形成阻碍掘进的泥饼，导致掘进效率降低",
                "default_reason": "扭矩略有波动，推进速度轻微下降，贯入度变化不大，土体粘性略有增加，但渣土排出仍然顺畅，设备负荷在安全范围内。",
            },
            "盾尾密封失效": {
                "result_key": "tail_seal_risk",
                "risk_type": "盾尾密封失效",
                "impact_parameters": ["盾尾密封压力4.2", "盾尾密封压力4.4", "注浆压力1", "注浆压力2"],
                "fault_cause": "盾尾密封刷密封件磨损、撕裂或者压损后，失去阻挡同步注浆浆液和地下水的能力",
                "default_reason": "密封压力稳定，泵出口压力正常，注脂量适宜，刀盘转速平稳，刀盘扭矩稳定，密封圈弹性良好，无渗水迹象，密封腔压力分布均匀，系统运行状态理想。",
            },
        }

        result = {
            "status": "success",
            "ring": latest_ring,
        }

        for label, meta in meta_map.items():
            row = rows.get(label)
            if row:
                risk_out = build_risk_out(row, latest_ring, meta)
                result[meta["result_key"]] = risk_out

        # 保持向后兼容：若某类风险无数据，仍返回空对象
        for key in ("mud_cake_risk", "clog_risk", "mdr_seal_risk", "tail_seal_risk"):
            result.setdefault(key, {})

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"获取最新环数据时发生错误: {str(e)}"}), 500

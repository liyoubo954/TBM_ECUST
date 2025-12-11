from flask import Blueprint, request, jsonify
from influxdb import InfluxDBClient
from config import Config
from app.risk.utils.clog_risk import calculate_clog_risk
from app.risk.utils.mdr_seal_risk import calculate_mdr_seal_risk
from app.risk.utils.tail_seal_risk import calculate_tail_seal_risk
from app.risk.utils.mud_cake_risk import MudCakeRiskCalculator
import traceback
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import pymysql
import threading
import queue

CONSECUTIVE_TRIGGER_N = int(os.environ.get('CONSECUTIVE_TRIGGER_N', '3'))
PROJECT_NAME = os.environ.get("PROJECT_NAME", "通苏嘉甬")
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


    def assess_mud_cake_risk_sequence(self, ring_points):
        try:
            if self.mud_cake_calculator is None:
                raise Exception("结泥饼风险计算器不可用")
            # 使用整环序列进行滑窗评估（步长=1，覆盖整环所有2环窗口）
            points = [p for p in ring_points if isinstance(p, dict)]
            points.sort(key=lambda x: x.get('time'))
            sequence = [self._format_mud_cake_point(p) for p in points]
            if not sequence:
                raise Exception("当前环缺少有效数据用于序列评估")
            result = self.mud_cake_calculator.calculate_ring_risk_sequence(sequence)
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
                "earliest_time": result.get('earliest_time', ''),
            }
        except Exception as e:
            raise

    # 移除冗余：最早预警时间由序列评估直接返回，路由侧不再重复计算

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
            elif probability >= 0.65:
                return 2 - (probability - 0.65) * 1.5 / (0.9 - 0.65)
            elif probability >= 0.3:
                return 3.5 - (probability - 0.3) * 1.5 / (0.65 - 0.3)
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

            # 依据环号读取相邻六环数据构建序列进行评估（严格与训练一致）
            ring_number = data.get('RING')
            if ring_number is None:
                raise RuntimeError("缺少环号，无法构建六环序列进行结泥饼风险评估")
            try:
                ring_number = float(ring_number)
                multi_ring_data = query_consecutive_ring_data(ring_number, count=6)
            except Exception:
                raise RuntimeError("环号解析或相邻环数据查询失败，无法进行序列评估")

            seq_result = self.assess_mud_cake_risk_sequence(multi_ring_data)
            # 取最后一环的最后一个点作为参数值来源（若不存在则回退到传入单点）
            last_point = multi_ring_data[-1] if isinstance(multi_ring_data, list) and multi_ring_data else {}
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
    """Parse time string and convert to Asia/Shanghai local datetime.
    Supports:
    - ISO UTC: 'YYYY-MM-DDTHH:MM:SSZ' or 'YYYY-MM-DDTHH:MM:SS.sssZ' (converted +8h)
    - RFC 2822: 'Sat, 11 Jan 2025 09:33:30 GMT' (converted +8h)
    - Local strings (assumed Asia/Shanghai): 'YYYY-MM-DD HH:MM:SS', 'MM/DD/YYYY HH:MM:SS'
    Also supports datetime and pandas.Timestamp objects.
    """
    # 支持 datetime/pandas.Timestamp
    try:
        import pandas as pd  # already imported at top
    except Exception:
        pd = None

    if isinstance(time_str, datetime):
        dt = time_str
        try:
            # 若是UTC（tzinfo存在且偏移为0），加8小时到上海时区
            if dt.tzinfo is not None and dt.utcoffset() == timedelta(0):
                dt = dt + timedelta(hours=8)
        except Exception:
            pass
        return dt

    if pd is not None and isinstance(time_str, pd.Timestamp):
        try:
            dt = time_str.to_pydatetime()
            if dt.tzinfo is not None and dt.utcoffset() == timedelta(0):
                dt = dt + timedelta(hours=8)
            return dt
        except Exception:
            pass

    if not isinstance(time_str, str):
        return None
    s = time_str.strip()
    patterns = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]
    for fmt in patterns:
        try:
            dt = datetime.strptime(s, fmt)
            if "Z" in s or "GMT" in s:
                dt = dt + timedelta(hours=8)
            return dt
        except ValueError:
            continue
    return None


def format_time_utc_to_shanghai(time_str: str) -> str:
    """Format time to 'YYYY/MM/DD HH:MM:SS' in Asia/Shanghai.
    Accepts UTC(Z/GMT) and local strings; applies +8h for UTC.
    """
    try:
        local_dt = _to_local_dt(time_str)
        if local_dt is None:
            return time_str
        # 返回不补零的小时：YYYY/M/D H:MM:SS（分钟/秒补零以保证两位）
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


def aggregate_ring_risk_results(original_ring_data, current_ring_number=None):
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

    # 结泥饼序列风险评估：读取连续六环数据进行“最后一环”风险判断
    try:
        if current_ring_number is not None:
            multi_ring_data = query_consecutive_ring_data(current_ring_number, count=6)
        else:
            # 无环号时退回当前环数据
            multi_ring_data = original_ring_data
        seq_risk = risk_assessor.assess_mud_cake_risk_sequence(multi_ring_data)
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


def collect_key_parameters(center_ring, fields, count=6):
    fields_list = list(fields or [])

    try:
        center = int(float(center_ring))
    except Exception:
        return []

    try:
        start_ring = center - (int(count) - 1)
    except Exception:
        start_ring = center - 5

    merged_points = []
    for r in range(start_ring, center + 1):
        pts = query_ring_data(r) or []
        for p in pts:
            try:
                local_dt = _to_local_dt(p.get("time"))
            except Exception:
                local_dt = None
            item = {"time": format_time_utc_to_shanghai(p.get("time"))}
            for f in fields_list:
                item[f] = _round_value_2(p.get(f))
            merged_points.append((local_dt, item))

    merged_points.sort(key=lambda x: x[0] or datetime.min)

    result = []
    for _, item in merged_points:
        result.append(item)

    return result


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
        risk_results = aggregate_ring_risk_results(original_ring_data, current_ring_number=ring_number)
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

                # 判断是否满足“前5环 + 当前环”的连续数据要求，不足则返回空列表
                multi_ring_data = query_consecutive_ring_data(ring_number, count=6)
                try:
                    required_rings = {int(float(ring_number)) - i for i in range(0, 6)}
                    present_rings = set()
                    for p in multi_ring_data:
                        r = p.get('RING')
                        try:
                            present_rings.add(int(float(r)))
                        except Exception:
                            pass
                    if not required_rings.issubset(present_rings):
                        result["mud_cake_risk"] = []
                        continue
                except Exception:
                    result["mud_cake_risk"] = []
                    continue

                # 构建风险对象（直接使用已汇总的序列风险结果，避免重复推理）
                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": PROJECT_NAME,
                    "fault_measures": risk["measures"],
                    "fault_reason": risk["reason"],
                    "fault_reason_analysis": get_fault_reason_analysis(risk_type_mapped, risk["risk_level"]),
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

                # 使用序列评估返回的“当前环”最早触发时刻，参数取当前环首个点
                earliest_time_raw = risk.get("earliest_time")
                # 当前环首个点参数
                earliest_params = None
                try:
                    first_point = next((p for p in multi_ring_data if int(float(p.get('RING', -1))) == int(result["ring"])), None)
                    if first_point:
                        earliest_params = {
                            "DP_SD": first_point.get("DP_SD"),
                            "DP_ZJ": first_point.get("DP_ZJ"),
                            "TJSD": first_point.get("TJSD"),
                            "TJL": first_point.get("TJL"),
                            "DP_SS_ZTL": first_point.get("DP_SS_ZTL")
                        }
                except Exception:
                    earliest_params = None

                if earliest_params:
                    earliest_params = round_params(earliest_params)
                # 仅当整环被判定为中/高风险时才输出预警时间
                final_level_text = risk.get("risk_level", "")
                should_warn_time = final_level_text in ("中风险Ⅲ", "高风险Ⅳ")
                risk_out["warning_time"] = (
                    format_time_utc_to_shanghai(earliest_time_raw)
                    if should_warn_time and earliest_time_raw and earliest_time_raw != "-" else "-"
                )
                risk_out["warning_parameters"] = earliest_params if earliest_params else "-"

                # 关键参数六环原始数据
                mud_fields = ["DP_SD", "DP_ZJ", "TJSD", "TJL", "DP_SS_ZTL"]
                risk_out["key_parameters"] = collect_key_parameters(ring_number, mud_fields, count=6)

                result["mud_cake_risk"] = risk_out

            elif risk_type == "滞排风险":
                risk_type_mapped = "滞排"

                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": PROJECT_NAME,
                    "fault_measures": risk["measures"],
                    "fault_reason": risk["reason"],
                    "fault_reason_analysis": get_fault_reason_analysis(risk_type_mapped, risk["risk_level"]),
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

                # 关键参数六环原始数据
                clog_fields = ["GZC_PRS1", "KWC_PRS4", "PJB_JNK_PRS2.1"]
                risk_out["key_parameters"] = collect_key_parameters(ring_number, clog_fields, count=6)

                result["clog_risk"] = risk_out

            elif risk_type == "主驱动密封失效风险":
                risk_type_mapped = "主驱动密封失效"

                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": PROJECT_NAME,
                    "fault_measures": risk["measures"],
                    "fault_reason": risk["reason"],
                    "fault_reason_analysis": get_fault_reason_analysis(risk_type_mapped, risk["risk_level"]),
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

                # 关键参数六环原始数据
                mdr_fields = ["CLY_YQ_PRS", "GZC_PRS1", "YQMF_FK_PRS", "YQMF_XLJC_QPRS", "ZQD_SS_PRS1"]
                risk_out["key_parameters"] = collect_key_parameters(ring_number, mdr_fields, count=6)

            elif risk_type == "盾尾密封失效风险":
                risk_type_mapped = "盾尾密封失效"
                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": PROJECT_NAME,
                    "fault_measures": risk["measures"],
                    "fault_reason": risk["reason"],
                    "fault_reason_analysis": get_fault_reason_analysis(risk_type_mapped, risk["risk_level"]),
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
                # 关键参数六环原始数据
                tail_fields = ["AYB_PRS1_1", "AYB_PRS1_2", "DW_PRS4.2", "DW_PRS4.4"]
                risk_out["key_parameters"] = collect_key_parameters(ring_number, tail_fields, count=6)
                result["tail_seal_risk"] = risk_out
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": f"获取数据时发生错误: {str(e)}"
        }), 500
class _MySQLConnectionPool:
    def __init__(self, host, port, user, password, database, charset, cursorclass, maxsize=5):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._charset = charset
        self._cursorclass = cursorclass
        self._maxsize = maxsize
        self._pool = queue.Queue()
        self._created = 0
        self._lock = threading.Lock()

    def _create_conn(self):
        return pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset=self._charset,
            cursorclass=self._cursorclass,
        )

    def acquire(self):
        try:
            conn = self._pool.get_nowait()
            return conn
        except Exception:
            with self._lock:
                if self._created < self._maxsize:
                    conn = self._create_conn()
                    self._created += 1
                    return conn
            return self._pool.get()

    def release(self, conn):
        try:
            self._pool.put(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

class _PooledConnection:
    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._pool.release(self._conn)
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
        return False

_MYSQL_POOL = None

def _mysql_connect():
    global _MYSQL_POOL
    try:
        if _MYSQL_POOL is None:
            size_env = os.environ.get("MYSQL_POOL_SIZE", "5")
            try:
                maxsize = int(size_env)
            except Exception:
                maxsize = 5
            _MYSQL_POOL = _MySQLConnectionPool(
                host="192.168.211.104",
                port=6446,
                user="root",
                password="7m@9X!zP2qA5LbNcRfTgYhJkM3nD4v6B",
                database="algorithm",
                charset="utf8",
                cursorclass=pymysql.cursors.DictCursor,
                maxsize=maxsize,
            )
        raw = _MYSQL_POOL.acquire()
        return _PooledConnection(_MYSQL_POOL, raw)
    except Exception as e:
        raise RuntimeError(f"MySQL连接失败: {e}")
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
def get_fault_reason_analysis(risk_type: str, risk_level: str) -> str:
    try:
        t = str(risk_type)
        lvl = _normalize_safety_level(risk_level)
        if "结泥饼" in t:
            mapping = {
                "无风险Ⅰ": "刀盘扭矩与刀盘转速匹配良好，总推力与推进速度稳定，贯入度无异常波动。渣土含水率与改良剂比例处于工艺目标，仓壁与面板未见黏附或板结痕迹，排泥连续顺畅，不具备泥饼形成的物理条件。",
                "低风险Ⅱ": "观察到刀盘扭矩轻微抬升、推进速度小幅下降，总推力与贯入度边界化波动，反映土体塑性增大与局部摩阻上升。面板可能出现初始黏附点，渣粒分布偏粗或含水率偏离最佳区间，短时排泥不均。若该状态持续，将促使黏附向板结演化，需要关注改良剂与水分的微调空间。",
                "中风险Ⅲ": "刀盘扭矩与总推力持续偏高，推进速度显著下降且波动加大，贯入度呈非线性起伏，指示面板与土仓内已有黏附/板结积聚。排泥阻力增大、管路压降上升、回流概率增加，渣土流变性恶化。成因多与高粘性土、改良不足或含水率偏低/偏高叠加导致黏结力提升有关，若不干预泥饼将沿面板扩展并破坏掘进稳定性与效率。",
                "高风险Ⅳ": "面板与仓内泥饼大面积形成，DP_ZJ异常抬升且伴随振动，TJSD显著受限或间断，总推力异常维持，排泥系统出现明显阻塞。进一步将导致驱动负荷过高、温升增大、密封与轴承受污染风险上升，存在设备损伤与渗水次生风险，应在可控前提下尽快处置以恢复切削与排泥通道。",
            }
            return mapping.get(lvl, "当前等级暂无更详细分析")
        if "滞排" in t:
            mapping = {
                "无风险Ⅰ": "开挖仓压力与工作仓压力稳定，排浆泵进泥口压力处于正常区间，压力梯度合理、流态均匀。渣土含水率与改良剂配比匹配，管路内无沉积或团聚迹象，排量连续可控。",
                "低风险Ⅱ": "管路摩阻略增，泵出口与进泥口压力出现轻微非同步波动，短时排量下降提示流动性边界化。渣土可能存在剪切变稀临界、团聚核初始形成或颗粒级配偏离，需关注参数微调以避免在弯头、缩径或高阻段形成稳定堵塞核。",
                "中风险Ⅲ": "进泥口压力显著抬升且泵压波动加剧，表征管内已有部分堵塞或团聚带。压力梯度异常、回流概率上升、脉动增大，局部存在沉积与再悬浮循环。根因多与改良不足、含水率不当、细粒黏性偏高或输送能级不匹配相关，需及时分段冲洗并联动调整改良与泵速以恢复连续性。",
                "高风险Ⅳ": "系统出现严重滞排或近乎完全堵塞，泵压异常、排量骤降甚至中断，压力梯度失衡并伴随强烈脉动。继续掘进将导致设备过载、密封与管路风险叠加，存在安全与质量隐患，应尽快在安全窗口下清障并重建稳定输送条件。",
            }
            return mapping.get(lvl, "当前等级暂无更详细分析")
        if "主驱动密封失效" in t:
            mapping = {
                "无风险Ⅰ": "工作仓压力、油气密封反馈压力、主驱动伸缩密封油脂压力、齿轮油油气密封压力或密封检测压力整体处于稳态，密封压力梯度合理。无渗漏迹象，油脂供给与流变性能正常，密封件弹性与磨耗处健康区间。",
                "低风险Ⅱ": "反馈或检测通道压力轻微偏移，梯度呈早期紊乱但尚能维持屏障。可能存在密封唇口初期磨耗、供脂压力波动或回油不畅等趋势，偶发微渗与油脂消耗偏快，建议提高监测敏感度并复核供脂策略与间隙设定。",
                "中风险Ⅲ": "检测压力升高与密封压力不稳并存，梯度失衡加剧；刀盘负荷与振动出现相关性波动，伴随间歇性渗漏征兆。综合表明密封功能下降、油脂屏障削弱，外界介质侵入风险上升，若不处置将危及主轴承清洁度与寿命。",
                "高风险Ⅳ": "密封系统屏障功能显著失效，压力异常与渗漏明确，油脂难以维持有效隔离。继续运行将导致土水介质进入主轴承腔，污染润滑与加速磨损，存在严重设备损伤风险，应在保护条件下尽快停机并修复或更换密封组件。",
            }
            return mapping.get(lvl, "当前等级暂无更详细分析")
        if "盾尾密封失效" in t:
            mapping = {
                "无风险Ⅰ": "盾尾密封压力与注浆压力处于工况目标区间，密封刷弹性良好、刷列贴合充分，无渗水或浆液串漏现象，与管片拼装配合稳定。",
                "低风险Ⅱ": "密封压力与注脂量轻微上调以维持密封效果，提示局部刷列出现初期磨耗或贴合度下降，个别位置可能存在微渗。若维持该趋势将加速磨耗与膜层破坏，需关注注脂分配均衡与刷列状态。",
                "中风险Ⅲ": "密封压力波动加剧，注脂量显著增加以对冲密封衰减；腔压与负荷出现不稳定相关性。可见局部渗漏与刷列磨耗加重，密封水/浆屏障降低，若不干预将扩大至环向失效并影响管片成型与对口质量。",
                "高风险Ⅳ": "刷列严重磨损、撕裂或压损导致渗漏明确，密封压力难以维持，同步注浆与防水能力显著下降。继续掘进将带来水浆侵入、质量与安全复合风险，应在保护窗口内处置并恢复密封能力。",
            }
            return mapping.get(lvl, "当前等级暂无更详细分析")
        return "当前等级暂无更详细分析"
    except Exception:
        return "当前等级暂无更详细分析"
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
        risk_type_param = request.args.get("risk_type")
        if not risk_type_param:
            body = request.get_json(silent=True) or {}
            risk_type_param = body.get("risk_type")
        if risk_type_param:
            rp = str(risk_type_param).strip()
            if rp.endswith("风险"):
                rp = rp[:-2]
            table = tables.get(rp)
            if not table:
                return jsonify({"status": "error", "error": "不支持的风险类型", "supported": list(tables.keys())}), 400
            with conn:
                row = _fetch_latest_row(conn, table)
                if row:
                    rows[rp] = row
                    r = row.get("ring")
                    if r is not None:
                        try:
                            latest_ring = int(float(r))
                        except Exception:
                            latest_ring = None
            if latest_ring is None:
                return jsonify({"status": "error", "error": "MySQL未找到指定风险的最新环数据"}), 404
        else:
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
                val = row.get(key)
                return round(float(val), ndigits) if val is not None and str(val).strip() != "" else "-"
            except Exception:
                return "-"

        def _parse_jsonish(val):
            try:
                if isinstance(val, str):
                    s = val.strip()
                    if s.startswith("[") or s.startswith("{"):
                        return json.loads(s)
                    if s == "" or s.lower() in ("null", "none"):
                        return "-"
                if val is None:
                    return "-"
                return val
            except Exception:
                return val

        def _dash(val):
            try:
                if val is None:
                    return "-"
                if isinstance(val, str):
                    s = val.strip()
                    if s == "" or s.lower() in ("null", "none"):
                        return "-"
                return val
            except Exception:
                return val

        def build_risk_out(row, meta):
            safety = row.get("safety_level")
            if safety is None or str(safety).strip() == "":
                safety = row.get("risk_level")
            ring_val = row.get("ring")
            try:
                ring_num = int(float(ring_val)) if ring_val is not None else "-"
            except Exception:
                ring_num = _dash(ring_val)

            # fault_measures
            fm = _get_any(row, ["fault_measures", "measures"])
            fm = _parse_jsonish(fm)
            fm = fm if fm not in (None, "") else "-"

            # impact_parameters 优先取数据库，否则用meta
            ip = row.get("impact_parameters")
            ip = _parse_jsonish(ip)
            if ip in (None, "-"):
                ip = meta.get("impact_parameters", "-")
            if meta.get("risk_type") == "结泥饼":
                try:
                    mp = meta.get("impact_parameters")
                    if isinstance(mp, list):
                        if isinstance(ip, list):
                            seen = set()
                            out = []
                            for x in ip + mp:
                                if x not in seen:
                                    seen.add(x)
                                    out.append(x)
                            ip = out
                        else:
                            ip = mp
                except Exception:
                    pass

            kp = row.get("key_parameters")
            kp = _parse_jsonish(kp)
            kp = kp if kp not in (None, "") else "-"

            return {
                "risk_type": meta["risk_type"],
                "ring": ring_num,
                "project_name": PROJECT_NAME,
                "key_parameters": kp,
                "safety_level": _dash(safety),
                "risk_score": _read_float(row, "risk_score", 2),
                "potential_risk": _dash(_get_any(row, ["potential_risk"]) or "-"),
                "fault_reason_analysis": _dash(row.get("fault_reason_analysis")),
                "fault_cause": _dash(row.get("fault_cause")),
                "fault_measures": fm,
                "impact_parameters": ip,
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
                "impact_parameters": ["刀盘转速", "刀盘扭矩", "总推力", "推进速度", "刀盘伸缩总推力"],
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

        result = {}

        if risk_type_param:
            rp = str(risk_type_param).strip()
            if rp.endswith("风险"):
                rp = rp[:-2]
            meta = meta_map.get(rp)
            row = rows.get(rp)
            if meta and row:
                risk_out = build_risk_out(row, meta)
                result[meta["result_key"]] = risk_out
        else:
            for label, meta in meta_map.items():
                row = rows.get(label)
                if row:
                    risk_out = build_risk_out(row, meta)
                    result[meta["result_key"]] = risk_out

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"获取最新环数据时发生错误: {str(e)}"}), 500
def query_consecutive_ring_data(center_ring, count=6):
    """按中心环号向前取 count-1 个环并包含中心环，共计 count 个连续环的数据，按时间升序合并返回。
    若某些前置环缺失，则仅返回可用的向前数据，不进行向后补齐。"""
    try:
        ring_int = int(float(center_ring))
        start_ring = ring_int - (count - 1)
        combined = []
        for r in range(start_ring, ring_int + 1):
            pts = query_ring_data(r)
            if pts:
                combined.extend(pts)
        # 去重并按时间升序
        seen = set()
        unique_points = []
        for p in combined:
            t = p.get('time')
            if t not in seen:
                seen.add(t)
                unique_points.append(p)
        unique_points.sort(key=lambda x: x.get('time'))
        return unique_points
    except Exception:
        # 安全回退：返回空列表，避免未定义变量导致异常
        return []

def _dash(val):
    try:
        if val is None:
            return "-"
        if isinstance(val, str):
            s = val.strip()
            if s == "" or s.lower() in ("null", "none"):
                return "-"
        return val
    except Exception:
        return val

@bp.route('/getAllRiskRecords', methods=['GET'])
def get_all_risk_records():
    try:
        conn = _mysql_connect()
        tables = {
            "滞排": "clog_risk",
            "主驱动密封失效": "mdr_seal_risk",
            "结泥饼": "mud_cake_risk",
            "盾尾密封失效": "tail_seal_risk",
        }
        records = []
        high_count = 0
        mid_count = 0
        low_count = 0
        with conn:
            with conn.cursor() as cur:
                for label, table in tables.items():
                    cur.execute(f"SELECT `ring`,`warning_time`,`warning_parameters`,`fault_reason`,`fault_measures`,`risk_level` FROM `{table}`")
                    rows = cur.fetchall() or []
                    def _is_target_level(v):
                        try:
                            s = str(v)
                            return ("Ⅱ" in s) or ("Ⅲ" in s) or ("Ⅳ" in s) or ("II" in s) or ("III" in s) or ("IV" in s)
                        except Exception:
                            return False
                    for row in rows:
                        ring_val = row.get("ring")
                        try:
                            ring_num = int(float(ring_val)) if ring_val is not None else "-"
                        except Exception:
                            ring_num = _dash(ring_val)
                        wp = row.get("warning_parameters")
                        try:
                            if isinstance(wp, str) and (wp.strip().startswith("[") or wp.strip().startswith("{")):
                                wp = json.loads(wp)
                        except Exception:
                            pass
                        if wp is None or (isinstance(wp, str) and (wp.strip() == "" or wp.strip().lower() in ("null", "none"))):
                            wp = "-"
                        fm = row.get("fault_measures")
                        try:
                            if isinstance(fm, str) and (fm.strip().startswith("[") or fm.strip().startswith("{")):
                                fm = json.loads(fm)
                        except Exception:
                            pass
                        if fm is None or (isinstance(fm, str) and (fm.strip() == "" or fm.strip().lower() in ("null", "none"))):
                            fm = "-"
                        level_raw = row.get("risk_level")
                        if not _is_target_level(level_raw):
                            continue
                        s = str(level_raw)
                        if ("Ⅳ" in s) or ("IV" in s):
                            high_count += 1
                        elif ("Ⅲ" in s) or ("III" in s):
                            mid_count += 1
                        elif ("Ⅱ" in s) or ("II" in s):
                            low_count += 1
                        level_out = _dash(level_raw)
                        fault_reason_out = _dash(row.get("fault_reason"))
                        warning_time_out = _dash(row.get("warning_time"))
                        rec = {
                            "risk_type": label,
                            "warning_time": warning_time_out,
                            "ring": ring_num,
                            "warning_parameters": wp,
                            "risk_level": level_out,
                            "fault_reason": fault_reason_out,
                            "fault_measures": fm,
                        }
                        records.append(rec)
        def _ring_num(rec):
            r = rec.get("ring")
            try:
                return int(float(r))
            except Exception:
                return -1
        records.sort(key=lambda rec: _ring_num(rec), reverse=True)

        return jsonify({
            "records": records,
            "high_risk_count": high_count,
            "mid_risk_count": mid_count,
            "low_risk_count": low_count,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"获取全部风险记录时发生错误: {str(e)}"}), 500

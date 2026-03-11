from flask import Blueprint, request, jsonify
from influxdb import InfluxDBClient
from app.risk.utils.clog_risk import calculate_clog_risk
from app.risk.utils.mdr_seal_risk import calculate_mdr_seal_risk
from app.risk.utils.tail_seal_risk import calculate_tail_seal_risk
from app.risk.utils.mud_cake_risk import MudCakeRiskCalculator, MODEL_FEATURE_MAP
import traceback
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import pymysql
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

CONSECUTIVE_TRIGGER_N = int(os.environ.get('CONSECUTIVE_TRIGGER_N', '3'))
PROJECT_NAME = os.environ.get("PROJECT_NAME", "通苏嘉甬")
INFLUXDB_HOST = os.environ.get('INFLUXDB_HOST') or '192.168.211.108'
INFLUXDB_PORT = int(os.environ.get('INFLUXDB_PORT') or 38086)
INFLUXDB_USERNAME = os.environ.get('INFLUXDB_USERNAME') or 'admin'
INFLUXDB_PASSWORD = os.environ.get('INFLUXDB_PASSWORD') or 'FZaStb0cXFuFbehPBM6YHCiuAAX6QIXr'
INFLUXDB_DATABASE = os.environ.get('INFLUXDB_DATABASE') or 'algorithm'
INFLUXDB_MEASUREMENT = os.environ.get('INFLUXDB_MEASUREMENT') or 'tsjy_dz1360_riskwarning'
INFLUXDB_TIMEOUT = int(os.environ.get('INFLUXDB_TIMEOUT') or 10)
client = InfluxDBClient(
    host=INFLUXDB_HOST,
    port=INFLUXDB_PORT,
    username=INFLUXDB_USERNAME,
    password=INFLUXDB_PASSWORD,
    database=INFLUXDB_DATABASE,
    timeout=INFLUXDB_TIMEOUT
)

POINT_MAP_CACHE = {}
RISK_CONFIG = {
    "clog_risk": {
        "name": "滞排风险",
        "fields": ["WorkCham.Pres.01", "ExcavCham.Pres.04", "SlurryPump.P2.1.MudIn.Pres"],
        "map": {
            "WorkCham.Pres.01": "工作舱压力01",
            "ExcavCham.Pres.04": "开挖舱压力04",
            "SlurryPump.P2.1.MudIn.Pres": "排浆泵P2.1进泥口压力",
        },
        "units": {
            "工作舱压力01": "bar",
            "开挖舱压力04": "bar",
            "排浆泵P2.1进泥口压力": "bar",
        }
    },
    "mdr_seal_risk": {
        "name": "主驱动密封失效风险",
        "fields": ["WorkCham.Pres.01", "MB.InSeal.Grs.Pres", "MD.TelSeal.Grs.Pres.01", "GOil.OilGasSeal.Pres", "OilGasSeal.LeakDetCham.Pres"],
        "map": {
            "WorkCham.Pres.01": "工作舱压力01",
            "MB.InSeal.Grs.Pres": "主轴承内密封油脂压力",
            "MD.TelSeal.Grs.Pres.01": "主驱动伸缩密封油脂压力01",
            "GOil.OilGasSeal.Pres": "齿轮油油气密封压力",
            "OilGasSeal.LeakDetCham.Pres": "油气密封泄露检测腔压力",
        },
        "units": {
            "工作舱压力01": "bar",
            "主轴承内密封油脂压力": "bar",
            "主驱动伸缩密封油脂压力01": "bar",
            "齿轮油油气密封压力": "bar",
            "油气密封泄露检测腔压力": "bar",
        }
    },
    "tail_seal_risk": {
        "name": "盾尾密封失效风险",
        "fields": ["Liquid.Vlv.A.OutPres.01", "Liquid.Vlv.A.OutPres.02", "ShieldTail.Seal.Rear.Pres.02", "ShieldTail.Seal.Rear.Pres.04"],
        "map": {
            "Liquid.Vlv.A.OutPres.01": "A液阀出口压力01",
            "Liquid.Vlv.A.OutPres.02": "A液阀出口压力02",
            "ShieldTail.Seal.Rear.Pres.02": "盾尾密封后压力02",
            "ShieldTail.Seal.Rear.Pres.04": "盾尾密封后压力04",
        },
        "units": {
            "A液阀出口压力01": "bar",
            "A液阀出口压力02": "bar",
            "盾尾密封后压力02": "bar",
            "盾尾密封后压力04": "bar",
        }
    },
    "mud_cake_risk": {
        "name": "结泥饼风险",
        "fields": ["CutterHead.Spd", "CutterHead.Torque", "Prop.Spd", "Thrust", "CutterHead.Total.Extr.Pres"],
        "map": {
            "CutterHead.Spd": "刀盘转速",
            "CutterHead.Torque": "刀盘扭矩",
            "Prop.Spd": "推进速度",
            "Thrust": "推力",
            "CutterHead.Total.Extr.Pres": "刀盘总挤压力",
        },
        "units": {
            "刀盘转速": "rpm",
            "刀盘扭矩": "kNm",
            "推进速度": "mm/min",
            "推力": "kN",
            "刀盘总挤压力": "kN",
        }
    }
}

RISK_TABLES = {
    "滞排": "clog_risk",
    "主驱动密封失效": "mdr_seal_risk",
    "结泥饼": "mud_cake_risk",
    "盾尾密封失效": "tail_seal_risk",
}


def _find_earliest_warning(original_ring_data, assessor_func, consecutive_n, fields):
    """查找连续N次高风险的最早预警时间及参数"""
    earliest_time_raw = None
    earliest_params = None
    consec = 0
    start_time = None
    start_params = None

    for point in original_ring_data:
        if isinstance(point, dict):
            try:
                risk_point = assessor_func(point)
                level = risk_point.get("risk_level")
                if level in ["低风险Ⅱ", "中风险Ⅲ", "高风险Ⅳ"]:
                    if consec == 0:
                        start_time = point.get("time") or "-"
                        start_params = {k: point.get(k) for k in fields}
                    consec += 1
                    if consec >= consecutive_n:
                        earliest_time_raw = start_time
                        earliest_params = start_params
                        break
                else:
                    consec = 0
                    start_time = None
                    start_params = None
            except Exception:
                pass
    return earliest_time_raw, earliest_params


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
            source_key = MODEL_FEATURE_MAP.get(field, field)
            val = point.get(source_key)
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
            'ring': point.get('Ring.No', point.get('RING', 0))
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
                    "reason": "刀盘扭矩略有波动，推进速度轻微下降，贯入度变化不大，土体粘性略有增加，但渣土排出仍然顺畅，设备负荷在安全范围内。"
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
                    "reason": "开挖舱压力稳定，工作舱压力正常，排浆泵压力平稳，排泥系统运行平稳，渣土含水率适宜，排泥管道通畅，刀盘转速和刀盘扭矩保持稳定，渣土排出连续顺畅。"
                },
                "低风险Ⅱ": {
                    "measures": ["监控压力趋势：每15分钟记录一次压力", "调整掘进参数：适当降低推进速度"],
                    "reason": "开挖舱压力略有波动，工作舱压力轻微升高，排浆泵压力小幅变化，排泥速度略有下降，刀盘扭矩略有增加，渣土含水率略有变化，但排泥系统整体运行仍在安全范围内。"
                },
                "中风险Ⅲ": {
                    "measures": ["加强密封监控：将密封压力监测频率提高到每30分钟一次",
                                 "检查压力梯度：详细分析各道密封之间的压力梯度"],
                    "reason": "开挖舱压力不稳定，工作舱压力明显升高，排浆泵压力波动较大，排泥速度明显下降，刀盘扭矩波动明显，渣土含水率异常。"
                },
                "高风险Ⅳ": {
                    "measures": ["立即停机：停止掘进作业", "检查压力系统：全面检查气垫舱压力"],
                    "reason": "开挖舱压力异常，工作舱压力过高，排浆泵压力急剧变化，刀盘扭矩急剧上升，排泥系统出现明显阻塞，渣土几乎无法排出。"
                }
            },
            "主驱动密封失效风险": {
                "无风险Ⅰ": {
                    "measures": ["定期保养：执行密封系统保养", "正常监控：记录密封压力数据"],
                    "reason": "工作舱压力稳定，密封反馈压力正常，主驱动密封压力适宜，刀盘转速平稳，刀盘扭矩稳定，密封检测压力在安全范围内，密封系统运行参数均在理想范围内。"            
                },
                "低风险Ⅱ": {
                    "measures": ["定期保养：执行密封系统保养", "正常监控：记录密封压力数据"],
                    "reason": "工作舱压力略有波动，密封反馈压力小幅变化，主驱动密封压力轻微下降，刀盘转速略有波动，刀盘扭矩小幅变化，密封检测压力轻微升高，但密封系统整体性能良好，无渗漏现象。"    
                },
                "中风险Ⅲ": {
                    "measures": ["加强密封监控：将密封压力监测频率提高到每30分钟一次",
                                 "检查压力梯度：详细分析各道密封之间的压力梯度"],
                    "reason": "开挖舱压力不稳定，密封反馈压力明显下降，主驱动密封压力异常，刀盘转速波动较大，刀盘扭矩不稳定，密封检测压力升高，出现间歇性压力波动。"
                },
                "高风险Ⅳ": {
                    "measures": ["立即停机：停止掘进作业", "全面检查密封压力：详细检查工作舱压力"],
                    "reason": "工作舱压力异常，密封反馈压力急剧下降，主驱动密封压力严重不足，刀盘转速异常，刀盘扭矩急剧变化，密封检测压力超出警戒值，密封腔出现明显渗漏。    "
                }
            },
            "盾尾密封失效风险": {
                "无风险Ⅰ": {
                    "measures": ["正常监控：监控密封状态", "定期保养：检查密封分配器和管路"],
                    "reason": "密封压力稳定，工作舱压力正常，注脂量适宜，刀盘转速平稳，刀盘扭矩稳定，密封圈弹性良好，无渗水迹象，密封腔压力分布均匀，系统运行状态理想。"
                },
                "低风险Ⅱ": {
                    "measures": ["正常监控：监控密封状态", "定期保养：检查密封分配器和管路"],
                    "reason": "密封压力有轻微波动，工作舱压力略有变化，注脂量略有增加，刀盘转速略有波动，刀盘扭矩小幅变化，个别密封圈磨损轻微，但整体密封效果良好，无明显渗漏现象。"
                },
                "中风险Ⅲ": {
                    "measures": ["加强密封监控：每小时检查一次密封状态", "检查密封圈状态：评估密封圈磨损程度"],
                    "reason": "密封压力不稳定，工作舱压力波动较大，注脂量明显增加，刀盘转速波动较大，刀盘扭矩不稳定，密封腔压力波动频繁，部分密封圈磨损加剧。"
                },
                "高风险Ⅳ": {
                    "measures": ["立即停机：停止掘进作业", "检查密封系统状态：详细检查工作舱压力"],
                    "reason": "密封压力无法维持，工作舱压力异常，注脂量异常增加，刀盘转速异常，刀盘扭矩急剧变化，多处出现明显渗漏，密封圈严重磨损或损坏。"
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

    def reverse_map_score_to_probability(self, risk_score, risk_type):
        """
        Reverse mapping from risk_score to probability based on risk_type thresholds.
        """
        if risk_type == "结泥饼风险":
            # thresholds: 0.9, 0.65, 0.3
            if risk_score <= 0.5:  # p >= 0.9
                return 0.9 + (0.5 - risk_score) * 0.2
            elif risk_score <= 2:  # 0.65 <= p < 0.9
                return 0.65 + (2 - risk_score) / 6
            elif risk_score <= 3.5:  # 0.3 <= p < 0.65
                return 0.3 + (3.5 - risk_score) * (0.35 / 1.5)
            else:  # p < 0.3
                return (4.0 - risk_score) * 0.6

        elif risk_type == "滞排风险":
            # thresholds: 0.9, 0.7, 0.3
            if risk_score <= 0.5:  # p >= 0.9
                return 0.9 + (0.5 - risk_score) * 0.2
            elif risk_score <= 2:  # 0.7 <= p < 0.9
                return 0.7 + (2 - risk_score) * (0.2 / 1.5)
            elif risk_score <= 3.5:  # 0.3 <= p < 0.7
                return 0.3 + (3.5 - risk_score) * (0.4 / 1.5)
            else:  # p < 0.3
                return (4.0 - risk_score) * 0.6

        elif risk_type == "主驱动密封失效风险":
            # thresholds: 0.95, 0.5, (implicit 0.125)
            if risk_score <= 0.5:  # p >= 0.95
                return 0.95 + (0.5 - risk_score) * 0.1
            elif risk_score <= 2:  # 0.5 <= p < 0.95
                return 0.5 + (2 - risk_score) * 0.3
            else:  # p < 0.5
                return (4.0 - risk_score) * 0.25

        elif risk_type == "盾尾密封失效风险":
            # thresholds: 0.9, 0.5, 0.3
            if risk_score <= 0.5:  # p >= 0.9
                return 0.9 + (0.5 - risk_score) * 0.2
            elif risk_score <= 2:  # 0.5 <= p < 0.9
                return 0.5 + (2 - risk_score) * (0.4 / 1.5)
            elif risk_score <= 3.5:  # 0.3 <= p < 0.5
                return 0.3 + (3.5 - risk_score) * (0.2 / 1.5)
            else:  # p < 0.3
                return (4.0 - risk_score) * 0.6

        return 0.0

    def get_probability_thresholds(self, risk_type):
        """
        获取指定风险类型的概率阈值
        Returns:
            (potential_risk_threshold, normal_probability_threshold)
            potential_risk_threshold: 中风险(III)起始阈值
            normal_probability_threshold: 低风险(II)起始阈值 (低于此值为无风险)
        """
        if risk_type == "结泥饼风险":
            return 0.65, 0.3
        elif risk_type == "滞排风险":
            return 0.7, 0.3
        elif risk_type == "主驱动密封失效风险":
            return 0.5, 0.125
        elif risk_type == "盾尾密封失效风险":
            return 0.5, 0.3
        else:
            return 0.5, 0.3

    def _assess_generic(self, data, calc_func, risk_type, fault_cause):
        """通用风险评估逻辑。"""
        try:
            result = calc_func(data)
            probability = result.get('probability', 0)
            mapped_score = self.map_probability_to_score(probability, risk_type)
            risk_level, measures, reason, potential_risk = self._get_risk_level_and_measures(mapped_score, risk_type)
            return {
                "risk_type": risk_type,
                "risk_level": risk_level,
                "risk_score": round(mapped_score, 2),
                "probability": round(probability, 3),
                "measures": measures,
                "reason": reason,
                "potential_risk": potential_risk,
                "fault_cause": fault_cause,
            }
        except Exception as e:
            raise

    def assess_clog_risk(self, data):
        return self._assess_generic(data, calculate_clog_risk, "滞排风险", "渣土改良不充分导致流动性不佳，最终在管道内发生滞排")

    def assess_mdr_seal_risk(self, data):
        return self._assess_generic(data, calculate_mdr_seal_risk, "主驱动密封失效风险", "主轴承密封系统因磨损老化或密封油脂压力不足，导致外部渣土或地下水浸入")

    def assess_tail_seal_risk(self, data):
        """评估盾尾密封失效风险"""
        return self._assess_generic(data, calculate_tail_seal_risk, "盾尾密封失效风险", "盾尾密封刷密封件磨损、撕裂或者压损后，失去阻挡同步注浆浆液和地下水的能力")

    def assess_all_risks(self, data_point):
        """评估所有风险（结泥饼在环序列中统一评估，不在单点评估）"""
        return [
            self.assess_clog_risk(data_point),
            self.assess_mdr_seal_risk(data_point),
            self.assess_tail_seal_risk(data_point)
        ]


def _clean_json_val(val, dash_if_empty=True):
    """统一清理和解析 JSON 字符串、处理空值。"""
    try:
        if val is None:
            return "-" if dash_if_empty else None
        
        if isinstance(val, (list, dict)):
            # 如果已经是对象，则递归清理其中的元素或直接返回
            if isinstance(val, list):
                return "，".join([str(_clean_json_val(x, False)) for x in val])
            return val
            
        if isinstance(val, str):
            s = val.strip()
            if s == "" or s.lower() in ("null", "none"):
                return "-" if dash_if_empty else None
            
            # 处理 JSON 编码的字符串 (e.g., "\"text\"", "[1,2,3]", "{\"a\":1}")
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")) or \
               s.startswith("[") or s.startswith("{"):
                try:
                    d = json.loads(s)
                    return _clean_json_val(d, dash_if_empty)
                except Exception:
                    # 如果解析失败，去掉引号或原样返回
                    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                        return s[1:-1]
                    return s
            return s
            
        return str(val)
    except Exception:
        return "-" if dash_if_empty else str(val)


def _to_plain_string(v):
    return _clean_json_val(v)


def _dash(val):
    return _clean_json_val(val)


def _decode_json_string(val):
    return _clean_json_val(val)


def _parse_jsonish(val):
    return _clean_json_val(val)


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


def aggregate_ring_risk_results(original_ring_data, current_ring_number=None, multi_ring_data=None):
    if not isinstance(original_ring_data, list):
        original_ring_data = [original_ring_data] if original_ring_data else []

    risk_by_type = {}
    for point in original_ring_data:
        if isinstance(point, dict):
            try:
                point_results = risk_assessor.assess_all_risks(point)
                for r in point_results:
                    rt = r.get('risk_type')
                    if rt and rt != "结泥饼风险":
                        risk_by_type.setdefault(rt, []).append(r)
            except Exception as e:
                pass
                continue

    final_results = []
    for rt, lst in risk_by_type.items():
        if not lst:
            continue
        final_item = _aggregate_with_consecutive(lst, rt, CONSECUTIVE_TRIGGER_N)
        final_results.append(final_item)

    # 结泥饼序列风险评估：读取连续六环数据进行“最后一环”风险判断
    try:
        if multi_ring_data is None:
            if current_ring_number is not None:
                multi_ring_data = query_consecutive_ring_data(current_ring_number, count=6)
            else:
                multi_ring_data = original_ring_data
        seq_risk = risk_assessor.assess_mud_cake_risk_sequence(multi_ring_data)
        if isinstance(seq_risk, dict):
            final_results.append(seq_risk)
    except Exception as e:
        pass

    # 若所有风险类型均无有效数据结果，则抛错而不是返回固定值
    if not final_results:
        raise RuntimeError("当前环缺少有效数据用于风险评估")

    return final_results


def query_ring_data(ring_number):
    """查询指定环号的数据"""
    try:
        ring_formatted = f"{int(float(ring_number))}.00"
        query_exact = f"""
        SELECT * FROM "{INFLUXDB_MEASUREMENT}" 
        WHERE "Ring.No"='{ring_formatted}' 
        ORDER BY time ASC
        """
        query_numeric = f"""
        SELECT * FROM "{INFLUXDB_MEASUREMENT}" 
        WHERE "Ring.No"={float(ring_number)} 
        ORDER BY time ASC
        """
        ring_int_val = int(float(ring_number))
        query_range = f"""
        SELECT * FROM "{INFLUXDB_MEASUREMENT}" 
        WHERE "Ring.No" >= {ring_int_val} AND "Ring.No" < {ring_int_val + 1} 
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


def collect_key_parameters(center_ring, fields, count=6, preload=None):
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
        if isinstance(preload, dict):
            pts = preload.get(r) or []
        else:
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


def _build_mud_cake_fallback_risk(center_ring, preload_map, multi_ring_data=None):
    ring_int = int(float(center_ring))
    config = RISK_CONFIG["mud_cake_risk"]
    safety_level = "无风险Ⅰ"
    risk_level = map_safety_to_level(safety_level)
    measures, reason = risk_assessor._get_measures_and_reason(safety_level, "结泥饼风险")
    if isinstance(measures, list):
        fm = "，".join([str(x) for x in measures])
    else:
        fm = str(measures) if measures is not None else "-"
    imp = "，".join([config["map"][k] for k in config["fields"]])
    mud_fields = config["fields"]
    mud_map = config["map"]
    mud_units = config["units"]
    key_params = _rename_keys(
        collect_key_parameters(center_ring, mud_fields, count=6, preload=preload_map), mud_map
    )
    key_params = _with_value_and_unit(key_params, mud_units)
    if multi_ring_data is None:
        multi_ring_data = query_consecutive_ring_data(center_ring, count=6)
    fallback_time = None
    if isinstance(multi_ring_data, list):
        first_point = next(
            (p for p in multi_ring_data if int(float(p.get('Ring.No', p.get('RING', -1)))) == int(ring_int)),
            None
        )
        if first_point:
            fallback_time = first_point.get("time")
    warning_time = format_time_utc_to_shanghai(fallback_time) if fallback_time else "-"
    return {
        "risk_type": "结泥饼",
        "ring": ring_int,
        "project_name": PROJECT_NAME,
        "fault_measures": fm,
        "fault_reason": reason,
        "fault_reason_analysis": get_fault_reason_analysis("结泥饼", safety_level),
        "fault_cause": "渣土在刀盘面板或土仓内板结硬化，形成阻碍掘进的泥饼，导致掘进效率降低",
        "impact_parameters": imp,
        "safety_level": safety_level,
        "risk_level": risk_level,
        "risk_score": 4.0,
        "probability": 0.0,
        "potential_risk": "-",
        "warning_time": warning_time,
        "warning_parameters": "-",
        "key_parameters": key_params,
    }


@bp.route('/getRiskLevel', methods=['POST'])
def get_risk_level():
    try:
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

        try:
            ring_number = float(ring_number)
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": f"环号参数格式错误: {ring_number}"
            }), 400

        ring_data = query_ring_data(ring_number)
        if not ring_data:
            return jsonify({
                "status": "error",
                "message": f"无法获取环号{ring_number}的数据",
                "suggestion": "请检查环号是否正确或联系管理员"
            }), 404

        original_ring_data = ring_data if isinstance(ring_data, list) else ([ring_data] if ring_data else [])
        multi_ring_data = query_consecutive_ring_data(ring_number, count=6)
        risk_results = aggregate_ring_risk_results(original_ring_data, current_ring_number=ring_number, multi_ring_data=multi_ring_data)

        center_int = int(float(ring_number))
        start_ring = center_int - 5
        preload_map = {}
        rings = list(range(start_ring, center_int + 1))
        preload_map[center_int] = original_ring_data
        rings_without_center = [r for r in rings if r != center_int]
        if rings_without_center:
            max_workers = min(len(rings_without_center), 6)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_to_ring = {ex.submit(query_ring_data, r): r for r in rings_without_center}
                for fut in as_completed(future_to_ring):
                    r = future_to_ring[fut]
                    try:
                        preload_map[r] = fut.result() or []
                    except Exception:
                        preload_map[r] = []
        result = {
            "ring": int(ring_number),
            "mud_cake_risk": {},
            "clog_risk": {},
            "mdr_seal_risk": {},
            "tail_seal_risk": {}
        }
        ring_data = ring_data if isinstance(ring_data, list) else ([ring_data] if ring_data else [])
        def _build_risk_item(risk_obj, risk_type_label, risk_config_key, assessor_func, fault_cause):
            """通用风险输出构造辅助函数。"""
            config = RISK_CONFIG[risk_config_key]
            
            # 处理措施
            fm = risk_obj.get("measures")
            fm = _clean_json_val(fm)
            
            # 处理影响参数
            imp = "，".join([config["map"].get(k, k) for k in config["fields"]])
            
            risk_out = {
                "risk_type": risk_type_label,
                "ring": int(result["ring"]),
                "project_name": PROJECT_NAME,
                "fault_measures": fm,
                "fault_reason": risk_obj.get("reason"),
                "fault_reason_analysis": get_fault_reason_analysis(risk_type_label, risk_obj["risk_level"]),
                "fault_cause": fault_cause,
                "impact_parameters": imp,
                "safety_level": risk_obj["risk_level"],
                "risk_level": map_safety_to_level(risk_obj["risk_level"]),
                "risk_score": round(risk_obj["risk_score"], 2),
                "probability": round(risk_obj["probability"], 2),
                "potential_risk": risk_obj["potential_risk"],
                "warning_time": "-",
                "warning_parameters": "-",
            }

            # 查找最早预警
            earliest_time_raw, earliest_params = _find_earliest_warning(
                original_ring_data, assessor_func, CONSECUTIVE_TRIGGER_N, config["fields"]
            )

            if earliest_params:
                earliest_params = round_params(earliest_params)
                
            # 设置预警时间
            if earliest_time_raw and earliest_time_raw != "-":
                risk_out["warning_time"] = format_time_utc_to_shanghai(earliest_time_raw)
            elif original_ring_data:
                # 备选使用该环第一条数据的时间
                t0 = original_ring_data[0].get("time")
                if t0:
                    risk_out["warning_time"] = format_time_utc_to_shanghai(t0)

            # 设置预警参数
            if earliest_params:
                wp_raw = {config["map"].get(k, k): earliest_params.get(k) for k in earliest_params}
                risk_out["warning_parameters"] = _append_units_to_map(wp_raw, config["units"])
            else:
                risk_out["warning_parameters"] = "-"
                
            # 设置历史关键参数
            risk_out["key_parameters"] = _rename_keys(
                collect_key_parameters(ring_number, config["fields"], count=6, preload=preload_map), config["map"])
            risk_out["key_parameters"] = _with_value_and_unit(risk_out["key_parameters"], config["units"])
            
            return risk_out

        for risk in risk_results:
            risk_type = risk['risk_type']

            if risk_type == "结泥饼风险":
                risk_type_mapped = "结泥饼"
                # 结泥饼逻辑略有特殊，仍保留部分特殊逻辑
                try:
                    center_ring_int = int(float(ring_number))
                    present_rings = set()
                    for p in multi_ring_data:
                        r = p.get('Ring.No', p.get('RING'))
                        try:
                            present_rings.add(int(float(r)))
                        except Exception:
                            pass
                    if center_ring_int not in present_rings or len(present_rings) < 6:
                        result["mud_cake_risk"] = _build_mud_cake_fallback_risk(ring_number, preload_map, multi_ring_data)
                        continue
                except Exception:
                    result["mud_cake_risk"] = _build_mud_cake_fallback_risk(ring_number, preload_map, multi_ring_data)
                    continue

                fm = _clean_json_val(risk["measures"])
                config = RISK_CONFIG["mud_cake_risk"]
                imp = "，".join([config["map"].get(k, k) for k in config["fields"]])
                risk_out = {
                    "risk_type": risk_type_mapped,
                    "ring": int(result["ring"]),
                    "project_name": PROJECT_NAME,
                    "fault_measures": fm,
                    "fault_reason": risk["reason"],
                    "fault_reason_analysis": get_fault_reason_analysis(risk_type_mapped, risk["risk_level"]),
                    "fault_cause": "渣土在刀盘面板或土仓内板结硬化，形成阻碍掘进的泥饼，导致掘进效率降低",
                    "impact_parameters": imp,
                    "safety_level": risk["risk_level"],
                    "risk_level": map_safety_to_level(risk["risk_level"]),
                    "risk_score": round(risk["risk_score"], 2),
                    "probability": round(risk["probability"], 2),
                    "potential_risk": risk["potential_risk"],
                    "warning_time": "-",
                    "warning_parameters": "-",
                }
                
                # 结泥饼的 warning_time 逻辑
                earliest_time_raw = risk.get("earliest_time")
                final_level_text = risk.get("risk_level", "")
                should_warn = final_level_text in ("低风险Ⅱ", "中风险Ⅲ", "高风险Ⅳ")
                
                if should_warn and earliest_time_raw and earliest_time_raw != "-":
                    risk_out["warning_time"] = format_time_utc_to_shanghai(earliest_time_raw)
                else:
                    # 备选：当前环的第一条数据时间
                    first_p = next((p for p in multi_ring_data if int(float(p.get('Ring.No', p.get('RING', -1)))) == int(result["ring"])), None)
                    if first_p:
                        risk_out["warning_time"] = format_time_utc_to_shanghai(first_p.get("time"))

                # 结泥饼的 warning_parameters (取当前环第一条数据快照)
                first_p = next((p for p in multi_ring_data if int(float(p.get('Ring.No', p.get('RING', -1)))) == int(result["ring"])), None)
                if first_p:
                    ep = {f: first_p.get(f) for f in config["fields"]}
                    wp_raw = {config["map"].get(k, k): ep.get(k) for k in ep}
                    risk_out["warning_parameters"] = _append_units_to_map(wp_raw, config["units"])
                
                risk_out["key_parameters"] = _rename_keys(
                    collect_key_parameters(ring_number, config["fields"], count=6, preload=preload_map), config["map"])
                risk_out["key_parameters"] = _with_value_and_unit(risk_out["key_parameters"], config["units"])
                result["mud_cake_risk"] = risk_out  

            elif risk_type == "滞排风险":
                result["clog_risk"] = _build_risk_item(risk, "滞排", "clog_risk", risk_assessor.assess_clog_risk, "渣土改良不充分导致流动性不佳，最终在管道内发生滞排")

            elif risk_type == "主驱动密封失效风险":
                result["mdr_seal_risk"] = _build_risk_item(risk, "主驱动密封失效", "mdr_seal_risk", risk_assessor.assess_mdr_seal_risk, "主轴承密封系统因磨损老化或密封油脂压力不足，导致外部渣土或地下水浸入")

            elif risk_type == "盾尾密封失效风险":
                result["tail_seal_risk"] = _build_risk_item(risk, "盾尾密封失效", "tail_seal_risk", risk_assessor.assess_tail_seal_risk, "盾尾密封刷密封件磨损、撕裂或者压损后，失去阻挡同步注浆浆液和地下水的能力")

        if not result.get("mud_cake_risk"):
            result["mud_cake_risk"] = _build_mud_cake_fallback_risk(ring_number, preload_map, multi_ring_data)

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
        read_timeout = int(os.environ.get("MYSQL_READ_TIMEOUT", "8"))
        write_timeout = int(os.environ.get("MYSQL_WRITE_TIMEOUT", "8"))
        return pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset=self._charset,
            cursorclass=self._cursorclass,
            autocommit=True,
            connect_timeout=10,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

    def acquire(self):
        try:
            conn = self._pool.get_nowait()
            try:
                conn.ping(reconnect=True)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = self._create_conn()
            return conn
        except Exception:
            with self._lock:
                if self._created < self._maxsize:
                    conn = self._create_conn()
                    self._created += 1
                    return conn
            wait_timeout = float(os.environ.get("MYSQL_POOL_WAIT", "5"))
            try:
                return self._pool.get(timeout=wait_timeout)
            except Exception:
                raise RuntimeError("MySQL连接池等待超时")

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
            if exc_type is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            else:
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
        _set_session_timeout(raw)  # 每次获取连接时设置会话超时
        return _PooledConnection(_MYSQL_POOL, raw)
    except Exception as e:
        raise RuntimeError(f"MySQL连接失败: {e}")

def _set_session_timeout(conn):
    try:
        max_ms = int(os.environ.get("MYSQL_MAX_EXECUTION_MS", "3000"))
        with conn.cursor() as cur:
            cur.execute("SET SESSION MAX_EXECUTION_TIME=%s", (max_ms,))
    except Exception:
        pass


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
                "无风险Ⅰ": "刀盘扭矩与刀盘转速匹配良好，推力与推进速度稳定，贯入度无异常波动。渣土含水率与改良剂比例处于工艺目标，仓壁与面板未见黏附或板结痕迹，排泥连续顺畅，不具备泥饼形成的物理条件。",
                "低风险Ⅱ": "观察到刀盘扭矩轻微抬升、推进速度小幅下降，推力与贯入度边界化波动，反映土体塑性增大与局部摩阻上升。面板可能出现初始黏附点，渣粒分布偏粗或含水率偏离最佳区间，短时排泥不均。若该状态持续，将促使黏附向板结演化，需要关注改良剂与水分的微调空间。",
                "中风险Ⅲ": "刀盘扭矩与推力持续偏高，推进速度显著下降且波动加大，贯入度呈非线性起伏，指示面板与土仓内已有黏附/板结积聚。排泥阻力增大、管路压降上升、回流概率增加，渣土流变性恶化。成因多与高粘性土、改良不足或含水率偏低/偏高叠加导致黏结力提升有关，若不干预泥饼将沿面板扩展并破坏掘进稳定性与效率。",
                "高风险Ⅳ": "面板与仓内泥饼大面积形成，刀盘扭矩异常抬升且伴随振动，推进速度显著受限或间断，推力异常维持，排泥系统出现明显阻塞。进一步将导致驱动负荷过高、温升增大、密封与轴承受污染风险上升，存在设备损伤与渗水次生风险，应在可控前提下尽快处置以恢复切削与排泥通道。",
            }
            return mapping.get(lvl, "当前等级暂无更详细分析")
        if "滞排" in t:
            mapping = {
                "无风险Ⅰ": "开挖舱压力与工作舱压力稳定，排浆泵进泥口压力处于正常区间，压力梯度合理、流态均匀。渣土含水率与改良剂配比匹配，管路内无沉积或团聚迹象，排量连续可控。",
                "低风险Ⅱ": "管路摩阻略增，泵出口与进泥口压力出现轻微非同步波动，短时排量下降提示流动性边界化。渣土可能存在剪切变稀临界、团聚核初始形成或颗粒级配偏离，需关注参数微调以避免在弯头、缩径或高阻段形成稳定堵塞核。",
                "中风险Ⅲ": "进泥口压力显著抬升且泵压波动加剧，表征管内已有部分堵塞或团聚带。压力梯度异常、回流概率上升、脉动增大，局部存在沉积与再悬浮循环。根因多与改良不足、含水率不当、细粒黏性偏高或输送能级不匹配相关，需及时分段冲洗并联动调整改良与泵速以恢复连续性。",
                "高风险Ⅳ": "系统出现严重滞排或近乎完全堵塞，泵压异常、排量骤降甚至中断，压力梯度失衡并伴随强烈脉动。继续掘进将导致设备过载、密封与管路风险叠加，存在安全与质量隐患，应尽快在安全窗口下清障并重建稳定输送条件。",
            }
            return mapping.get(lvl, "当前等级暂无更详细分析")
        if "主驱动密封失效" in t:
            mapping = {
                "无风险Ⅰ": "工作舱压力、油气密封反馈压力、主驱动伸缩密封油脂压力、齿轮油油气密封压力或密封检测压力整体处于稳态，密封压力梯度合理。无渗漏迹象，油脂供给与流变性能正常，密封件弹性与磨耗处健康区间。",
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
    _set_session_timeout(conn)
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT * FROM `{table}` WHERE `id`=(SELECT MAX(`id`) FROM `{table}`)")
            row = cur.fetchone()
            if row:
                return row
        except Exception:
            pass
        try:
            cur.execute(f"SELECT * FROM `{table}` WHERE `ring`=(SELECT MAX(`ring`) FROM `{table}`)")
            row = cur.fetchone()
            if row:
                return row
        except Exception:
            pass
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
            table = RISK_TABLES.get(rp)
            if not table:
                return jsonify({"status": "error", "error": "不支持的风险类型", "supported": list(RISK_TABLES.keys())}), 400
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
                for k, t in RISK_TABLES.items():
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

        # 使用全局 _clean_json_val 代替内嵌逻辑

        def build_risk_out(row, meta):
            safety = row.get("safety_level")
            if safety is None or str(safety).strip() == "":
                safety = row.get("risk_level")
            ring_val = row.get("ring")
            try:
                ring_num = int(float(ring_val)) if ring_val is not None else "-"
            except Exception:
                ring_num = _dash(ring_val)

            fm = _get_any(row, ["fault_measures", "measures"])
            fm = _clean_json_val(fm)

            ip = row.get("impact_parameters")
            ip = _clean_json_val(ip)

            kp = row.get("key_parameters")
            kp = _clean_json_val(kp, dash_if_empty=True)

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
                "fault_cause": "渣土改良不充分导致流动性不佳，最终在管道内发生滞排",
            },
            "主驱动密封失效": {
                "result_key": "mdr_seal_risk",
                "risk_type": "主驱动密封失效",
                "fault_cause": "主轴承密封系统因磨损老化或密封油脂压力不足，导致外部渣土或地下水浸入",
            },
            "结泥饼": {
                "result_key": "mud_cake_risk",
                "risk_type": "结泥饼",
                "fault_cause": "渣土在刀盘面板或土仓内板结硬化，形成阻碍掘进的泥饼，导致掘进效率降低",
            },
            "盾尾密封失效": {
                "result_key": "tail_seal_risk",
                "risk_type": "盾尾密封失效",
                "fault_cause": "盾尾密封刷密封件磨损、撕裂或者压损后，失去阻挡同步注浆浆液和地下水的能力",
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
                return jsonify(risk_out)
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


@bp.route('/getLatestRiskLevelSimple', methods=['GET'])
def get_latest_risk_level_simple():
    try:
        conn = _mysql_connect()

        result = {}
        
        with conn:
            for k, t in RISK_TABLES.items():
                # k 是中文风险类型 (例如 "滞排")
                # t 是对应的表名
                try:
                    row = _fetch_latest_row(conn, t)
                    if row:
                        level_raw = row.get("risk_level")
                        if level_raw is None or str(level_raw).strip() == "":
                            level_out = "-"
                        else:
                            level_out = level_raw

                        ring_val = row.get("ring")
                        try:
                            ring_num = int(float(ring_val)) if ring_val is not None else "-"
                        except Exception:
                            ring_num = "-" if ring_val is None else str(ring_val)

                        result[k] = {
                            "risk_level": level_out,
                            "ring": ring_num
                        }
                except Exception:
                    # 单个风险获取失败不应阻塞其他风险
                    continue
                    
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"获取最新风险等级简报时发生错误: {str(e)}"}), 500


def query_consecutive_ring_data(center_ring, count=6):
    """以当前环为终点，向前回溯收集最多 count 个有数据的环，按时间升序合并返回。"""
    try:
        ring_int = int(float(center_ring))
        combined = []
        rings_collected = set()
        r = ring_int
        target_count = int(count)
        while r > 0 and len(rings_collected) < target_count:
            pts = query_ring_data(r)
            if pts:
                combined.extend(pts)
                rings_collected.add(r)
            r -= 1
        if not combined:
            return []
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
        return []


def _rename_keys(items, mapping):
    try:
        renamed = []
        for it in items or []:
            if not isinstance(it, dict):
                renamed.append(it)
                continue
            new_it = {}
            for k, v in it.items():
                mk = mapping.get(k, k)
                new_it[mk] = v
            renamed.append(new_it)
        return renamed
    except Exception:
        return items

def _with_value_and_unit(items, unit_map):
    try:
        out = []
        for it in items or []:
            if not isinstance(it, dict):
                out.append(it)
                continue
            ni = {}
            for k, v in it.items():
                if k == "time":
                    ni["time"] = v
                else:
                    u = unit_map.get(k, "-")
                    ni[k] = {"value": v, "unit": u}
            out.append(ni)
        return out
    except Exception:
        return items

def _append_units_to_map(data_map, units_map):
    try:
        out = {}
        for k, v in (data_map or {}).items():
            u = units_map.get(k, "")
            if v is None or (isinstance(v, str) and v.strip() in ("", "null", "none", "-")):
                out[k] = "-"
            else:
                out[k] = f"{v}{u}" if u else str(v)
        return out
    except Exception:
        return data_map


@bp.route('/getAllRiskRecords', methods=['GET'])
def get_all_risk_records():
    try:
        conn = _mysql_connect()
        risk_type_param = request.args.get("risk_type")
        if not risk_type_param:
            body = request.get_json(silent=True) or {}
            risk_type_param = body.get("risk_type")
        limit_env = int(os.environ.get("RISK_RECORDS_LIMIT", "1000"))
        max_limit_env = int(os.environ.get("RISK_RECORDS_LIMIT_MAX", "5000"))
        try:
            limit_arg = int(request.args.get("limit", str(limit_env)))
        except Exception:
            limit_arg = limit_env
        if limit_arg <= 0:
            limit_arg = limit_env
        if limit_arg > max_limit_env:
            limit_arg = max_limit_env
        if risk_type_param:
            rp = str(risk_type_param).strip()
            if rp.endswith("风险"):
                rp = rp[:-2]
            table = RISK_TABLES.get(rp)
            if not table:
                return jsonify({"status": "error", "error": "不支持的风险类型", "supported": list(RISK_TABLES.keys())}), 400
            loop_items = [(rp, table)]
        else:
            loop_items = list(RISK_TABLES.items())
        records = []
        high_count = 0
        mid_count = 0
        low_count = 0
        with conn:
            with conn.cursor() as cur:
                for label, table in loop_items:
                    try:
                        cur.execute(
                            f"SELECT `ring`,`warning_time`,`warning_parameters`,`fault_reason`,`fault_measures`,`risk_level` FROM `{table}` ORDER BY `id` DESC LIMIT %s",
                            (limit_arg,)
                        )
                        rows = cur.fetchall() or []
                    except Exception:
                        cur.execute(
                            f"SELECT `ring`,`warning_time`,`warning_parameters`,`fault_reason`,`fault_measures`,`risk_level` FROM `{table}` ORDER BY `ring` DESC LIMIT %s",
                            (limit_arg,)
                        )
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
                        
                        wp = _clean_json_val(row.get("warning_parameters"))
                        fm = _clean_json_val(row.get("fault_measures"))
                        
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


def _get_point_info_connection():
    return pymysql.connect(
        host="172.16.105.12",
        port=13366,
        user="root",
        password="123456",
        database="point_info",
        charset="utf8",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
    )


@bp.route('/history/parameters', methods=['GET'])
def get_available_parameters():
    try:
        shieldid = request.args.get("shieldid")
        if not shieldid:
            data = request.get_json(silent=True) or {}
            shieldid = data.get("shieldid")
        if not shieldid:
            shieldid = "tsjy_dz1360"
        prefix = shieldid
            
        table_name = f"point_info_{prefix}"
        
        try:
            conn = _get_point_info_connection()

            with conn:
                with conn.cursor() as cur:
                    # Query parameters where point_code, point_cn_name, and point_unit are all valid
                    query = f"SELECT point_cn_name, point_code, point_unit FROM `{table_name}` WHERE point_unit IS NOT NULL AND point_unit != '' AND point_code IS NOT NULL AND point_code != '' AND point_cn_name IS NOT NULL AND point_cn_name != ''"
                    cur.execute(query)
                    rows = cur.fetchall()
                    
                    result = []
                    for row in rows:
                        result.append({
                            "name": row['point_cn_name'],
                            "code": row['point_code'],
                            "unit": row['point_unit']
                        })
                    return jsonify(result)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": f"MySQL Query Error: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@bp.route('/history/query', methods=['POST'])
def query_history_data():
    try:
        # 配置常量
        TARGET_DB_NAME = "tsjy_dz1360_cleanDate"
        PROJECT_PREFIX = "tsjy_dz1360"
        MYSQL_TABLE_NAME = f"point_info_{PROJECT_PREFIX}"
        
        # Get parameters from JSON body
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "error": "Missing JSON body"}), 400
            
        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")
        
        parameters = []
        params_raw = data.get("parameters")
        if isinstance(params_raw, list):
            parameters = params_raw
        elif isinstance(params_raw, str) and params_raw:
            params_raw = params_raw.strip()
            # Handle list format in string
            if params_raw.startswith("[") and params_raw.endswith("]"):
                try:
                    parameters = json.loads(params_raw)
                except Exception:
                    parameters = [p.strip() for p in params_raw.strip("[]").split(",") if p.strip()]
            else:
                parameters = [p.strip() for p in params_raw.split(",") if p.strip()]

        if not start_date_str or not end_date_str:
            return jsonify({"status": "error", "error": "Missing 'start_date' or 'end_date' parameter"}), 400
        
        if not parameters:
            return jsonify({"status": "error", "error": "Missing 'parameters' parameter"}), 400
            
        if not (1 <= len(parameters) <= 4):
            return jsonify({"status": "error", "error": "The number of parameters must be between 1 and 4"}), 400

        try:
            start_dt_local = pd.to_datetime(start_date_str).tz_localize('Asia/Shanghai')
            end_dt_local = pd.to_datetime(end_date_str).tz_localize('Asia/Shanghai') + timedelta(days=1)
            
            if start_dt_local >= end_dt_local:
                return jsonify({"status": "error", "error": "start_date cannot be later than end_date"}), 400
            
            start_dt_utc = start_dt_local.tz_convert('UTC')
            end_dt_utc = end_dt_local.tz_convert('UTC')
                
        except ValueError:
            return jsonify({"status": "error", "error": "Invalid date format. Use YYYY-MM-DD"}), 400

        point_mapping = {}
        
        cache_key = (MYSQL_TABLE_NAME, tuple(parameters))
        try:
            if cache_key in POINT_MAP_CACHE:
                point_mapping = POINT_MAP_CACHE.get(cache_key, {})
            else:
                conn = _get_point_info_connection()
                with conn:
                    with conn.cursor() as cur:
                        format_strings = ','.join(['%s'] * len(parameters))
                        query = f"SELECT point_code, point_cn_name, point_unit FROM `{MYSQL_TABLE_NAME}` WHERE point_cn_name IN ({format_strings})"
                        cur.execute(query, tuple(parameters))
                        rows = cur.fetchall()
                        for row in rows:
                            code_val = row['point_code']
                            if code_val:
                                code_val = code_val.strip()
                            point_mapping[row['point_cn_name']] = {
                                'code': code_val,
                                'unit': row['point_unit']
                            }
                POINT_MAP_CACHE[cache_key] = point_mapping
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": f"MySQL Query Error: {str(e)}"}), 500

        if not point_mapping:
            return jsonify({"status": "error", "error": "No matching parameters found in database"}), 404

        fields = []
        for p in parameters:
            if p in point_mapping:
                fields.append(f'"{point_mapping[p]["code"]}"')
        
        if not fields:
            return jsonify({"status": "error", "error": "No valid fields to query"}), 400

        select_clause = ", ".join(fields)
        
        time_filter = f"time >= '{start_dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}' AND time < '{end_dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        
        all_points = []
        curr_date = start_dt_local.date() - timedelta(days=1)
        target_end_date = end_dt_local.date() + timedelta(days=1)
        
        measurement_names = []
        while curr_date < target_end_date:
            tn = f"{PROJECT_PREFIX}_DZ1360_{curr_date.strftime('%Y_%m_%d')}"
            measurement_names.append(tn)
            curr_date += timedelta(days=1)
            
        def _q(measurement_name):
            q = f'SELECT {select_clause} FROM "{measurement_name}" WHERE {time_filter}'
            try:
                # 使用新的数据库名称 TARGET_DB_NAME
                res = client.query(q, database=TARGET_DB_NAME, epoch='ms')
                pts = list(res.get_points())
                return pts
            except Exception as e:
                print(f"Error querying {measurement_name}: {e}")
                return []
        with ThreadPoolExecutor(max_workers=min(16, len(measurement_names) or 1)) as ex:
            futs = [ex.submit(_q, m) for m in measurement_names]
            for f in as_completed(futs):
                pts = f.result() or []
                if pts:
                    all_points.extend(pts)
        all_points.sort(key=lambda x: x.get('time', ''))

        mapping_info_list = []
        for cn_name in parameters:
            info = point_mapping.get(cn_name)
            if info:
                mapping_info_list.append((cn_name, info['code'], info['unit']))
        
        output_data = []
        tz_offset = timedelta(hours=8)
        
        for pt in all_points:
            ts = pt.get('time')
            # 快速时间格式化 (ts is milliseconds timestamp)
            dt = datetime.utcfromtimestamp(ts / 1000) + tz_offset
            time_str = dt.strftime("%Y/%m/%d %H:%M:%S")

            item = {"time": time_str}
            for cn_name, code, unit in mapping_info_list:
                val = pt.get(code)
                item[cn_name] = {
                    "value": val if val is not None else 0.0,
                    "unit": unit if unit else ""
                }
            output_data.append(item)

        return jsonify(output_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@bp.route('/filter/options', methods=['GET'])
def get_filter_options():
    try:
        conn = _mysql_connect()
        risk_type_param = request.args.get('risk_type')
        if not risk_type_param:
            body = request.get_json(silent=True) or {}
            risk_type_param = body.get("risk_type")
            
        if not risk_type_param:
             risk_type_param = '结泥饼风险'

        rp = str(risk_type_param).strip()
        if rp.endswith("风险"):
            rp = rp[:-2]
            
        table = RISK_TABLES.get(rp)
        if not table:
             return jsonify({"status": "error", "error": "不支持的风险类型"}), 400

        safety_level = "无风险Ⅰ"
        fault_measures = "-"
        potential_risk = "-"
        
        try:
            with conn:
                row = _fetch_latest_row(conn, table)
                if row:
                    # 优先读取 safety_level，如果为空则读取 risk_level
                    sl = row.get('safety_level')
                    if sl and str(sl).strip():
                        safety_level = sl
                    else:
                        safety_level = row.get('risk_level') or "无风险Ⅰ"
                        
                    # Match getLatestRiskLevel: check fault_measures then measures
                    fm = row.get('fault_measures')
                    if fm in (None, "", "-"):
                        fm = row.get('measures')
                    
                    fault_measures = _decode_json_string(fm)
        except Exception as e:
            traceback.print_exc()
            pass
            
        potential_risk = (rp + "预警") if safety_level != "无风险Ⅰ" else "-"
        
        return jsonify({
            "project_name": PROJECT_NAME,
            "safety_level": safety_level,
            "potential_risk": potential_risk,
            "fault_measures": fault_measures
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@bp.route('/info/reverse', methods=['GET'])
def get_reverse_risk_info():
    try:
        conn = _mysql_connect()
        risk_type_param = request.args.get('risk_type')
        if not risk_type_param:
            body = request.get_json(silent=True) or {}
            risk_type_param = body.get("risk_type")
            
        if not risk_type_param:
             risk_type_param = '结泥饼风险'
             
        rp = str(risk_type_param).strip()
        if rp.endswith("风险"):
            rp = rp[:-2]
            
        table = RISK_TABLES.get(rp)
        if not table:
             return jsonify({"status": "error", "error": "不支持的风险类型"}), 400
             
        risk_score = 0.0
        try:
            with conn:
                row = _fetch_latest_row(conn, table)
                if row:
                     # 优先读取 risk_score
                     rs = row.get('risk_score')
                     if rs is not None:
                         try:
                             risk_score = float(rs)
                         except:
                             pass
        except Exception:
            traceback.print_exc()
            pass

        # Append "风险" for risk_assessor calls
        full_risk_name = rp + "风险"
            
        probability = risk_assessor.reverse_map_score_to_probability(risk_score, full_risk_name)
        
        # Determine thresholds based on risk type using the centralized method
        potential_risk_threshold, normal_probability_threshold = risk_assessor.get_probability_thresholds(full_risk_name)
            
        return jsonify({
            "当前风险概率": round(probability, 3),
            "潜在风险阈值": potential_risk_threshold,
            "正常概率阈值": normal_probability_threshold
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


CORRELATION_DATA_CACHE = None
CORRELATION_DATA_LOCK = threading.Lock()
CORRELATION_MAPS_CACHE = None

def _get_correlation_data():
    global CORRELATION_DATA_CACHE
    with CORRELATION_DATA_LOCK:
        if CORRELATION_DATA_CACHE is None:
            json_path = os.path.join(os.path.dirname(__file__), 'data', '监测参数关联.json')
            if not os.path.exists(json_path):
                return None
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    CORRELATION_DATA_CACHE = json.load(f)
            except Exception as e:
                traceback.print_exc()
                return None
        return CORRELATION_DATA_CACHE

def _get_correlation_maps():
    global CORRELATION_MAPS_CACHE
    
    # Check if maps are already built
    if CORRELATION_MAPS_CACHE is not None:
        return CORRELATION_MAPS_CACHE
        
    data = _get_correlation_data()
    if not data:
        return None, None
        
    with CORRELATION_DATA_LOCK:
        # Double check inside lock
        if CORRELATION_MAPS_CACHE is not None:
            return CORRELATION_MAPS_CACHE
            
        # Build parameter to system mapping
        param_system_map = {}
        if 'nodes' in data:
            for sys_name, nodes in data['nodes'].items():
                for node in nodes:
                    props = node.get('properties', {})
                    p_name = props.get('name')
                    if p_name:
                        param_system_map[p_name] = sys_name
                        
        # Build adjacency list for relationships
        # map: param_name -> list of {name, correlation, direction, level}
        adjacency_map = {}
        if 'relationships' in data:
            for rel in data['relationships']:
                props = rel.get('properties', {})
                start_node = props.get('start_node')
                end_node = props.get('end_node')
                correlation = props.get('correlation')
                
                if not (start_node and end_node and correlation is not None):
                    continue
                    
                item = {
                    "correlation": float(correlation),
                    "abs_correlation": abs(float(correlation)),
                    "direction": props.get("direction"),
                    "level": props.get("level")
                }
                
                # Add for start_node -> end_node
                item_start = item.copy()
                item_start["name"] = end_node
                if start_node not in adjacency_map:
                    adjacency_map[start_node] = []
                adjacency_map[start_node].append(item_start)
                
                # Add for end_node -> start_node
                item_end = item.copy()
                item_end["name"] = start_node
                if end_node not in adjacency_map:
                    adjacency_map[end_node] = []
                adjacency_map[end_node].append(item_end)
        
        CORRELATION_MAPS_CACHE = (param_system_map, adjacency_map)
        return CORRELATION_MAPS_CACHE

def _get_param_value(keys):
    try:
        for k in keys:
            v = request.args.get(k)
            if v:
                return v
        for k in keys:
            try:
                v = request.values.get(k)
            except Exception:
                v = None
            if v:
                return v
        body = request.get_json(silent=True)
        if isinstance(body, dict):
            for k in keys:
                v = body.get(k)
                if v:
                    return v
        if not body and request.data:
            try:
                raw = json.loads(request.data.decode('utf-8'))
                if isinstance(raw, dict):
                    for k in keys:
                        v = raw.get(k)
                        if v:
                            return v
            except Exception:
                pass
    except Exception:
        pass
    return None

@bp.route('/correlation/systems', methods=['GET'])
def get_correlation_systems():
    try:
        data = _get_correlation_data()
        if not data or 'nodes' not in data:
            return jsonify({"status": "error", "error": "关联数据不可用"}), 500
        
        systems = list(data['nodes'].keys())
        return jsonify({"systems": systems})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

@bp.route('/correlation/system_params', methods=['POST'])
def get_system_params():
    try:
        system_name = _get_param_value(['system', 'system_name', 'sys'])

        if not system_name:
            return jsonify({"status": "error", "error": "Missing system"}), 400
        system_name = str(system_name).strip()
            
        data = _get_correlation_data()
        if not data or 'nodes' not in data:
            return jsonify({"status": "error", "error": "关联数据不可用"}), 500
            
        nodes = data['nodes'].get(system_name)
        if not nodes:
            return jsonify({"system": system_name, "parameters": []})
            
        params = []
        for node in nodes:
            props = node.get('properties', {})
            name = props.get('name')
            if name:
                params.append(name)
                
        return jsonify({"parameters": params})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

@bp.route('/correlation/top5', methods=['POST'])
def get_top5_correlated():
    try:
        param_name = _get_param_value(['parameter', 'parameter_name'])
            
        if not param_name:
            return jsonify({"status": "error", "error": "Missing parameter"}), 400
            
        param_name = str(param_name).strip()

        param_system_map, adjacency_map = _get_correlation_maps()
        
        if param_system_map is None or adjacency_map is None:
            return jsonify({"status": "error", "error": "关联数据不可用"}), 500
            
        correlated = adjacency_map.get(param_name, [])
                
        correlated.sort(key=lambda x: x['abs_correlation'], reverse=True)
        top = correlated[:5]
        result = {"parameter": param_name}
        for idx, item in enumerate(top, start=1):
            # item is shared in cache, create copy before modifying if needed, 
            # but we just read here. Wait, we popped 'abs_correlation' before.
            # We should NOT pop from cached objects.
            
            p_name = item.get("name")
            result[str(idx)] = {
                "name": p_name,
                "correlation": item.get("correlation"),
                "direction": item.get("direction"),
                "level": item.get("level"),
                "system": param_system_map.get(p_name, "未知系统")
            }
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

@bp.route('/get_latest_state', methods=['GET'])
def get_latest_state():
    try:
        target_db = 'tsjy_dz1360_map'
        # 1. 获取该数据库下的所有 measurements
        result = client.query('SHOW MEASUREMENTS', database=target_db)
        measurements = list(result.get_points())
        
        # 2. 找到最新的 measurement (格式: tsjy_dz1360_DZ1360_YYYY_MM_DD)
        # 过滤出符合预期的 measurement 名称
        m_names = [m['name'] for m in measurements if m['name'].startswith('tsjy_dz1360_DZ1360_')]
        
        if not m_names:
            return jsonify({"state": "-", "time": "-"})
            
        # 按字符串排序取最后一个，即日期最新的那个
        latest_measurement = sorted(m_names)[-1]
        
        # 3. 查询最新数据
        query = f'SELECT state FROM "{latest_measurement}" ORDER BY time DESC LIMIT 1'
        result = client.query(query, database=target_db)
        points = list(result.get_points())
        
        if points:
            state_val = points[0].get("state", "-")
            time_val = points[0].get("time", "-")
            
            # 添加备注逻辑
            remark = "-"
            try:
                state_int = int(state_val)
                if state_int == 0:
                    remark = "停机"
                elif state_int == 1:
                    remark = "掘进"
                elif state_int == 2:
                    remark = "拼环"
            except (ValueError, TypeError):
                pass
            
            # 时间格式转换
            if time_val != "-":
                time_val = format_time_utc_to_shanghai(time_val)

            return jsonify({
                "state": state_val,
                "time": time_val,
                "remark": remark
            })
        return jsonify({"state": "-", "time": "-", "remark": "-"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

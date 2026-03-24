import numpy as np
from typing import Dict, Any, List, Optional
UNIFIED_REQUIRED_PARAMS = {
    'WorkCham.Pres.01': '工作舱压力01',
    'ExcavCham.Pres.04': '开挖舱压力04',
    'SlurryPump.P2.1.MudIn.Pres': '排浆泵P2.1进泥口压力'
}
REQUIRED_EXCAVATION_PARAMS = UNIFIED_REQUIRED_PARAMS
CORE_PARAMS_CHINESE = {
    'bubble_pressure': '工作舱压力01',
    'excavation_pressure': '开挖舱压力04',
    'pump_pressure': '排浆泵P2.1进泥口压力'
}
def map_project_data(data: Dict[str, Any]) -> Dict[str, float]:
    mapped_data = {
        'bubble_pressure': data.get('WorkCham.Pres.01', 0),
        'excavation_pressure': data.get('ExcavCham.Pres.04', 0),
        'pump_pressure': data.get('SlurryPump.P2.1.MudIn.Pres', 0)
    }
    for key, value in mapped_data.items():
        if not isinstance(value, (int, float)) or np.isnan(value):
            mapped_data[key] = 0.0
        else:
            mapped_data[key] = float(value)
    return mapped_data
def is_sensor_detected(data: Dict[str, Any], sensor_key: str) -> bool:
    if sensor_key not in data:
        return False
    value = data[sensor_key]
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return not np.isnan(value)
    return False
def validate_sensor_data(data: Dict[str, Any], required_sensors: List[str]) -> Dict[str, bool]:
    validation_results = {}
    for sensor in required_sensors:
        validation_results[sensor] = is_sensor_detected(data, sensor)
    return validation_results
 
def calculate_universal_clog_risk(data: Dict[str, Any], shield_id: str = None) -> Dict[str, Any]:
    try:
        # 严格校验必要字段
        validations = validate_sensor_data(data, list(UNIFIED_REQUIRED_PARAMS.keys()))
        if not all(validations.values()):
            missing = [k for k, ok in validations.items() if not ok]
            raise ValueError(f"滞排风险计算缺少必要字段: {', '.join(missing)}")

        mapped_data = map_project_data(data)
        P_bubble = mapped_data['bubble_pressure']      # 工作舱压力01
        P_excav = mapped_data['excavation_pressure']   # 开挖舱压力04
        P_pump = mapped_data['pump_pressure']          # 排浆泵进泥口压力

        diff_bubble_exc = P_bubble - P_excav
        diff_bubble_pump = P_bubble - P_pump

        if diff_bubble_exc < -1 and diff_bubble_pump > 2:
            risk_level, msg, prob = '高风险', '泥水环流系统滞排现象严重', 0.9
        elif diff_bubble_exc < -1:
            risk_level, msg, prob = '中风险', '开挖舱内明显滞排', 0.7
        elif diff_bubble_pump > 2:
            risk_level, msg, prob = '中风险', '泥水脱离格栅处明显滞排', 0.7
        elif -1 <= diff_bubble_exc <= -0.5:
            risk_level, msg, prob = '低风险', '开挖舱内轻微滞排', 0.3
        elif 1 <= diff_bubble_pump <= 2:
            risk_level, msg, prob = '低风险', '泥水脱离格栅处轻微滞排', 0.3
        else:
            risk_level, msg, prob = '无风险', '泥水环流系统无滞排风险', 0.0

        return {
            'probability': round(prob, 3),
            'risk_level': risk_level,
            'details': f"{msg} (ΔP_仓={diff_bubble_exc:.2f}, ΔP_泵={diff_bubble_pump:.2f})"
        }
    except Exception as e:
        raise RuntimeError(f"滞排风险评估失败: {str(e)}")





def risk_metadata() -> Dict[str, Any]:
    return {
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
        },
    }


def get_measures_and_reason(risk_level: str):
    mapping = {
        "无风险Ⅰ": {
            "measures": ["维持状态：保持当前掘进参数", "正常作业：按照标准流程进行"],
            "reason": "开挖舱压力稳定，工作舱压力正常，排浆泵压力平稳，排泥系统运行平稳，渣土含水率适宜，排泥管道通畅，刀盘转速和刀盘扭矩保持稳定，渣土排出连续顺畅。"
        },
        "低风险Ⅱ": {
            "measures": ["监控压力趋势：每15分钟记录一次压力", "调整掘进参数：适当降低推进速度"],
            "reason": "开挖舱压力略有波动，工作舱压力轻微升高，排浆泵压力小幅变化，排泥速度略有下降，刀盘扭矩略有增加，渣土含水率略有变化，但排泥系统整体运行仍在安全范围内。"
        },
        "中风险Ⅲ": {
            "measures": ["加强密封监控：将密封压力监测频率提高到每30分钟一次", "检查压力梯度：详细分析各道密封之间的压力梯度"],
            "reason": "开挖舱压力不稳定，工作舱压力明显升高，排浆泵压力波动较大，排泥速度明显下降，刀盘扭矩波动明显，渣土含水率异常。"
        },
        "高风险Ⅳ": {
            "measures": ["立即停机：停止掘进作业", "检查压力系统：全面检查气垫舱压力"],
            "reason": "开挖舱压力异常，工作舱压力过高，排浆泵压力急剧变化，刀盘扭矩急剧上升，排泥系统出现明显阻塞，渣土几乎无法排出。"
        }
    }
    d = mapping.get(risk_level)
    return (d.get("measures", []), d.get("reason", "")) if d else ([], "")


def probability_to_score(probability: float) -> float:
    p = float(probability or 0.0)
    if p >= 0.9:
        return 0.5 - (p - 0.9) * 0.5 / 0.1
    elif p >= 0.7:
        return 2 - (p - 0.7) * 1.5 / (0.9 - 0.7)
    elif p >= 0.3:
        return 3.5 - (p - 0.3) * 1.5 / (0.7 - 0.3)
    else:
        return 4.0 - p * 0.5 / 0.3


def reverse_score_to_probability(score: float) -> float:
    s = float(score or 0.0)
    if s <= 0.5:
        return 0.9 + (0.5 - s) * 0.2
    elif s <= 2:
        return 0.7 + (2 - s) * (0.2 / 1.5)
    elif s <= 3.5:
        return 0.3 + (3.5 - s) * (0.4 / 1.5)
    else:
        return (4.0 - s) * 0.6


def probability_thresholds():
    return 0.7, 0.3

import numpy as np
from typing import Dict, List, Any

UNIFIED_REQUIRED_PARAMS = {
    "Tail.Seal.Rear.Prs.02": "2#盾尾密封后压力",
    "Tail.Seal.Rear.Prs.04": "4#盾尾密封后压力",
    "Tail.Seal.Rear.Prs.06": "6#盾尾密封后压力",
    "Tail.Seal.Rear.Prs.08": "8#盾尾密封后压力",
    "Tail.Seal.Rear.Prs.10": "10#盾尾密封后压力",
    "Tail.Seal.Rear.Prs.12": "12#盾尾密封后压力",
    "Tail.Seal.Rear.Prs.14": "14#盾尾密封后压力",
    "Tail.Seal.Rear.Prs.16": "16#盾尾密封后压力",
    "LiqPump.A.Out.Prs.01": "1#A液泵出口压力",
    "LiqPump.A.Out.Prs.02": "2#A液泵出口压力",
    "LiqPump.A.Out.Prs.03": "3#A液泵出口压力",
    "LiqPump.A.Out.Prs.04": "4#A液泵出口压力",
    "LiqPump.A.Out.Prs.05": "5#A液泵出口压力",
    "LiqPump.A.Out.Prs.06": "6#A液泵出口压力",
    "LiqPump.A.Out.Prs.07": "7#A液泵出口压力",
    "LiqPump.A.Out.Prs.08": "8#A液泵出口压力",
}

UNIFIED_SEAL_SENSORS = [
    "Tail.Seal.Rear.Prs.02",
    "Tail.Seal.Rear.Prs.04",
    "Tail.Seal.Rear.Prs.06",
    "Tail.Seal.Rear.Prs.08",
    "Tail.Seal.Rear.Prs.10",
    "Tail.Seal.Rear.Prs.12",
    "Tail.Seal.Rear.Prs.14",
    "Tail.Seal.Rear.Prs.16",
]
UNIFIED_GROUT_SENSORS = [
    "LiqPump.A.Out.Prs.01",
    "LiqPump.A.Out.Prs.02",
    "LiqPump.A.Out.Prs.03",
    "LiqPump.A.Out.Prs.04",
    "LiqPump.A.Out.Prs.05",
    "LiqPump.A.Out.Prs.06",
    "LiqPump.A.Out.Prs.07",
    "LiqPump.A.Out.Prs.08",
]

RISK_SPEC = {
    "name": "盾尾密封失效风险",
    "risk_type_label": "盾尾密封失效",
    "full_risk_type": "盾尾密封失效风险",
    "output_key": "tail_seal_risk",
    "fault_cause": "盾尾密封刷密封件磨损、撕裂或者压损后，失去阻挡同步注浆浆液和地下水的能力",
    "potential_risk": "盾尾密封失效预警",
    "fields": list(UNIFIED_REQUIRED_PARAMS.keys()),
    "map": dict(UNIFIED_REQUIRED_PARAMS),
    "units": {
        "2#盾尾密封后压力": "bar",
        "4#盾尾密封后压力": "bar",
        "6#盾尾密封后压力": "bar",
        "8#盾尾密封后压力": "bar",
        "10#盾尾密封后压力": "bar",
        "12#盾尾密封后压力": "bar",
        "14#盾尾密封后压力": "bar",
        "16#盾尾密封后压力": "bar",
        "1#A液泵出口压力": "bar",
        "2#A液泵出口压力": "bar",
        "3#A液泵出口压力": "bar",
        "4#A液泵出口压力": "bar",
        "5#A液泵出口压力": "bar",
        "6#A液泵出口压力": "bar",
        "7#A液泵出口压力": "bar",
        "8#A液泵出口压力": "bar",
    },
    "score_points": [(0.0, 4.0), (0.3, 3.5), (0.5, 2.0), (0.9, 0.5), (1.0, 0.0)],
    "probability_thresholds": (0.5, 0.3),
    "fault_reason_analysis": {
        "无风险Ⅰ": "盾尾密封压力与注浆压力处于工况目标区间，密封刷弹性良好、刷列贴合充分，无渗水或浆液串漏现象，与管片拼装配合稳定。",
        "低风险Ⅱ": "密封压力与注脂量轻微上调以维持密封效果，提示局部刷列出现初期磨耗或贴合度下降，个别位置可能存在微渗。若维持该趋势将加速磨耗与膜层破坏，需关注注脂分配均衡与刷列状态。",
        "中风险Ⅲ": "密封压力波动加剧，注脂量显著增加以对冲密封衰减；腔压与负荷出现不稳定相关性。可见局部渗漏与刷列磨耗加重，密封水/浆屏障降低，若不干预将扩大至环向失效并影响管片成型与对口质量。",
        "高风险Ⅳ": "刷列严重磨损、撕裂或压损导致渗漏明确，密封压力难以维持，同步注浆与防水能力显著下降。继续掘进将带来水浆侵入、质量与安全复合风险，应在保护窗口内处置并恢复密封能力。",
    },
    "measures": {
        "无风险Ⅰ": {
            "measures": ["正常监控：监控密封状态", "定期保养：检查密封分配器和管路"],
            "reason": "密封压力与注浆压力处于工况目标区间，密封刷贴合充分，无渗水或浆液串漏现象。",
        },
        "低风险Ⅱ": {
            "measures": ["正常监控：监控密封状态", "定期保养：检查密封分配器和管路"],
            "reason": "密封压力与注脂量轻微上调以维持密封效果，提示局部刷列初期磨耗或贴合度下降，个别位置可能存在微渗。",
        },
        "中风险Ⅲ": {
            "measures": ["加强密封监控：每小时检查一次密封状态", "检查密封圈状态：评估密封圈磨损程度"],
            "reason": "密封压力波动加剧，注脂量明显增加以对冲密封衰减；局部渗漏与刷列磨耗加重，屏障能力降低。",
        },
        "高风险Ⅳ": {
            "measures": ["立即停机：停止掘进作业", "检查密封系统状态：详细检查工作舱压力"],
            "reason": "刷列严重磨损、撕裂或压损导致渗漏明确，密封压力难以维持，同步注浆与防水能力显著下降。",
        },
    },
}


def _to_sensor_value(value: Any) -> float:
    return float(value) if _is_valid_sensor_value(value) else np.nan


def _is_valid_sensor_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return not (np.isnan(value) or np.isinf(value))


def _is_valid_pressure(value: Any) -> bool:
    return _is_valid_sensor_value(value) and float(value) >= 0.0


def _get_sensor_raw_value(data: Dict[str, Any], sensor_key: str) -> Any:
    if sensor_key in data and data[sensor_key] is not None:
        return data[sensor_key]
    return None


def map_project_data(data: Dict[str, Any]) -> Dict[str, List[float]]:
    mapped_data = {
        'seal_pressures': [],
        'grout_pressures': []
    }

    # 严格校验：固定输入字段必须全部有效，禁止使用部分传感器数据兜底计算。
    missing_seal = [s for s in UNIFIED_SEAL_SENSORS if not _is_valid_pressure(_get_sensor_raw_value(data, s))]
    missing_grout = [s for s in UNIFIED_GROUT_SENSORS if not _is_valid_pressure(_get_sensor_raw_value(data, s))]

    missing = missing_seal + missing_grout
    if missing:
        raise ValueError(f"盾尾密封风险计算缺少必要字段: {', '.join(missing)}")

    for sensor in UNIFIED_SEAL_SENSORS:
        mapped_data['seal_pressures'].append(_to_sensor_value(_get_sensor_raw_value(data, sensor)))

    for sensor in UNIFIED_GROUT_SENSORS:
        mapped_data['grout_pressures'].append(_to_sensor_value(_get_sensor_raw_value(data, sensor)))

    return mapped_data


def calculate_universal_tail_seal_risk(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        mapped_data = map_project_data(data)
        seal_pressures = mapped_data['seal_pressures']
        grout_pressures = mapped_data['grout_pressures']

        risk_prob = 0.0
        valid_points = 0
        sensor_count = min(len(seal_pressures), len(grout_pressures))

        for i in range(sensor_count):
            P_seal, P_grout = seal_pressures[i], grout_pressures[i]

            if not (np.isnan(P_seal) or np.isnan(P_grout)):
                valid_points += 1
                point_risk = 0.0

                # 逻辑1：压差风险
                if P_grout > P_seal:
                    point_risk += 1 / 16

                # 逻辑2：绝对压力风险
                if P_seal >= 50:
                    point_risk += 1 / 16
                elif 40 <= P_seal < 50:
                    point_risk += (P_seal - 40) / 160

                risk_prob += point_risk

        if valid_points == 0:
            raise ValueError("有效盾尾密封传感器数据缺失")

        probability = min(risk_prob, 1.0)

        if probability >= 0.9:
            level = '高风险'
        elif probability >= 0.5:
            level = '中风险'
        elif probability >= 0.3:
            level = '低风险'
        else:
            level = '无风险'

        return {
            'probability': round(probability, 3),
            'risk_level': level,
            'details': f"有效点: {valid_points}/{sensor_count}, 累计概率: {probability:.3f}"
        }
    except Exception as e:
        raise RuntimeError(f"盾尾密封风险评估失败: {str(e)}")

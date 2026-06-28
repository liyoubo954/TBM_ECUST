from typing import Dict, List, Any

from app.risk.utils.sensor_validation import require_nonnegative_sensor_values

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


def map_project_data(data: Dict[str, Any]) -> Dict[str, List[float]]:
    values = require_nonnegative_sensor_values(data, UNIFIED_REQUIRED_PARAMS, "盾尾密封风险")
    return {
        'seal_pressures': [values[sensor] for sensor in UNIFIED_SEAL_SENSORS],
        'grout_pressures': [values[sensor] for sensor in UNIFIED_GROUT_SENSORS],
    }


def calculate_universal_tail_seal_risk(data: Dict[str, Any]) -> Dict[str, Any]:
    mapped_data = map_project_data(data)
    risk_prob = 0.0
    for seal_pressure, grout_pressure in zip(
        mapped_data['seal_pressures'],
        mapped_data['grout_pressures'],
    ):
        if grout_pressure > seal_pressure:
            risk_prob += 1 / 16
        if seal_pressure >= 50:
            risk_prob += 1 / 16
        elif seal_pressure >= 40:
            risk_prob += (seal_pressure - 40) / 160

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
        'details': f"有效点: 8/8, 累计概率: {probability:.3f}"
    }

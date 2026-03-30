import numpy as np
from typing import Dict, List, Any
UNIFIED_SEAL_SENSORS = [
    'ShieldTail.Seal.Rear.Pres.02', 'ShieldTail.Seal.Rear.Pres.04', 'ShieldTail.Seal.Rear.Pres.06', 'ShieldTail.Seal.Rear.Pres.09',
    'ShieldTail.Seal.Rear.Pres.12', 'ShieldTail.Seal.Rear.Pres.15', 'ShieldTail.Seal.Rear.Pres.17', 'ShieldTail.Seal.Rear.Pres.19'
]
UNIFIED_GROUT_SENSORS = [
    'Liquid.Pump.A.OutPres.01', 'Liquid.Pump.A.OutPres.02', 'Liquid.Pump.A.OutPres.03', 'Liquid.Pump.A.OutPres.04',
    'Liquid.Pump.A.OutPres.05', 'Liquid.Pump.A.OutPres.06', 'Liquid.Pump.A.OutPres.07', 'Liquid.Pump.A.OutPres.08'
]
UNIFIED_PARAMS_CHINESE = {
    'ShieldTail.Seal.Rear.Pres.02': '盾尾密封后压力02',
    'ShieldTail.Seal.Rear.Pres.04': '盾尾密封后压力04',
    'ShieldTail.Seal.Rear.Pres.06': '盾尾密封后压力06',
    'ShieldTail.Seal.Rear.Pres.09': '盾尾密封后压力09',
    'ShieldTail.Seal.Rear.Pres.12': '盾尾密封后压力12',
    'ShieldTail.Seal.Rear.Pres.15': '盾尾密封后压力15',
    'ShieldTail.Seal.Rear.Pres.17': '盾尾密封后压力17',
    'ShieldTail.Seal.Rear.Pres.19': '盾尾密封后压力19',
    'Liquid.Pump.A.OutPres.01': 'A液泵出口压力01',
    'Liquid.Pump.A.OutPres.02': 'A液泵出口压力02',
    'Liquid.Pump.A.OutPres.03': 'A液泵出口压力03',
    'Liquid.Pump.A.OutPres.04': 'A液泵出口压力04',
    'Liquid.Pump.A.OutPres.05': 'A液泵出口压力05',
    'Liquid.Pump.A.OutPres.06': 'A液泵出口压力06',
    'Liquid.Pump.A.OutPres.07': 'A液泵出口压力07',
    'Liquid.Pump.A.OutPres.08': 'A液泵出口压力08'
}

def map_project_data(data: Dict[str, Any]) -> Dict[str, List[float]]:
    mapped_data = {
        'seal_pressures': [],
        'grout_pressures': []
    }
    
    # 严格校验：如果所有的密封压力传感器或者注浆压力传感器都在数据中缺失，则抛出异常
    missing_seal = [s for s in UNIFIED_SEAL_SENSORS if s not in data or data[s] is None]
    missing_grout = [s for s in UNIFIED_GROUT_SENSORS if s not in data or data[s] is None]
    
    if len(missing_seal) == len(UNIFIED_SEAL_SENSORS) and len(missing_grout) == len(UNIFIED_GROUT_SENSORS):
         raise ValueError("盾尾密封风险计算缺少所有压力数据")

    for sensor in UNIFIED_SEAL_SENSORS:
        val = data.get(sensor)
        if val is not None and isinstance(val, (int, float)):
            mapped_data['seal_pressures'].append(float(val))
        else:
            mapped_data['seal_pressures'].append(np.nan) # 使用 NaN 表示缺失，后续计算会跳过该点
            
    for sensor in UNIFIED_GROUT_SENSORS:
        val = data.get(sensor)
        if val is not None and isinstance(val, (int, float)):
            mapped_data['grout_pressures'].append(float(val))
        else:
            mapped_data['grout_pressures'].append(np.nan) # 使用 NaN 表示缺失
            
    return mapped_data


# 入口函数保持兼容签名，但忽略 shield_id
def calculate_tail_seal_risk(data_dict: dict, shield_id: str = 'ignored') -> dict:
    try:
        result = calculate_universal_tail_seal_risk(data_dict)
        return {
            'probability': result.get('probability', 0.0),
            'risk_level': result.get('risk_level', '无风险'),
            'details': result.get('details', '无详细信息')
        }
    except Exception as e:
        raise


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


def risk_metadata() -> Dict[str, Any]:
    return {
        "name": "盾尾密封失效风险",
        "fields": [
            "ShieldTail.Seal.Rear.Pres.02", "ShieldTail.Seal.Rear.Pres.04", "ShieldTail.Seal.Rear.Pres.06", "ShieldTail.Seal.Rear.Pres.09",
            "ShieldTail.Seal.Rear.Pres.12", "ShieldTail.Seal.Rear.Pres.15", "ShieldTail.Seal.Rear.Pres.17", "ShieldTail.Seal.Rear.Pres.19",
            "Liquid.Pump.A.OutPres.01", "Liquid.Pump.A.OutPres.02", "Liquid.Pump.A.OutPres.03", "Liquid.Pump.A.OutPres.04",
            "Liquid.Pump.A.OutPres.05", "Liquid.Pump.A.OutPres.06", "Liquid.Pump.A.OutPres.07", "Liquid.Pump.A.OutPres.08"
        ],
        "map": {
            "ShieldTail.Seal.Rear.Pres.02": "盾尾密封后压力02",
            "ShieldTail.Seal.Rear.Pres.04": "盾尾密封后压力04",
            "ShieldTail.Seal.Rear.Pres.06": "盾尾密封后压力06",
            "ShieldTail.Seal.Rear.Pres.09": "盾尾密封后压力09",
            "ShieldTail.Seal.Rear.Pres.12": "盾尾密封后压力12",
            "ShieldTail.Seal.Rear.Pres.15": "盾尾密封后压力15",
            "ShieldTail.Seal.Rear.Pres.17": "盾尾密封后压力17",
            "ShieldTail.Seal.Rear.Pres.19": "盾尾密封后压力19",
            "Liquid.Pump.A.OutPres.01": "A液泵出口压力01",
            "Liquid.Pump.A.OutPres.02": "A液泵出口压力02",
            "Liquid.Pump.A.OutPres.03": "A液泵出口压力03",
            "Liquid.Pump.A.OutPres.04": "A液泵出口压力04",
            "Liquid.Pump.A.OutPres.05": "A液泵出口压力05",
            "Liquid.Pump.A.OutPres.06": "A液泵出口压力06",
            "Liquid.Pump.A.OutPres.07": "A液泵出口压力07",
            "Liquid.Pump.A.OutPres.08": "A液泵出口压力08",
        },
        "units": {
            "盾尾密封后压力02": "bar",
            "盾尾密封后压力04": "bar",
            "盾尾密封后压力06": "bar",
            "盾尾密封后压力09": "bar",
            "盾尾密封后压力12": "bar",
            "盾尾密封后压力15": "bar",
            "盾尾密封后压力17": "bar",
            "盾尾密封后压力19": "bar",
            "A液泵出口压力01": "bar",
            "A液泵出口压力02": "bar",
            "A液泵出口压力03": "bar",
            "A液泵出口压力04": "bar",
            "A液泵出口压力05": "bar",
            "A液泵出口压力06": "bar",
            "A液泵出口压力07": "bar",
            "A液泵出口压力08": "bar",
        },
    }


def get_measures_and_reason(risk_level: str):
    mapping = {
        "无风险Ⅰ": {
            "measures": ["正常监控：监控密封状态", "定期保养：检查密封分配器和管路"],
            "reason": "密封压力与注浆压力处于工况目标区间，密封刷贴合充分，无渗水或浆液串漏现象。"
        },
        "低风险Ⅱ": {
            "measures": ["正常监控：监控密封状态", "定期保养：检查密封分配器和管路"],
            "reason": "密封压力与注脂量轻微上调以维持密封效果，提示局部刷列初期磨耗或贴合度下降，个别位置可能存在微渗。"
        },
        "中风险Ⅲ": {
            "measures": ["加强密封监控：每小时检查一次密封状态", "检查密封圈状态：评估密封圈磨损程度"],
            "reason": "密封压力波动加剧，注脂量明显增加以对冲密封衰减；局部渗漏与刷列磨耗加重，屏障能力降低。"
        },
        "高风险Ⅳ": {
            "measures": ["立即停机：停止掘进作业", "检查密封系统状态：详细检查工作舱压力"],
            "reason": "刷列严重磨损、撕裂或压损导致渗漏明确，密封压力难以维持，同步注浆与防水能力显著下降。"
        }
    }
    d = mapping.get(risk_level)
    return (d.get("measures", []), d.get("reason", "")) if d else ([], "")


def probability_to_score(probability: float) -> float:
    p = float(probability or 0.0)
    if p >= 0.9:
        return 0.5 - (p - 0.9) * 0.5 / 0.1
    elif p >= 0.5:
        return 2 - (p - 0.5) * 1.5 / (0.9 - 0.5)
    elif p >= 0.3:
        return 3.5 - (p - 0.3) * 1.5 / (0.5 - 0.3)
    else:
        return 4.0 - p * 0.5 / 0.3


def reverse_score_to_probability(score: float) -> float:
    s = float(score or 0.0)
    if s <= 0.5:
        return 0.9 + (0.5 - s) * 0.2
    elif s <= 2:
        return 0.5 + (2 - s) * (0.4 / 1.5)
    elif s <= 3.5:
        return 0.3 + (3.5 - s) * (0.2 / 1.5)
    else:
        return (4.0 - s) * 0.6


def probability_thresholds():
    return 0.5, 0.3

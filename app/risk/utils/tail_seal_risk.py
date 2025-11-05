import numpy as np
from typing import Dict, List, Any
UNIFIED_SEAL_SENSORS = [
    'DW_PRS4.2', 'DW_PRS4.4', 'DW_PRS78', 'DW_PRS81',
    'DW_PRS84', 'DW_PRS87_2', 'DW_PRS89', 'DW_PRS91'
]
UNIFIED_GROUT_SENSORS = [
    'AYB_PRS1_1', 'AYB_PRS1_2', 'AYB_PRS1_3', 'AYB_PRS1_4',
    'AYB_PRS5', 'AYB_PRS6', 'AYB_PRS7', 'AYB_PRS8'
]
UNIFIED_PARAMS_CHINESE = {
    'DW_PRS4.2': '盾尾密封压力4.2',
    'DW_PRS4.4': '盾尾密封压力4.4',
    'DW_PRS78': '盾尾密封压力4.6',
    'DW_PRS81': '盾尾密封压力4.9',
    'DW_PRS84': '盾尾密封压力4.12',
    'DW_PRS87_2': '盾尾密封压力4.15',
    'DW_PRS89': '盾尾密封压力4.17',
    'DW_PRS91': '盾尾密封压力4.19',
    'AYB_PRS1_1': '1#A液泵出口压力',
    'AYB_PRS1_2': '2#A液泵出口压力',
    'AYB_PRS1_3': '3#A液泵出口压力',
    'AYB_PRS1_4': '4#A液泵出口压力',
    'AYB_PRS5': '5#A液泵出口压力',
    'AYB_PRS6': '6#A液泵出口压力',
    'AYB_PRS7': '7#A液泵出口压力',
    'AYB_PRS8': '8#A液泵出口压力'
}

def map_project_data(data: Dict[str, Any]) -> Dict[str, List[float]]:
    mapped_data = {
        'seal_pressures': [],
        'grout_pressures': []
    }
    for sensor in UNIFIED_SEAL_SENSORS:
        if sensor in data and isinstance(data[sensor], (int, float)):
            mapped_data['seal_pressures'].append(float(data[sensor]))
        else:
            mapped_data['seal_pressures'].append(0.0)
    for sensor in UNIFIED_GROUT_SENSORS:
        if sensor in data and isinstance(data[sensor], (int, float)):
            mapped_data['grout_pressures'].append(float(data[sensor]))
        else:
            mapped_data['grout_pressures'].append(0.0)
    return mapped_data


# 入口函数保持兼容签名，但忽略 shield_id
def calculate_tail_seal_risk(data_dict: dict, shield_id: str = 'ignored') -> dict:
    """
    盾尾密封失效风险计算入口函数（不区分项目，统一字段）。
    """
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
    """
    通用盾尾密封失效风险计算函数
    基于双重预警机制，对8个密封点分别计算风险，支持多项目字段映射

    双重预警逻辑：
    1. 第一种预警逻辑：如果注浆压力 > 密封压力 → 增加 1/16 风险
    2. 第二种预警逻辑：
       - 密封压力 < 40 → 不增加风险
       - 40 ≤ 密封压力 < 50 → 增加 (密封压力-40)/160 风险
       - 密封压力 ≥ 50 → 增加 1/16 风险

    风险等级判断：
    - 高风险：风险概率 ≥ 0.9
    - 中风险：0.5 ≤ 风险概率 < 0.9
    - 低风险：0 < 风险概率 < 0.5
    - 无风险：风险概率 = 0
    """
    try:
        # 使用统一字段映射
        mapped_data = map_project_data(data)
        
        seal_pressures = mapped_data['seal_pressures']  # 盾尾密封压力列表
        grout_pressures = mapped_data['grout_pressures']  # 注浆压力列表

        risk_prob = 0.0
        valid_points = 0
        warning_details = []

        # 确保两个列表长度一致，取较小值
        sensor_count = min(len(seal_pressures), len(grout_pressures))

        # 对每个密封点分别计算风险
        for i in range(sensor_count):
            sealing_pressure = seal_pressures[i]
            grouting_pressure = grout_pressures[i]

            if isinstance(sealing_pressure, (int, float)) and isinstance(grouting_pressure, (int, float)):
                if not (np.isnan(sealing_pressure) or np.isnan(grouting_pressure)):
                    valid_points += 1
                    point_risk = 0.0

                    # 第一种预警逻辑：注浆压力 > 密封压力
                    if grouting_pressure > sealing_pressure:
                        point_risk += 1 / 16  # 每个点贡献1/16的风险
                        warning_details.append(
                            f'点{i + 1}: 注浆压力({grouting_pressure:.2f}) > 密封压力({sealing_pressure:.2f})')

                    # 第二种预警逻辑：密封压力阈值判断
                    if sealing_pressure < 40:
                        # 密封压力 < 40，不增加风险
                        pass
                    elif 40 <= sealing_pressure < 50:
                        # 40 ≤ 密封压力 < 50，增加 (密封压力-40)/160 风险
                        additional_risk = (sealing_pressure - 40) / 160
                        point_risk += additional_risk
                        warning_details.append(f'点{i + 1}: 密封压力({sealing_pressure:.2f})处于中等风险区间')
                    else:  # sealing_pressure >= 50
                        # 密封压力 ≥ 50，增加 1/16 风险
                        point_risk += 1 / 16
                        warning_details.append(f'点{i + 1}: 密封压力({sealing_pressure:.2f})过高')

                    risk_prob += point_risk

        if valid_points == 0:
            raise ValueError('盾尾密封传感器数据无效')

        final_probability = min(risk_prob, 1.0)

        # 风险等级判定
        if final_probability >= 0.9:
            risk_level = '高风险'
        elif final_probability >= 0.5:
            risk_level = '中风险'
        elif final_probability >=0.3:
            risk_level = '低风险'
        else:
            risk_level = '无风险'

        # 构建详细信息（不区分项目）
        details_str = f'有效密封点数: {valid_points}/{sensor_count}, 累积风险概率: {final_probability:.3f}'
        if warning_details:
            details_str += f'\n预警详情: {"; ".join(warning_details[:3])}' + ('...' if len(warning_details) > 3 else '')

        return {
            'probability': round(final_probability, 3),
            'risk_level': risk_level,
            'details': details_str
        }

    except Exception as e:
        raise


def get_required_params_for_project(shield_id: str) -> Dict[str, str]:
    return UNIFIED_PARAMS_CHINESE

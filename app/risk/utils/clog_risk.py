import numpy as np
from typing import Dict, Any, List, Optional
UNIFIED_REQUIRED_PARAMS = {
    'GZC_PRS1': '工作仓压力1#',
    'KWC_PRS4': '开挖仓压力4',
    'PJB_JNK_PRS2.1': '排浆泵P2.1进泥口压力检测'
}
REQUIRED_EXCAVATION_PARAMS = UNIFIED_REQUIRED_PARAMS
CORE_PARAMS_CHINESE = {
    'bubble_pressure': '工作仓压力1#',
    'excavation_pressure': '开挖仓压力',
    'pump_pressure': '排浆泵P2.1进泥口压力检测'
}
def map_project_data(data: Dict[str, Any]) -> Dict[str, float]:
    mapped_data = {
        'bubble_pressure': data.get('GZC_PRS1', 0),
        'excavation_pressure': data.get('KWC_PRS4', 0),
        'pump_pressure': data.get('PJB_JNK_PRS2.1', 0)
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
def detect_data_quality(data: Dict[str, Any], sensor_key: str, threshold: float = 0.1) -> str:
    if not is_sensor_detected(data, sensor_key):
        return "无效数据"
    value = data[sensor_key]
    if abs(value) < threshold:
        return "低质量数据"
    return "正常数据"
def sliding_average_smooth(values: List[float], window_size: int = 3) -> List[float]:
    """滑动平均平滑处理"""
    if len(values) < window_size:
        return values
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(values), i + window_size // 2 + 1)
        window_values = values[start_idx:end_idx]
        smoothed.append(sum(window_values) / len(window_values))
    return smoothed
def calculate_pressure_difference(pressure1: float, pressure2: float) -> float:
    return pressure1 - pressure2
def calculate_universal_clog_risk(data: Dict[str, Any], shield_id: str = None) -> Dict[str, Any]:
    try:
        validations = validate_sensor_data(data, ['GZC_PRS1', 'KWC_PRS4', 'PJB_JNK_PRS2.1'])
        if not all(validations.values()):
            missing = [k for k, ok in validations.items() if not ok]
            raise ValueError(f'滞排风险传感器数据无效: {", ".join(missing)}')
        mapped_data = map_project_data(data)
        bubble_pressure = mapped_data['bubble_pressure']  # 气泡舱压力
        excavation_pressure = mapped_data['excavation_pressure']  # 开挖舱压力
        pump_pressure = mapped_data['pump_pressure']  # P2.1排浆泵吸口压力
        diff_bubble_exc = bubble_pressure - excavation_pressure  # 气泡舱压力 - 开挖舱压力
        diff_bubble_pump = bubble_pressure - pump_pressure  # 气泡舱压力 - P2.1排浆泵吸口压力
        # 规则5: 气泡舱压力 - 开挖舱压力 < -1 bar 且 气泡舱压力 - P2.1排浆泵吸口压力 > 2 bar
        # 最高风险：泥水环流系统滞排现象严重
        if diff_bubble_exc < -1 and diff_bubble_pump > 2:
            risk_level = '高风险'
            warning_message = '泥水环流系统滞排现象严重'
            probability = 0.9
        # 规则3: 气泡舱压力 - 开挖舱压力 < -1 bar
        # 中等风险：开挖舱内明显滞排
        elif diff_bubble_exc < -1:
            risk_level = '中风险'
            warning_message = '开挖舱内明显滞排'
            probability = 0.7
        # 规则4: 气泡舱压力 - P2.1排浆泵吸口压力 > 2 bar
        # 中等风险：泥水脱离格栅处明显滞排
        elif diff_bubble_pump > 2:
            risk_level = '中风险'
            warning_message = '泥水脱离格栅处明显滞排'
            probability = 0.7
        # 规则1: -1 bar ≤ 气泡舱压力 - 开挖舱压力 ≤ -0.5 bar
        # 低风险：开挖舱内轻微滞排
        elif -1 <= diff_bubble_exc <= -0.5:
            risk_level = '低风险'
            warning_message = '开挖舱内轻微滞排'
            probability = 0.3

        # 规则2: 1 bar ≤ 气泡舱压力 - P2.1排浆泵吸口压力 ≤ 2 bar
        # 低风险：泥水脱离格栅处轻微滞排
        elif 1 <= diff_bubble_pump <= 2:
            risk_level = '低风险'
            warning_message = '泥水脱离格栅处轻微滞排'
            probability = 0.3

        # 规则6: 其他情况
        # 无风险：泥水环流系统无滞排风险
        else:
            risk_level = '无风险'
            warning_message = '泥水环流系统无滞排风险'
            probability = 0.0

        return {
            'probability': round(probability, 3),
            'risk_level': risk_level,
            'details': f'{warning_message}，气泡舱-开挖舱压力差: {diff_bubble_exc:.2f} bar, 气泡舱-泵吸口压力差: {diff_bubble_pump:.2f} bar'
        }

    except Exception as e:
        raise RuntimeError(f'滞排风险计算错误: {str(e)}')


# 兼容性函数，保持原有接口
def calculate_clog_risk(data_dict: dict, shield_id: str = '6d62a15ef87fc3f4a80604e7547edc14') -> dict:
    """兼容性函数，返回风险概率值和风险等级"""
    try:
        # 调用通用风险计算函数，避免递归调用
        result = calculate_universal_clog_risk(data_dict, shield_id)
        # 返回包含概率和风险等级的字典
        return {
            'probability': result.get('probability', 0.0),
            'risk_level': result.get('risk_level', '无风险'),
            'details': result.get('details', '无详细信息')
        }
    except Exception as e:
        raise


def get_required_params_for_project(shield_id: str) -> Dict[str, str]:
    return REQUIRED_EXCAVATION_PARAMS

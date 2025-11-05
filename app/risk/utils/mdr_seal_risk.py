import numpy as np
from typing import Dict, Any
REQUIRED_EXCAVATION_PARAMS = {
    'GZC_PRS1': '工作仓压力1#',
    'YQMF_FK_PRS': '油气密封反馈压力',
    'ZQD_SS_PRS1': '1#主驱动伸缩密封油脂压力',
    'CLY_YQ_PRS': '齿轮油油气密封压力',
    'YQMF_XLJC_QPRS': '油气密封泄露检测腔压力'
}
CORE_PARAMS_CHINESE = {
    'P0': '工作仓压力1#',
    'P1': '油气密封反馈压力',
    'P2': '1#主驱动伸缩密封油脂压力',
    'P3': '齿轮油油气密封压力',
    'P4': '油气密封泄露检测腔压力'
}


def map_project_data(data: Dict[str, Any]) -> Dict[str, float]:
    mapped_data = {
        'P0': data.get('GZC_PRS1', 0),       # 工作仓压力1#
        'P1': data.get('YQMF_FK_PRS', 0),    # 油气密封反馈压力
        'P2': data.get('ZQD_SS_PRS1', 0),    # 1#主驱动伸缩密封油脂压力
        'P3': data.get('CLY_YQ_PRS', 0),     # 齿轮油油气密封压力
        'P4': data.get('YQMF_XLJC_QPRS', 0)  # 油气密封泄露检测腔压力
    }
    for key, value in mapped_data.items():
        if not isinstance(value, (int, float)) or np.isnan(value):
            mapped_data[key] = 0.0
        else:
            mapped_data[key] = float(value)
    return mapped_data

def calculate_mdr_seal_risk(data_dict: dict) -> dict:
    """主驱动密封失效风险计算入口函数（统一标准字段，无项目映射）"""
    try:
        result = calculate_universal_mdr_seal_risk(data_dict)
        return {
            'probability': result.get('probability', 0.0),
            'risk_level': result.get('risk_level', '无风险'),
            'details': result.get('details', '无详细信息')
        }
    except Exception as e:
        raise


def calculate_universal_mdr_seal_risk(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    通用主驱动密封失效风险计算函数（连续概率版）

    将三道密封的失效程度按差值幅度连续映射为 [0,1]，再进行组合，避免返回固定概率。
    """
    try:
        mapped = map_project_data(data)
        P0, P1, P2, P3, P4 = mapped['P0'], mapped['P1'], mapped['P2'], mapped['P3'], mapped['P4']
        if any(p < 0 for p in [P0, P1, P2, P3, P4]):
            raise ValueError('存在负压力值，数据异常')

        # 阈值
        DELTA_P1, DELTA_P2, DELTA_P3 = 1.0, 29.0, 4.0

        # 连续映射函数（缓增型）：x>=0 时 1-exp(-x/k)
        def soft_increase(x: float, k: float) -> float:
            x = float(x)
            return 0.0 if x <= 0 else 1.0 - np.exp(-x / max(k, 1e-6))

        # 第一、二、三道密封失效连续程度
        s1_a = soft_increase(P0 - P1 - DELTA_P1, 1.0)   # P0 > P1+ΔP1
        s1_b = soft_increase(P1 - P2, 10.0)             # P1 > P2
        s1 = max(s1_a, s1_b)

        s2_a = soft_increase(P3 - P2, 10.0)             # P3 > P2
        s2_b = soft_increase(P2 - P3 - DELTA_P2, 10.0)  # P2 > P3+ΔP2
        s2 = max(s2_a, s2_b)

        s3_a = soft_increase(P4 - P3, 5.0)              # P4 > P3
        s3_b = soft_increase(P3 - P4 - DELTA_P3, 5.0)   # P3 > P4+ΔP3
        s3 = max(s3_a, s3_b)

        # 组合：避免固定值，采用 1-乘积 形式（并发风险合成），可调权重
        w1, w2, w3 = 0.35, 0.35, 0.30
        combined = 1.0 - (1.0 - w1 * s1) * (1.0 - w2 * s2) * (1.0 - w3 * s3)
        probability = max(0.0, min(1.0, combined))

        # 风险等级按概率阈值映射
        if probability >= 0.9:
            risk_level = '高风险'
        elif probability >= 0.5:
            risk_level = '中风险'
        elif probability >= 0.3:
            risk_level = '低风险'
        else:
            risk_level = '无风险'

        details = (
            f"连续概率评估：s1={s1:.3f}, s2={s2:.3f}, s3={s3:.3f}; "
            f"P0={P0:.2f}, P1={P1:.2f}, P2={P2:.2f}, P3={P3:.2f}, P4={P4:.2f}; "
            f"prob={probability:.3f}"
        )
        return {
            'probability': round(probability, 3),
            'risk_level': risk_level,
            'details': details
        }
    except Exception as e:
        raise


def get_required_params_for_project(shield_id: str) -> Dict[str, str]:
    return REQUIRED_EXCAVATION_PARAMS
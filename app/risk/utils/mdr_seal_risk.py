import numpy as np
from typing import Dict, Any
REQUIRED_EXCAVATION_PARAMS = {
    'WorkCham.Pres.01': '工作舱压力01',
    'MB.InSeal.Grs.FeedPres': '油气密封反馈压力',
    'MD.TelSeal.Grs.Pres.01': '主驱动伸缩密封油脂压力01',
    'GOil.OilGasSeal.Pres': '齿轮油油气密封压力',
    'OilGasSeal.LeakDetCham.Pres': '油气密封泄露检测腔压力'
}
CORE_PARAMS_CHINESE = {
    'P0': '工作舱压力01',
    'P1': '油气密封反馈压力',
    'P2': '主驱动伸缩密封油脂压力01',
    'P3': '齿轮油油气密封压力',
    'P4': '油气密封泄露检测腔压力'
}


def map_project_data(data: Dict[str, Any]) -> Dict[str, float]:
    # 严格校验：如果必要字段缺失，直接抛出异常，而不是默认 0
    missing = [k for k in REQUIRED_EXCAVATION_PARAMS.keys() if k not in data or data[k] is None]
    if missing:
        raise ValueError(f"主驱动密封风险计算缺少必要字段: {', '.join(missing)}")

    mapped_data = {
        'P0': data.get('WorkCham.Pres.01'),       # 工作仓压力1#
        'P1': data.get('MB.InSeal.Grs.FeedPres'),    # 油气密封反馈压力
        'P2': data.get('MD.TelSeal.Grs.Pres.01'),    # 1#主驱动伸缩密封油脂压力
        'P3': data.get('GOil.OilGasSeal.Pres'),     # 齿轮油油气密封压力
        'P4': data.get('OilGasSeal.LeakDetCham.Pres')  # 油气密封泄露检测腔压力
    }
    for key, value in mapped_data.items():
        if not isinstance(value, (int, float)) or np.isnan(value):
            mapped_data[key] = 0.0
        else:
            mapped_data[key] = float(value)
    return mapped_data

def calculate_mdr_seal_risk(data_dict: dict) -> dict:
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
    try:
        mapped = map_project_data(data)
        P0, P1, P2, P3, P4 = mapped['P0'], mapped['P1'], mapped['P2'], mapped['P3'], mapped['P4']
        
        if any(p < 0 for p in [P0, P1, P2, P3, P4]):
            raise ValueError("存在负压力值，数据异常")

        # 阈值配置
        DELTA_P1, DELTA_P2, DELTA_P3 = 1.0, 29.0, 4.0

        def soft_increase(x: float, k: float) -> float:
            return 0.0 if x <= 0 else 1.0 - np.exp(-x / max(k, 1e-6))

        # 三道密封失效连续程度计算
        s1 = max(soft_increase(P0 - P1 - DELTA_P1, 1.0), soft_increase(P1 - P2, 10.0))
        s2 = max(soft_increase(P3 - P2, 10.0), soft_increase(P2 - P3 - DELTA_P2, 10.0))
        s3 = max(soft_increase(P4 - P3, 5.0), soft_increase(P3 - P4 - DELTA_P3, 5.0))

        # 并发风险合成
        w1, w2, w3 = 0.35, 0.35, 0.30
        probability = max(0.0, min(1.0, 1.0 - (1.0 - w1 * s1) * (1.0 - w2 * s2) * (1.0 - w3 * s3)))

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
            'details': f"prob={probability:.3f} (s1={s1:.2f}, s2={s2:.2f}, s3={s3:.2f})"
        }
    except Exception as e:
        raise RuntimeError(f"主驱动密封风险评估失败: {str(e)}")


def risk_metadata() -> Dict[str, Any]:
    return {
        "name": "主驱动密封失效风险",
        "fields": ["WorkCham.Pres.01", "MB.InSeal.Grs.FeedPres", "MD.TelSeal.Grs.Pres.01", "GOil.OilGasSeal.Pres", "OilGasSeal.LeakDetCham.Pres"],
        "map": {
            "WorkCham.Pres.01": "工作舱压力01",
            "MB.InSeal.Grs.FeedPres": "油气密封反馈压力",
            "MD.TelSeal.Grs.Pres.01": "主驱动伸缩密封油脂压力01",
            "GOil.OilGasSeal.Pres": "齿轮油油气密封压力",
            "OilGasSeal.LeakDetCham.Pres": "油气密封泄露检测腔压力",
        },
        "units": {
            "工作舱压力01": "bar",
            "油气密封反馈压力": "bar",
            "主驱动伸缩密封油脂压力01": "bar",
            "齿轮油油气密封压力": "bar",
            "油气密封泄露检测腔压力": "bar",
        },
    }


def get_measures_and_reason(risk_level: str):
    mapping = {
        "无风险Ⅰ": {
            "measures": ["定期保养：执行密封系统保养", "正常监控：记录密封压力数据"],
            "reason": "工作舱压力稳定，密封反馈压力正常，主驱动密封压力适宜，刀盘转速平稳，刀盘扭矩稳定，密封检测压力在安全范围内，密封系统运行参数均在理想范围内。"
        },
        "低风险Ⅱ": {
            "measures": ["定期保养：执行密封系统保养", "正常监控：记录密封压力数据"],
            "reason": "工作舱压力略有波动，密封反馈压力小幅变化，主驱动密封压力轻微下降，刀盘转速略有波动，刀盘扭矩小幅变化，密封检测压力轻微升高，但密封系统整体性能良好，无渗漏现象。"
        },
        "中风险Ⅲ": {
            "measures": ["加强密封监控：将密封压力监测频率提高到每30分钟一次", "检查压力梯度：详细分析各道密封之间的压力梯度"],
            "reason": "检测压力升高与密封压力不稳并存，梯度失衡加剧，刀盘负荷与振动出现波动，伴随间歇性渗漏征兆，综合表明密封功能下降、油脂屏障削弱。"
        },
        "高风险Ⅳ": {
            "measures": ["立即停机：停止掘进作业", "全面检查密封压力：详细检查工作舱压力"],
            "reason": "密封系统屏障功能显著失效，压力异常与渗漏明确，外界介质侵入主轴承腔风险高，存在严重设备损伤风险。"
        }
    }
    d = mapping.get(risk_level)
    return (d.get("measures", []), d.get("reason", "")) if d else ([], "")


def probability_to_score(probability: float) -> float:
    p = float(probability or 0.0)
    if p >= 0.95:
        return 0.5 - (p - 0.95) * 0.5 / 0.05
    elif p >= 0.5:
        return 2 - (p - 0.5) * 1.5 / 0.45
    else:
        return 4.0 - p * 2 / 0.5


def reverse_score_to_probability(score: float) -> float:
    s = float(score or 0.0)
    if s <= 0.5:
        return 0.95 + (0.5 - s) * 0.1
    elif s <= 2:
        return 0.5 + (2 - s) * 0.3
    else:
        return (4.0 - s) * 0.25


def probability_thresholds():
    return 0.5, 0.125

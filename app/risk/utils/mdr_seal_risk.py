import numpy as np
from typing import Dict, Any

UNIFIED_REQUIRED_PARAMS = {
    'WorkCham.Prs.01': '1#工作仓压力',
    'EP2.OutSeal.Grs.Prs.01': '1#主驱动EP2外密封油脂压力',
    'MD.Tele.SealGrs.Prs.01': '主驱动伸缩密封油脂压力01',
    'GearOil.AirOil.Seal.Prs': '齿轮油油气密封压力',
    'AirOil.Seal.LeakDetCham.Prs': '油气密封泄露检测腔压力'
}

RISK_SPEC = {
    "name": "主驱动密封失效风险",
    "risk_type_label": "主驱动密封失效",
    "full_risk_type": "主驱动密封失效风险",
    "output_key": "mdr_seal_risk",
    "fault_cause": "主轴承密封系统因磨损老化或密封油脂压力不足，导致外部渣土或地下水浸入",
    "potential_risk": "主驱动密封失效预警",
    "fields": list(UNIFIED_REQUIRED_PARAMS.keys()),
    "map": dict(UNIFIED_REQUIRED_PARAMS),
    "units": {
        "1#工作仓压力": "bar",
        "1#主驱动EP2外密封油脂压力": "bar",
        "主驱动伸缩密封油脂压力01": "bar",
        "齿轮油油气密封压力": "bar",
        "油气密封泄露检测腔压力": "bar",
    },
    "score_points": [(0.0, 4.0), (0.5, 2.0), (0.95, 0.5), (1.0, 0.0)],
    "probability_thresholds": (0.5, 0.125),
    "fault_reason_analysis": {
        "无风险Ⅰ": "工作仓压力、油气密封反馈压力、主驱动伸缩密封油脂压力、齿轮油油气密封压力或密封检测压力整体处于稳态，密封压力梯度合理。无渗漏迹象，油脂供给与流变性能正常，密封件弹性与磨耗处健康区间。",
        "低风险Ⅱ": "反馈或检测通道压力轻微偏移，梯度呈早期紊乱但尚能维持屏障。可能存在密封唇口初期磨耗、供脂压力波动或回油不畅等趋势，偶发微渗与油脂消耗偏快，建议提高监测敏感度并复核供脂策略与间隙设定。",
        "中风险Ⅲ": "检测压力升高与密封压力不稳并存，梯度失衡加剧；刀盘负荷与振动出现相关性波动，伴随间歇性渗漏征兆。综合表明密封功能下降、油脂屏障削弱，外界介质侵入风险上升，若不处置将危及主轴承清洁度与寿命。",
        "高风险Ⅳ": "密封系统屏障功能显著失效，压力异常与渗漏明确，油脂难以维持有效隔离。继续运行将导致土水介质进入主轴承腔，污染润滑与加速磨损，存在严重设备损伤风险，应在保护条件下尽快停机并修复或更换密封组件。",
    },
    "measures": {
        "无风险Ⅰ": {
            "measures": ["定期保养：执行密封系统保养", "正常监控：记录密封压力数据"],
            "reason": "工作舱压力稳定，密封反馈压力正常，主驱动密封压力适宜，刀盘转速平稳，刀盘扭矩稳定，密封检测压力在安全范围内，密封系统运行参数均在理想范围内。",
        },
        "低风险Ⅱ": {
            "measures": ["定期保养：执行密封系统保养", "正常监控：记录密封压力数据"],
            "reason": "工作舱压力略有波动，密封反馈压力小幅变化，主驱动密封压力轻微下降，刀盘转速略有波动，刀盘扭矩小幅变化，密封检测压力轻微升高，但密封系统整体性能良好，无渗漏现象。",
        },
        "中风险Ⅲ": {
            "measures": ["加强密封监控：将密封压力监测频率提高到每30分钟一次", "检查压力梯度：详细分析各道密封之间的压力梯度"],
            "reason": "检测压力升高与密封压力不稳并存，梯度失衡加剧，刀盘负荷与振动出现波动，伴随间歇性渗漏征兆，综合表明密封功能下降、油脂屏障削弱。",
        },
        "高风险Ⅳ": {
            "measures": ["立即停机：停止掘进作业", "全面检查密封压力：详细检查工作舱压力"],
            "reason": "密封系统屏障功能显著失效，压力异常与渗漏明确，外界介质侵入主轴承腔风险高，存在严重设备损伤风险。",
        },
    },
}


def to_valid_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return value
    except (TypeError, ValueError):
        return 0.0


def map_project_data(data: Dict[str, Any]) -> Dict[str, float]:
    # 严格校验：如果必要字段缺失，直接抛出异常，而不是默认 0
    missing = [k for k in UNIFIED_REQUIRED_PARAMS.keys() if k not in data or data[k] is None]
    if missing:
        raise ValueError(f"主驱动密封风险计算缺少必要字段: {', '.join(missing)}")

    mapped_data = {
        'P0': data.get('WorkCham.Prs.01'),
        'P1': data.get('EP2.OutSeal.Grs.Prs.01'),
        'P2': data.get('MD.Tele.SealGrs.Prs.01'),
        'P3': data.get('GearOil.AirOil.Seal.Prs'),
        'P4': data.get('AirOil.Seal.LeakDetCham.Prs'),
    }
    return {key: to_valid_float(value) for key, value in mapped_data.items()}


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

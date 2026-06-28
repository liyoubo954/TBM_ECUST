import numpy as np
from typing import Dict, Any

from app.risk.utils.sensor_validation import require_finite_sensor_values

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


def map_project_data(data: Dict[str, Any]) -> Dict[str, float]:
    values = require_finite_sensor_values(data, UNIFIED_REQUIRED_PARAMS, "主驱动密封风险")
    return {
        'P0': values['WorkCham.Prs.01'],
        'P1': values['EP2.OutSeal.Grs.Prs.01'],
        'P2': values['MD.Tele.SealGrs.Prs.01'],
        'P3': values['GearOil.AirOil.Seal.Prs'],
        'P4': values['AirOil.Seal.LeakDetCham.Prs'],
    }


def calculate_universal_mdr_seal_risk(data: Dict[str, Any]) -> Dict[str, Any]:
    mapped = map_project_data(data)
    P0, P1, P2, P3, P4 = (mapped[f'P{i}'] for i in range(5))
    delta_p1, delta_p2, delta_p3 = 1.0, 29.0, 4.0

    def soft_increase(x: float, k: float) -> float:
        return 0.0 if x <= 0 else 1.0 - np.exp(-x / k)

    s1 = max(soft_increase(P0 - P1 - delta_p1, 1.0), soft_increase(P1 - P2, 10.0))
    s2 = max(soft_increase(P3 - P2, 10.0), soft_increase(P2 - P3 - delta_p2, 10.0))
    s3 = max(soft_increase(P4 - P3, 5.0), soft_increase(P3 - P4 - delta_p3, 5.0))

    w1, w2, w3 = 0.35, 0.35, 0.30
    probability = 1.0 - (1.0 - w1 * s1) * (1.0 - w2 * s2) * (1.0 - w3 * s3)

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

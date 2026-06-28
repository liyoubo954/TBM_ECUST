from typing import Dict, Any

from app.risk.utils.sensor_validation import require_nonnegative_sensor_values

LOW_RISK_PROBABILITY = 0.3
MEDIUM_RISK_PROBABILITY = 0.7
HIGH_RISK_PROBABILITY = 0.9

UNIFIED_REQUIRED_PARAMS = {
    'WorkCham.Prs.01': '1#工作仓压力',
    'ExcCham.Prs.04': '4#开挖仓压力',
    'DischPump.P2.1.In.Prs': '排浆泵P2.1进口压力'
}
        
RISK_SPEC = {
    "name": "滞排风险",
    "risk_type_label": "滞排",
    "full_risk_type": "滞排风险",
    "fault_cause": "渣土改良不充分导致流动性不佳，最终在管道内发生滞排",
    "potential_risk": "滞排预警",
    "fields": list(UNIFIED_REQUIRED_PARAMS.keys()),
    "map": dict(UNIFIED_REQUIRED_PARAMS),
    "units": {
        "1#工作仓压力": "bar",
        "4#开挖仓压力": "bar",
        "排浆泵P2.1进口压力": "bar",            
    },
    "score_points": [(0.0, 4.0), (0.3, 3.5), (0.7, 2.0), (0.9, 0.5), (1.0, 0.0)],
    "fault_reason_analysis": {
        "无风险Ⅰ": "开挖仓压力与工作仓压力稳定，排浆泵P2.1进口压力处于正常区间，压力梯度合理、流态均匀。渣土含水率与改良剂配比匹配，管路内无沉积或团聚迹象，排量连续可控。",
        "低风险Ⅱ": "管路摩阻略增，泵出口与进口压力出现轻微非同步波动，短时排量下降提示流动性边界化。渣土可能存在剪切变稀临界、团聚核初始形成或颗粒级配偏离，需关注参数微调以避免在弯头、缩径或高阻段形成稳定堵塞核。",
        "中风险Ⅲ": "进口压力显著抬升且泵压波动加剧，表征管内已有部分堵塞或团聚带。压力梯度异常、回流概率上升、脉动增大，局部存在沉积与再悬浮循环。根因多与改良不足、含水率不当、细粒黏性偏高或输送能级不匹配相关，需及时分段冲洗并联动调整改良与泵速以恢复连续性。",
        "高风险Ⅳ": "系统出现严重滞排或近乎完全堵塞，泵压异常、排量骤降甚至中断，压力梯度失衡并伴随强烈脉动。继续掘进将导致设备过载、密封与管路风险叠加，存在安全与质量隐患，应尽快在安全窗口下清障并重建稳定输送条件。",
    },
    "measures": {
        "无风险Ⅰ": {
            "measures": ["维持状态：保持当前掘进参数", "正常作业：按照标准流程进行"],
            "reason": "开挖仓压力稳定，工作仓压力正常，排浆泵压力平稳，排泥系统运行平稳，渣土含水率适宜，排泥管道通畅，刀盘转速和刀盘扭矩保持稳定，渣土排出连续顺畅。",
        },
        "低风险Ⅱ": {
            "measures": ["监控压力趋势：每15分钟记录一次压力", "调整掘进参数：适当降低推进速度"],
            "reason": "开挖舱压力略有波动，工作舱压力轻微升高，排浆泵压力小幅变化，排泥速度略有下降，刀盘扭矩略有增加，渣土含水率略有变化，但排泥系统整体运行仍在安全范围内。",
        },
        "中风险Ⅲ": {
            "measures": ["加强密封监控：将密封压力监测频率提高到每30分钟一次", "检查压力梯度：详细分析各道密封之间的压力梯度"],
            "reason": "开挖仓压力不稳定，工作仓压力明显升高，排浆泵压力波动较大，排泥速度明显下降，刀盘扭矩波动明显，渣土含水率异常。",
        },
        "高风险Ⅳ": {
            "measures": ["立即停机：停止掘进作业", "检查压力系统：全面检查气垫舱压力"],
            "reason": "开挖仓压力异常，工作舱压力过高，排浆泵压力急剧变化，刀盘扭矩急剧上升，排泥系统出现明显阻塞，渣土几乎无法排出。",
        },
    },
}


def map_project_data(data: Dict[str, Any]):
    values = require_nonnegative_sensor_values(data, UNIFIED_REQUIRED_PARAMS, "滞排风险")
    return {
        'P_bubble': values['WorkCham.Prs.01'],
        'P_excav': values['ExcCham.Prs.04'],
        'P_pump': values['DischPump.P2.1.In.Prs'],
    }


def calculate_universal_clog_risk(data: Dict[str, Any]) -> Dict[str, Any]:
    mapped = map_project_data(data)
    diff_bubble_exc = mapped['P_bubble'] - mapped['P_excav']
    diff_bubble_pump = mapped['P_bubble'] - mapped['P_pump']

    if diff_bubble_exc < -1 and diff_bubble_pump > 2:
        risk_level, msg, prob = '高风险', '泥水环流系统滞排现象严重', HIGH_RISK_PROBABILITY
    elif diff_bubble_exc < -1:
        risk_level, msg, prob = '中风险', '开挖舱内明显滞排', MEDIUM_RISK_PROBABILITY
    elif diff_bubble_pump > 2:
        risk_level, msg, prob = '中风险', '泥水脱离格栅处明显滞排', MEDIUM_RISK_PROBABILITY
    elif -1 <= diff_bubble_exc <= -0.5:
        risk_level, msg, prob = '低风险', '开挖舱内轻微滞排', LOW_RISK_PROBABILITY
    elif 1 <= diff_bubble_pump <= 2:
        risk_level, msg, prob = '低风险', '泥水脱离格栅处轻微滞排', LOW_RISK_PROBABILITY
    else:
        risk_level, msg, prob = '无风险', '泥水环流系统无滞排风险', 0.0

    return {
        'probability': round(prob, 3),
        'risk_level': risk_level,
        'details': f"{msg} (ΔP_仓={diff_bubble_exc:.2f}, ΔP_泵={diff_bubble_pump:.2f})"
    }

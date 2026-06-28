from typing import Any, Dict, Iterable

import numpy as np


def require_finite_sensor_values(
    data: Dict[str, Any],
    fields: Iterable[str],
    risk_name: str,
) -> Dict[str, float]:
    if not isinstance(data, dict):
        raise ValueError(f"{risk_name}计算输入必须为字典")

    values: Dict[str, float] = {}
    invalid = []
    for field in fields:
        value = data.get(field)
        if value is None or isinstance(value, bool):
            invalid.append(field)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            invalid.append(field)
            continue
        if not np.isfinite(number):
            invalid.append(field)
            continue
        values[field] = number

    if invalid:
        raise ValueError(f"{risk_name}计算缺少或包含无效必要字段: {', '.join(invalid)}")
    return values

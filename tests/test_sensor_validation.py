import unittest

import numpy as np

from app.risk.utils.clog_risk import (
    UNIFIED_REQUIRED_PARAMS as CLOG_FIELDS,
    calculate_universal_clog_risk,
)
from app.risk.utils.mdr_seal_risk import (
    UNIFIED_REQUIRED_PARAMS as MDR_SEAL_FIELDS,
    calculate_universal_mdr_seal_risk,
)
from app.risk.utils.sensor_validation import require_finite_sensor_values
from app.risk.utils.tail_seal_risk import (
    UNIFIED_REQUIRED_PARAMS as TAIL_SEAL_FIELDS,
    calculate_universal_tail_seal_risk,
)


class SensorValidationTests(unittest.TestCase):
    def test_finite_negative_values_are_preserved(self):
        values = require_finite_sensor_values(
            {"negative": -3.1, "zero": 0, "positive": "2.5"},
            ("negative", "zero", "positive"),
            "test risk",
        )

        self.assertEqual(
            values,
            {"negative": -3.1, "zero": 0.0, "positive": 2.5},
        )

    def test_invalid_values_are_rejected(self):
        invalid_values = (None, "", True, "not-a-number", np.nan, np.inf, -np.inf)

        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    require_finite_sensor_values(
                        {"sensor": invalid_value},
                        ("sensor",),
                        "test risk",
                    )

    def test_missing_field_is_rejected(self):
        with self.assertRaises(ValueError):
            require_finite_sensor_values({}, ("sensor",), "test risk")

    def test_three_risk_calculators_accept_finite_negative_values(self):
        calculators_and_fields = (
            (calculate_universal_clog_risk, CLOG_FIELDS),
            (calculate_universal_mdr_seal_risk, MDR_SEAL_FIELDS),
            (calculate_universal_tail_seal_risk, TAIL_SEAL_FIELDS),
        )

        for calculator, fields in calculators_and_fields:
            data = {field: 1.0 for field in fields}
            data[next(iter(fields))] = -3.1

            with self.subTest(calculator=calculator.__name__):
                result = calculator(data)
                self.assertIn("probability", result)
                self.assertIn("risk_level", result)


if __name__ == "__main__":
    unittest.main()

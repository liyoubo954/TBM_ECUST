from flask import Blueprint, request, jsonify
from influxdb import InfluxDBClient
from app.risk.utils.clog_risk import (
    RISK_SPEC as CLOG_RISK_SPEC,
    calculate_universal_clog_risk as calculate_clog_risk,
)
from app.risk.utils.mdr_seal_risk import (
    RISK_SPEC as MDR_SEAL_RISK_SPEC,
    calculate_universal_mdr_seal_risk as calculate_mdr_seal_risk,
)
from app.risk.utils.tail_seal_risk import (
    RISK_SPEC as TAIL_SEAL_RISK_SPEC,
    calculate_universal_tail_seal_risk as calculate_tail_seal_risk,
)
from app.risk.utils.mud_cake_risk import (
    GEO_FEATURES,
    TBM_FEATURES,
    RISK_SPEC as MUD_CAKE_RISK_SPEC,
    StratumMudCakeRiskCalculator,
)
import traceback
import os
import json
import re
import time
import pandas as pd
from datetime import datetime, timedelta
import pymysql
import threading
import queue
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed


def _env_int(name, default):
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'sf9glGVa20pyM1NtdukZ'

    INFLUXDB_HOST = os.environ.get('INFLUXDB_HOST') or '192.168.211.108'
    INFLUXDB_PORT = _env_int('INFLUXDB_PORT', 38086)
    INFLUXDB_USERNAME = os.environ.get('INFLUXDB_USERNAME') or 'admin'
    INFLUXDB_PASSWORD = os.environ.get('INFLUXDB_PASSWORD') or 'FZaStb0cXFuFbehPBM6YHCiuAAX6QIXr'
    INFLUXDB_DATABASE = os.environ.get('INFLUXDB_DATABASE') or 'algorithm'
    INFLUXDB_TIMEOUT = _env_int('INFLUXDB_TIMEOUT', 10)

    MYSQL_HOST = os.environ.get("MYSQL_HOST", "172.16.105.12")
    MYSQL_PORT = _env_int("MYSQL_PORT", 13366)
    MYSQL_USER = os.environ.get("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "123456")
    MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "algorithm")
    MYSQL_READ_TIMEOUT = _env_int("MYSQL_READ_TIMEOUT", 8)
    MYSQL_WRITE_TIMEOUT = _env_int("MYSQL_WRITE_TIMEOUT", 8)
    MYSQL_MAX_EXECUTION_MS = _env_int("MYSQL_MAX_EXECUTION_MS", 3000)

    POINT_INFO_MYSQL_HOST = os.environ.get("POINT_INFO_MYSQL_HOST", MYSQL_HOST)
    POINT_INFO_MYSQL_PORT = _env_int("POINT_INFO_MYSQL_PORT", MYSQL_PORT)
    POINT_INFO_MYSQL_USER = os.environ.get("POINT_INFO_MYSQL_USER", MYSQL_USER)
    POINT_INFO_MYSQL_PASSWORD = os.environ.get("POINT_INFO_MYSQL_PASSWORD", MYSQL_PASSWORD)
    POINT_INFO_MYSQL_DATABASE = os.environ.get("POINT_INFO_MYSQL_DATABASE", "point_info")

    HISTORY_TARGET_DB_NAME = os.environ.get("HISTORY_TARGET_DB_NAME", "tsjy_dz1360_cleanDate")
    HISTORY_PROJECT_PREFIX = os.environ.get("HISTORY_PROJECT_PREFIX", "tsjy_dz1360")
    LATEST_STATE_TARGET_DB = os.environ.get("LATEST_STATE_TARGET_DB", "tsjy_dz1360_map")


CONSECUTIVE_TRIGGER_N = int(os.environ.get('CONSECUTIVE_TRIGGER_N', '3'))
MUD_CAKE_SEQUENCE_RING_COUNT = 6
INFLUXDB_HOST = Config.INFLUXDB_HOST
INFLUXDB_PORT = Config.INFLUXDB_PORT
INFLUXDB_USERNAME = Config.INFLUXDB_USERNAME
INFLUXDB_PASSWORD = Config.INFLUXDB_PASSWORD
INFLUXDB_DATABASE = Config.INFLUXDB_DATABASE
INFLUXDB_TIMEOUT = Config.INFLUXDB_TIMEOUT
client = InfluxDBClient(
    host=INFLUXDB_HOST,
    port=INFLUXDB_PORT,
    username=INFLUXDB_USERNAME,
    password=INFLUXDB_PASSWORD,
    database=INFLUXDB_DATABASE,
    timeout=INFLUXDB_TIMEOUT
)

SHIELD_CONFIG = {
    "gag_dl1116": {"measurement": "gag_dl1116_riskwarning", "project_code": "GAG_DL1116"},
    "htcjsd_dz1368": {"measurement": "htcjsd_dz1368_riskwarning", "project_code": "HTCJSD_DZ1368"},
    "tsjy_dz1360": {"measurement": "tsjy_dz1360_riskwarning", "project_code": "TSJY_DZ1360"},
    "yztl_dz1266": {"measurement": "yztl_dz1266_riskwarning", "project_code": "YZTL_DZ1266"},
    "sjtl_dl898": {"measurement": "sjtl_dl898_riskwarning", "project_code": "SJTL_DL898"},
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _quote_influx_identifier(identifier):
    text = str(identifier or "").strip()
    if not text:
        raise ValueError("InfluxDB identifier cannot be empty")
    if any(ch in text for ch in ('"', "\\", "\x00", "\n", "\r")):
        raise ValueError(f"Unsafe InfluxDB identifier: {text}")
    return f'"{text}"'


def _quote_mysql_identifier(identifier):
    text = str(identifier or "").strip()
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(f"Unsafe MySQL identifier: {text}")
    return f"`{text}`"


class InvalidShieldIdError(ValueError):
    pass


def _resolve_shield_context(shield_id: str):
    sid = str(shield_id or "").strip().lower()
    if not sid:
        raise InvalidShieldIdError("缺少 shield_id 参数")
    if not _IDENTIFIER_RE.fullmatch(sid):
        raise InvalidShieldIdError("shield_id 格式非法")
    config = SHIELD_CONFIG.get(sid)
    if config is None:
        raise InvalidShieldIdError(
            f"不支持的 shield_id: {sid}；支持值: {', '.join(sorted(SHIELD_CONFIG))}"
        )
    return config["measurement"], config["project_code"]


POINT_MAP_CACHE = {}
GEO_SOURCE_RING_CACHE = None
RISK_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_CLUSTER_CSV = os.environ.get(
    "GEO_CLUSTER_CSV",
    os.path.join(RISK_DIR, "合并数据_环号聚类结果_按环聚合_K6.csv"),
)

SOURCE_FILE_ALIASES = {
    "gag_dl1116": "广澳港处理后的数据",
    "GAG_DL1116": "广澳港处理后的数据",
    "htcjsd_dz1368": "海太处理后数据.xlsx",
    "HTCJSD_DZ1368": "海太处理后数据.xlsx",
    "tsjy_dz1360": "通苏嘉甬处理后的数据",
    "TSJY_DZ1360": "通苏嘉甬处理后的数据",
    "通苏嘉甬": "通苏嘉甬处理后的数据",
    "yztl_dz1266": "甬舟铁路处理后的数据",
    "YZTL_DZ1266": "甬舟铁路处理后的数据",
    "甬舟铁路": "甬舟铁路处理后的数据",
    "甬舟": "甬舟铁路处理后的数据",
    "sjtl_dl898": "深江铁路处理后的数据",
    "SJTL_DL898": "深江铁路处理后的数据",
    "深江铁路": "深江铁路处理后的数据",
}
RISK_CONFIG = {
    "clog_risk": CLOG_RISK_SPEC,
    "mdr_seal_risk": MDR_SEAL_RISK_SPEC,
    "tail_seal_risk": TAIL_SEAL_RISK_SPEC,
    "mud_cake_risk": MUD_CAKE_RISK_SPEC,
}

RISK_DISPLAY_FIELDS = {
    "mud_cake_risk": list(TBM_FEATURES),
    "clog_risk": list(CLOG_RISK_SPEC["fields"]),
    "mdr_seal_risk": list(MDR_SEAL_RISK_SPEC["fields"]),
    "tail_seal_risk": list(TAIL_SEAL_RISK_SPEC["fields"]),
}

TAIL_SEAL_KEY_PARAMETER_FIELDS = [
    "LiqPump.A.Out.Prs.01",
    "LiqPump.A.Out.Prs.02",
    "Tail.Seal.Rear.Prs.02",
    "Tail.Seal.Rear.Prs.04",
]

RISK_TABLES = {
    "滞排": "clog_risk",
    "主驱动密封失效": "mdr_seal_risk",
    "结泥饼": "mud_cake_risk",
    "盾尾密封失效": "tail_seal_risk",
}


RISK_OUTPUT_SPECS = {
    key: {
        "output_key": spec["output_key"],
        "risk_type_label": spec["risk_type_label"],
        "full_risk_type": spec["full_risk_type"],
        "fault_cause": spec["fault_cause"],
    }
    for key, spec in RISK_CONFIG.items()
}

WARNING_LEVELS = {"低风险Ⅱ", "中风险Ⅲ", "高风险Ⅳ"}
LATEST_RISK_ROW_COLUMNS = [
    "ring",
    "project_code",
    "key_parameters",
    "safety_level",
    "risk_level",
    "risk_score",
    "potential_risk",
    "fault_reason_analysis",
    "fault_cause",
    "fault_measures",
    "measures",
    "impact_parameters",
]

_TTL_CACHE_LOCK = threading.Lock()
_TTL_CACHES = {
    "risk_level_result": {},
    "mysql_latest_row": {},
    "mysql_table_columns": {},
    "point_info_tables": {},
    "history_parameters": {},
    "latest_state": {},
}


def _ttl_cache_get(cache_name: str, key):
    now = time.time()
    with _TTL_CACHE_LOCK:
        cache = _TTL_CACHES.get(cache_name) or {}
        item = cache.get(key)
        if not item:
            return None
        exp, value = item
        if exp is not None and exp < now:
            cache.pop(key, None)
            return None
        return value


def _ttl_cache_set(cache_name: str, key, value, ttl_sec: float, max_items: int = 1024):
    try:
        ttl = float(ttl_sec or 0.0)
    except Exception:
        ttl = 0.0
    exp = (time.time() + ttl) if ttl > 0 else None
    with _TTL_CACHE_LOCK:
        cache = _TTL_CACHES.get(cache_name)
        if cache is None:
            cache = {}
            _TTL_CACHES[cache_name] = cache
        if max_items and len(cache) >= int(max_items):
            cache.clear()
        cache[key] = (exp, value)
    return value


def _risk_config_by_type(risk_type):
    for config in RISK_CONFIG.values():
        if risk_type in {
            config.get("name"),
            config.get("risk_type_label"),
            config.get("full_risk_type"),
        }:
            return config
    return None


def _interpolate_score_from_probability(probability, score_points):
    p = max(0.0, min(1.0, float(probability or 0.0)))
    points = sorted(score_points, key=lambda item: item[0])
    if p <= points[0][0]:
        return float(points[0][1])
    if p >= points[-1][0]:
        return float(points[-1][1])
    for (p0, s0), (p1, s1) in zip(points, points[1:]):
        if p0 <= p <= p1:
            ratio = (p - p0) / max(p1 - p0, 1e-9)
            return float(s0 + ratio * (s1 - s0))
    return float(points[-1][1])


def _measures_and_reason(risk_level, risk_type):
    config = _risk_config_by_type(risk_type)
    item = (config or {}).get("measures", {}).get(risk_level, {})
    return item.get("measures", []), item.get("reason", "")


def _read_csv_with_fallback(path, **kwargs):
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(path, **kwargs)


def _source_key(value):
    if value is None:
        return ""
    return str(value).strip()


def _source_candidates(shield_id=None, project_code=None):
    candidates = []
    for value in (
            shield_id,
            project_code,
    ):
        key = _source_key(value)
        if not key:
            continue
        candidates.append(key)
        alias = SOURCE_FILE_ALIASES.get(key)
        if alias:
            candidates.append(alias)
    deduped = []
    seen = set()
    for value in candidates:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _load_project_geo_ring_cache():
    global GEO_SOURCE_RING_CACHE
    if GEO_SOURCE_RING_CACHE is not None:
        return GEO_SOURCE_RING_CACHE
    cache = {}
    path = GEO_CLUSTER_CSV
    if not path or not os.path.isfile(path):
        GEO_SOURCE_RING_CACHE = cache
        return GEO_SOURCE_RING_CACHE
    try:
        df = _read_csv_with_fallback(path)
        if "Ring_Index" not in df.columns and "Ring.No" in df.columns:
            df = df.rename(columns={"Ring.No": "Ring_Index"})
        if "Source_File" not in df.columns or "Ring_Index" not in df.columns:
            GEO_SOURCE_RING_CACHE = cache
            return GEO_SOURCE_RING_CACHE
        keep = ["Source_File", "Ring_Index"]
        if "Cluster_Label" in df.columns:
            keep.append("Cluster_Label")
        keep.extend([feature for feature in GEO_FEATURES if feature in df.columns and feature not in keep])
        df = df[keep].copy()
        df["Ring_Index"] = pd.to_numeric(df["Ring_Index"], errors="coerce")
        df = df.dropna(subset=["Ring_Index"]).copy()
        df["Ring_Index"] = df["Ring_Index"].astype(int)
        for _, row in df.iterrows():
            source = _source_key(row.get("Source_File"))
            if not source:
                continue
            ring = int(row["Ring_Index"])
            item = {"Source_File": source, "Ring.No": ring}
            if "Cluster_Label" in row and not pd.isna(row.get("Cluster_Label")):
                try:
                    item["Cluster_Label"] = int(float(row.get("Cluster_Label")))
                except Exception:
                    item["Cluster_Label"] = row.get("Cluster_Label")
            for feature in GEO_FEATURES:
                if feature in row and not pd.isna(row.get(feature)):
                    item[feature] = row.get(feature)
            cache.setdefault(source, {})[ring] = item
    except Exception:
        traceback.print_exc()
    GEO_SOURCE_RING_CACHE = cache
    return GEO_SOURCE_RING_CACHE



def _default_project_geo(ring=None):
    item = {"Cluster_Label": 0}
    if ring is not None:
        try:
            item["Ring.No"] = int(float(ring))
        except Exception:
            pass
    for feature in GEO_FEATURES:
        item[feature] = 0.0
    return item


def _lookup_project_geo(ring, source_candidates):
    try:
        ring = int(float(ring))
    except Exception:
        return _default_project_geo()
    cache = _load_project_geo_ring_cache()
    for raw_source in source_candidates or []:
        source = _source_key(raw_source)
        if not source:
            continue
        to_try = [source]
        alias = SOURCE_FILE_ALIASES.get(source)
        if alias and alias != source:
            to_try.append(alias)
        for key in to_try:
            direct = cache.get(key, {}).get(ring)
            if direct:
                cl = direct.get("Cluster_Label")
                if cl is not None and not pd.isna(cl):
                    return direct
                return _default_project_geo(ring)
    return _default_project_geo(ring)


def _enrich_points_with_geo(points, source_candidates=None):
    if not isinstance(points, list):
        return points
    enriched = []
    geo_by_ring = {}
    for point in points:
        if not isinstance(point, dict):
            enriched.append(point)
            continue
        item = dict(point)
        ring_val = item.get("Ring.No", item.get("RING", item.get("ring")))
        try:
            ring = int(float(ring_val))
        except Exception:
            enriched.append(item)
            continue
        if ring not in geo_by_ring:
            geo_by_ring[ring] = _lookup_project_geo(ring, source_candidates)
        geo = geo_by_ring.get(ring) or _default_project_geo(ring)
        for feature, value in geo.items():
            if feature in ("Ring.No", "Source_File"):
                continue
            item[feature] = value
        enriched.append(item)
    return enriched


def _require_json(required_fields):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, jsonify({"status": "error", "error": "请求体必须为有效的 JSON 对象"}), 400
    for field, err_msg in required_fields:
        if not data.get(field):
            return None, jsonify({"status": "error", "error": err_msg}), 400
    return data, None, None



def _build_select_clause(fields):
    if not isinstance(fields, (list, tuple)):
        return "*"
    safe_fields = [field.strip() for field in fields if isinstance(field, str) and field.strip()]
    return ", ".join([_quote_influx_identifier(field) for field in safe_fields]) if safe_fields else "*"



def _risk_query_fields():
    cached = _ttl_cache_get("risk_level_result", "__risk_query_fields__")
    if cached:
        return list(cached)
    fields = set()
    for cfg in RISK_CONFIG.values():
        for field in cfg.get("fields", []) or []:
            fields.add(field)
    fields.add("Ring.No")
    out = sorted(fields)
    _ttl_cache_set("risk_level_result", "__risk_query_fields__", out, ttl_sec=3600.0, max_items=16)
    return out


def _risk_display_fields(risk_config_key):
    fields = RISK_DISPLAY_FIELDS.get(risk_config_key)
    if fields:
        return list(fields)
    return list(RISK_CONFIG.get(risk_config_key, {}).get("fields", []) or [])


def _risk_key_parameter_fields(risk_config_key):
    if risk_config_key == "tail_seal_risk":
        return list(TAIL_SEAL_KEY_PARAMETER_FIELDS)
    return _risk_display_fields(risk_config_key)


def _risk_display_impact(risk_config_key):
    cache_key = ("impact", risk_config_key)
    cached = _ttl_cache_get("risk_level_result", cache_key)
    if cached is not None:
        return cached
    config = RISK_CONFIG[risk_config_key]
    value = "，".join([config["map"].get(k, k) for k in _risk_display_fields(risk_config_key)])
    _ttl_cache_set("risk_level_result", cache_key, value, ttl_sec=3600.0, max_items=64)
    return value


def _dedupe_points_by_time(points):
    seen = set()
    unique_points = []
    for point in points or []:
        point_time = point.get("time")
        if point_time in seen:
            continue
        seen.add(point_time)
        unique_points.append(point)
    unique_points.sort(key=lambda x: x.get("time"))
    return unique_points


def _parse_ring_number(value):
    try:
        return float(value), None, None
    except (ValueError, TypeError):
        return None, jsonify({
            "status": "error",
            "message": f"环号参数格式错误: {value}"
        }), 400


def _normalize_limit(value, default, maximum):
    try:
        limit = int(value)
    except Exception:
        limit = default
    if limit <= 0:
        return default
    return min(limit, maximum)


def _find_earliest_warning_from_point_results(point_results, consecutive_n):
    try:
        consec = 0
        start_time = None
        start_params = None
        for r in point_results or []:
            try:
                level = r.get("risk_level")
                if level in ["低风险Ⅱ", "中风险Ⅲ", "高风险Ⅳ"]:
                    if consec == 0:
                        start_time = r.get("_time") or "-"
                        start_params = r.get("_params") or None
                    consec += 1
                    if consec >= consecutive_n:
                        return start_time, start_params
                else:
                    consec = 0
                    start_time = None
                    start_params = None
            except Exception:
                continue
    except Exception:
        pass
    return None, None



class RiskAssessor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, 'models')
        self.mud_cake_calculator = None
        self._init_mud_cake_calculator()

    def _init_mud_cake_calculator(self):
        stratum_root = os.path.join(self.model_dir, "mud_cake_by_stratum")
        if os.path.isdir(stratum_root):
            calculator = StratumMudCakeRiskCalculator(model_root=stratum_root)
            if calculator.calculators:
                self.mud_cake_calculator = calculator
                return
        self.mud_cake_calculator = None

    def _format_mud_cake_point(self, point):
        property_dict = {}
        try:
            feature_list = list(getattr(self.mud_cake_calculator, 'all_features', []) or [])
            if not feature_list:
                feature_list = list(TBM_FEATURES) + list(GEO_FEATURES) + ["Cluster_Label"]
        except Exception:
            feature_list = list(TBM_FEATURES) + list(GEO_FEATURES) + ["Cluster_Label"]

        for field in feature_list:
            if field in ('RING', 'state'):
                continue
            val = point.get(field)
            try:
                if val is not None and val != '' and not pd.isna(val):
                    property_dict[field] = float(val)
                else:
                    property_dict[field] = 0.0 if field in GEO_FEATURES or field == "Cluster_Label" else None
            except Exception:
                property_dict[field] = 0.0 if field in GEO_FEATURES or field == "Cluster_Label" else None
        if "Cluster_Label" not in property_dict:
            property_dict["Cluster_Label"] = 0.0

        formatted = {
            'property': property_dict,
            'state': point.get('state', 1),
            'timestamp': point.get('ts(Asia/Shanghai)', point.get('time', '')),
            'ring': point.get('Ring.No', point.get('RING', 0)),
            'Cluster_Label': property_dict.get("Cluster_Label"),
        }
        return formatted

    def assess_mud_cake_risk_sequence(self, ring_points):
        if self.mud_cake_calculator is None:
            return None
        points = [p for p in ring_points if isinstance(p, dict)]
        points.sort(key=lambda x: x.get('time'))
        if not points:
            return None

        ring_vals = []
        for p in points:
            rv = p.get("Ring.No", p.get("RING", p.get("ring")))
            try:
                ring_vals.append(int(float(rv)))
            except Exception:
                continue
        if len(set(ring_vals)) < MUD_CAKE_SEQUENCE_RING_COUNT:
            return None

        current_ring = max(ring_vals) if ring_vals else None
        current_points = []
        if current_ring is not None:
            for p in points:
                rv = p.get("Ring.No", p.get("RING", p.get("ring")))
                try:
                    if int(float(rv)) == int(current_ring):
                        current_points.append(p)
                except Exception:
                    continue
        if not current_points:
            return None
        if len(current_points) < 48:
            return None

        def _valid(v):
            try:
                return v not in (None, "", "-") and not pd.isna(v)
            except Exception:
                return v not in (None, "", "-")

        for f in TBM_FEATURES:
            if not any(_valid(p.get(f)) for p in current_points):
                return None

        cluster_label = 0
        for p in current_points:
            if _valid(p.get("Cluster_Label")):
                cluster_label = p.get("Cluster_Label")
                break

        sequence = [self._format_mud_cake_point(p) for p in points]
        if not sequence:
            return None

        try:
            result = self.mud_cake_calculator.calculate_ring_risk_sequence_for_label(sequence, cluster_label)
        except Exception as exc:
            raise RuntimeError(f"结泥饼模型调用异常: {exc}") from exc

        if not isinstance(result, dict):
            raise RuntimeError("结泥饼模型返回结果格式错误")
        if result.get('status') != 'success':
            raise RuntimeError(result.get('message') or "结泥饼模型评估失败")

        try:
            probability = float(result.get('combined_risk', 0.0))
        except Exception:
            probability = 0.0
        if probability < 0.0:
            probability = 0.0
        elif probability > 1.0:
            probability = 1.0
        probability = round(probability, 2)
        risk_level_direct = result.get('risk_level', 'no_risk')
        risk_level, measures, reason, potential_risk = self._get_model_level_and_measures(
            risk_level_direct,
            "结泥饼风险",
        )
        mapped_score = self.map_probability_to_score(probability, "结泥饼风险")
        return {
            "risk_type": "结泥饼风险",
            "risk_level": risk_level,
            "risk_level_model": risk_level_direct,
            "risk_score": round(mapped_score, 2),
            "probability": probability,
            "measures": measures,
            "reason": reason,
            "potential_risk": potential_risk,
            "earliest_time": result.get('earliest_time', ''),
            "stratum_label": result.get('stratum_label', cluster_label),
            "training_mode": result.get('training_mode', 'stratum_specific'),
        }

    def _normalize_model_risk_level(self, risk_level):
        if isinstance(risk_level, str):
            text = risk_level.strip()
            if text in {"high", "高风险", "高风险Ⅳ", "Ⅳ", "IV"}:
                return "高风险Ⅳ"
            if text in {"medium", "中风险", "中风险Ⅲ", "Ⅲ", "III"}:
                return "中风险Ⅲ"
            if text in {"low", "低风险", "低风险Ⅱ", "Ⅱ", "II"}:
                return "低风险Ⅱ"
            if text in {"no_risk", "none", "normal", "无风险", "无风险Ⅰ", "Ⅰ", "I"}:
                return "无风险Ⅰ"
        raise ValueError(f"模型返回了无法识别的风险等级: {risk_level}")

    def _get_model_level_and_measures(self, model_level, risk_type):
        risk_level = self._normalize_model_risk_level(model_level)
        config = _risk_config_by_type(risk_type) or {}
        potential_risk = (
            config.get("potential_risk", "系统预警")
            if risk_level not in {"无风险Ⅰ", "低风险Ⅱ"}
            else "-"
        )
        measures, reason = self._get_measures_and_reason(risk_level, risk_type)
        return risk_level, measures, reason, potential_risk

    def _get_measures_and_reason(self, risk_level, risk_type):
        return _measures_and_reason(risk_level, risk_type)

    def map_probability_to_score(self, probability, risk_type):
        try:
            config = _risk_config_by_type(risk_type)
            if config:
                return _interpolate_score_from_probability(probability, config["score_points"])
        except Exception:
            pass
        return float(probability or 0.0)

    def _assess_generic_raw(self, data, calc_func, risk_type, fault_cause):
        result = calc_func(data)
        probability = round(float(result.get('probability', 0) or 0), 2)
        risk_level, measures, reason, potential_risk = self._get_model_level_and_measures(
            result.get('risk_level'),
            risk_type,
        )
        mapped_score = self.map_probability_to_score(probability, risk_type)
        return {
            "risk_type": risk_type,
            "risk_level": risk_level,
            "risk_score": round(mapped_score, 2),
            "probability": probability,
            "measures": measures,
            "reason": reason,
            "details": result.get("details"),
            "potential_risk": potential_risk,
            "fault_cause": fault_cause,
        }

    def assess_single_point_risks(self, data_point):
        specs = (
            ("clog_risk", calculate_clog_risk),
            ("mdr_seal_risk", calculate_mdr_seal_risk),
            ("tail_seal_risk", calculate_tail_seal_risk),
        )
        results = []
        for risk_key, calc_func in specs:
            config = RISK_CONFIG[risk_key]
            try:
                results.append(self._assess_generic_raw(
                    data_point,
                    calc_func,
                    config["full_risk_type"],
                    config["fault_cause"],
                ))
            except Exception:
                continue
        return results

def _clean_json_val(val, dash_if_empty=True):
    try:
        if val is None:
            return "-" if dash_if_empty else None

        if isinstance(val, (list, dict)):
            # 如果已经是对象，则递归清理其中的元素或直接返回
            if isinstance(val, list):
                return "，".join([str(_clean_json_val(x, False)) for x in val])
            return val

        if isinstance(val, str):
            s = val.strip()
            if s == "" or s.lower() in ("null", "none"):
                return "-" if dash_if_empty else None

            # 处理 JSON 编码的字符串 (e.g., "\"text\"", "[1,2,3]", "{\"a\":1}")
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")) or \
                    s.startswith("[") or s.startswith("{"):
                try:
                    d = json.loads(s)
                    return _clean_json_val(d, dash_if_empty)
                except Exception:
                    # 如果解析失败，去掉引号或原样返回
                    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                        return s[1:-1]
                    return s
            return s

        return str(val)
    except Exception:
        return "-" if dash_if_empty else str(val)


risk_assessor = RiskAssessor()
bp = Blueprint('risk', __name__)



def _to_local_dt(time_str: str):
    # 支持 datetime/pandas.Timestamp
    if isinstance(time_str, datetime):
        dt = time_str
        try:
            # 若是UTC（tzinfo存在且偏移为0），加8小时到上海时区
            if dt.tzinfo is not None and dt.utcoffset() == timedelta(0):
                dt = dt + timedelta(hours=8)
        except Exception:
            pass
        return dt

    if pd is not None and isinstance(time_str, pd.Timestamp):
        try:
            dt = time_str.to_pydatetime()
            if dt.tzinfo is not None and dt.utcoffset() == timedelta(0):
                dt = dt + timedelta(hours=8)
            return dt
        except Exception:
            pass

    if not isinstance(time_str, str):
        return None
    s = time_str.strip()
    patterns = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]
    for fmt in patterns:
        try:
            dt = datetime.strptime(s, fmt)
            if "Z" in s or "GMT" in s:
                dt = dt + timedelta(hours=8)
            return dt
        except ValueError:
            continue
    return None


def format_time_utc_to_shanghai(time_str: str) -> str:
    try:
        local_dt = _to_local_dt(time_str)
        if local_dt is None:
            return time_str
        # 返回不补零的小时：YYYY/M/D H:MM:SS（分钟/秒补零以保证两位）
        return f"{local_dt.year}/{local_dt.month}/{local_dt.day} {local_dt.hour}:{local_dt.minute:02d}:{local_dt.second:02d}"
    except Exception:
        return time_str


def _round_value_2(v):
    try:
        if isinstance(v, (int, float)):
            return round(v, 2)
        if isinstance(v, str):
            return round(float(v), 2)
    except Exception:
        return v
    return v


def round_params(params):
    if not isinstance(params, dict):
        return params
    return {k: _round_value_2(v) for k, v in params.items()}


def map_safety_to_level(safety_level: str) -> str:
    try:
        if isinstance(safety_level, str):
            for sym in ("Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ"):
                if sym in safety_level:
                    return sym
    except Exception:
        pass
    return "-"


def _risk_priority(level: str) -> int:
    try:
        if isinstance(level, str):
            if ("Ⅳ" in level) or ("IV" in level) or ("高风险" in level):
                return 4
            if ("Ⅲ" in level) or ("III" in level) or ("中风险" in level):
                return 3
            if ("Ⅱ" in level) or ("II" in level) or ("低风险" in level):
                return 2
            if ("Ⅰ" in level) or ("无风险" in level):
                return 1
    except Exception:
        pass
    return 0


def _aggregate_with_consecutive(lst, risk_type: str, n: int):
    if not lst:
        return {}

    priorities = []
    for r in lst:
        try:
            priorities.append(_risk_priority(r.get('risk_level')))
        except Exception:
            priorities.append(0)
    valid_pairs = [(item, pri) for item, pri in zip(lst, priorities) if pri > 0]
    if not valid_pairs:
        return {}
    lst = [item for item, _ in valid_pairs]
    priorities = [pri for _, pri in valid_pairs]

    def _probability(item):
        try:
            return float(item.get('probability', 0) or 0)
        except Exception:
            return 0.0

    def _set_final_item(items, risk_level):
        max_prob = round(max((_probability(item) for item in items), default=0.0), 2)
        chosen = max(items, key=_probability) if items else lst[-1]
        final_item = dict(chosen)
        final_item['risk_level'] = risk_level
        final_item['probability'] = max_prob
        try:
            measures, reason = risk_assessor._get_measures_and_reason(risk_level, risk_type)
            final_item['measures'] = measures
            final_item['reason'] = reason
            config = _risk_config_by_type(risk_type) or {}
            final_item['potential_risk'] = (
                config.get("potential_risk", "系统预警")
                if _risk_priority(risk_level) > 2
                else "-"
            )
        except Exception:
            pass
        try:
            final_item['risk_score'] = round(
                risk_assessor.map_probability_to_score(max_prob, risk_type),
                2,
            )
        except Exception:
            pass
        if _risk_priority(risk_level) <= 2:
            final_item['potential_risk'] = "-"
        return final_item

    def find_segment(min_priority):
        count = 0
        start = None
        for i, pri in enumerate(priorities):
            if pri >= min_priority:
                if count == 0:
                    start = i
                count += 1
                if count >= n:
                    return start, i
            else:
                count = 0
                start = None
        return None, None

    hi_start, _ = find_segment(4)
    if hi_start is not None:
        j = hi_start
        while j < len(priorities) and priorities[j] >= 4:
            j += 1
        return _set_final_item(lst[hi_start:j], "高风险Ⅳ")

    mid_flags = [1 if pri >= 3 else 0 for pri in priorities]
    count = 0
    mid_start = None
    for i, flag in enumerate(mid_flags):
        if flag:
            if count == 0:
                mid_start = i
            count += 1
            if count >= n:
                break
        else:
            count = 0
            mid_start = None

    if mid_start is not None and count >= n:
        j = mid_start
        while j < len(mid_flags) and mid_flags[j] == 1:
            j += 1
        return _set_final_item(lst[mid_start:j], "中风险Ⅲ")

    low_items = [item for item, pri in zip(lst, priorities) if pri == 2]
    if low_items:
        return _set_final_item(low_items, "低风险Ⅱ")

    no_risk_items = [item for item, pri in zip(lst, priorities) if pri == 1]
    final_items = no_risk_items if no_risk_items else lst
    final_item = _set_final_item(final_items, "无风险Ⅰ")
    return final_item


def aggregate_ring_risk_results(
        original_ring_data,
        current_ring_number=None,
        multi_ring_data=None,
        measurement=None,
        source_candidates=None,
):
    if not isinstance(original_ring_data, list):
        original_ring_data = [original_ring_data] if original_ring_data else []

    risk_by_type = {}
    risk_type_meta = {
        "滞排风险": ("clog_risk", _risk_display_fields("clog_risk")),
        "主驱动密封失效风险": ("mdr_seal_risk", _risk_display_fields("mdr_seal_risk")),
        "盾尾密封失效风险": ("tail_seal_risk", _risk_display_fields("tail_seal_risk")),
    }

    for point in original_ring_data:
        if not isinstance(point, dict):
            continue
        point_results = risk_assessor.assess_single_point_risks(point)
        for r in point_results:
            rt = r.get('risk_type')
            if rt and rt != "结泥饼风险":
                cfg_key, fields = risk_type_meta.get(rt, (None, None))
                rr = dict(r)
                rr["_time"] = point.get("time")
                rr["_params"] = {k: point.get(k) for k in fields} if fields else None
                risk_by_type.setdefault(rt, []).append(rr)

    final_results = []
    for rt, lst in risk_by_type.items():
        if not lst:
            continue
        final_item = _aggregate_with_consecutive(lst, rt, CONSECUTIVE_TRIGGER_N)
        final_results.append(final_item)

    # 结泥饼序列风险评估
    if multi_ring_data is None:
        if current_ring_number is not None:
            multi_ring_data = query_consecutive_ring_data(
                current_ring_number,
                count=MUD_CAKE_SEQUENCE_RING_COUNT,
                measurement=measurement,
            )
        else:
            multi_ring_data = original_ring_data
    if not original_ring_data:
        return final_results, risk_by_type

    mud_cake_points = multi_ring_data
    if isinstance(mud_cake_points, list):
        mud_cake_points = _enrich_points_with_geo(mud_cake_points, source_candidates=source_candidates)
    seq_risk = risk_assessor.assess_mud_cake_risk_sequence(mud_cake_points)
    if isinstance(seq_risk, dict) and seq_risk.get("risk_type") == "结泥饼风险":
        final_results.append(seq_risk)

    return final_results, risk_by_type



def query_ring_data(ring_number, measurement=None, fields=None, limit=None):
    try:
        m = _quote_influx_identifier(measurement)
        ring_int_val = int(float(ring_number))
        select_clause = _build_select_clause(fields)
        limit_clause = ""
        try:
            if limit is not None:
                limit_int = int(limit)
                if limit_int > 0:
                    limit_clause = f" LIMIT {limit_int}"
        except Exception:
            limit_clause = ""

        query_range = f"""
        SELECT {select_clause} FROM {m} 
        WHERE "Ring.No" >= {ring_int_val} AND "Ring.No" < {ring_int_val + 1} 
        ORDER BY time ASC{limit_clause}
        """
        res = client.query(query_range)
        points = list(res.get_points()) if res is not None else []
        if points:
            return _dedupe_points_by_time(points)
        return None

    except Exception as e:
        traceback.print_exc()
        return None


def query_ring_range_data(start_ring_inclusive, end_ring_exclusive, measurement=None, fields=None):
    try:
        m = _quote_influx_identifier(measurement)
        start_int = int(float(start_ring_inclusive))
        end_int = int(float(end_ring_exclusive))
        select_clause = _build_select_clause(fields)

        q = f"""
        SELECT {select_clause} FROM {m}
        WHERE "Ring.No" >= {start_int} AND "Ring.No" < {end_int}
        ORDER BY time ASC
        """
        res = client.query(q)
        points = list(res.get_points()) if res is not None else []
        if not points:
            return []
        return _dedupe_points_by_time(points)
    except Exception:
        return []


def _group_points_by_ring(points):
    ring_map = {}
    for p in points or []:
        if not isinstance(p, dict):
            continue
        r = p.get("Ring.No", p.get("RING"))
        try:
            ring_int = int(float(r))
        except Exception:
            continue
        ring_map.setdefault(ring_int, []).append(p)
    return ring_map


def _evenly_sample_points(items, sample_size):
    total = len(items or [])
    if total <= sample_size:
        return list(items or [])
    if sample_size <= 1:
        return [items[-1]]
    last_index = total - 1
    return [
        items[int(i * last_index / (sample_size - 1))]
        for i in range(sample_size)
    ]


def collect_key_parameters(center_ring, fields, count=6, preload=None):
    fields_list = list(fields or [])
    if not isinstance(preload, dict):
        return []

    try:
        center = int(float(center_ring))
    except Exception:
        return []

    try:
        target_count = int(count)
    except Exception:
        target_count = 6

    ring_keys = []
    for r, pts in preload.items():
        if not pts:
            continue
        try:
            ring_int = int(float(r))
        except Exception:
            continue
        if ring_int <= center:
            ring_keys.append(ring_int)
    ring_keys = sorted(set(ring_keys))[-target_count:]

    merged_points = []
    per_ring_sample = 300
    for r in ring_keys:
        pts = preload.get(r) or []
        ring_points = []
        for p in pts:
            raw_time = p.get("time") or ""
            item = {"time": format_time_utc_to_shanghai(raw_time)}
            for f in fields_list:
                item[f] = _round_value_2(p.get(f))
            ring_points.append((raw_time, item))
        ring_points.sort(key=lambda x: x[0])
        merged_points.extend(_evenly_sample_points(ring_points, per_ring_sample))

    merged_points.sort(key=lambda x: x[0])

    result = []
    for _, item in merged_points:
        result.append(item)

    return result


@bp.route('/getRiskLevel', methods=['POST'])
def get_risk_level():
    try:
        # 使用 silent=True，避免无效 JSON 直接抛出 BadRequest 导致 500
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "请求体必须为有效的 JSON 对象"
            }), 400

        ring_number = data.get('RING')
        if ring_number is None:
            return jsonify({
                "status": "error",
                "message": "未提供环号参数"
            }), 400

        ring_number, parse_err_resp, parse_err_code = _parse_ring_number(ring_number)
        if parse_err_resp is not None:
            return parse_err_resp, parse_err_code

        shield_id = data.get('shield_id')
        try:
            measurement, project_code_request = _resolve_shield_context(shield_id)
        except InvalidShieldIdError as exc:
            return jsonify({"status": "error", "error": str(exc)}), 400
        ring_int_cache = int(float(ring_number))
        cache_ttl = float(os.environ.get("GETRISKLEVEL_CACHE_TTL_SEC", "10") or 10)
        cache_key = (str(shield_id).strip() if shield_id else "", ring_int_cache)
        if cache_ttl > 0:
            cached = _ttl_cache_get("risk_level_result", cache_key)
            if isinstance(cached, dict):
                return jsonify(cached)
        source_candidates = _source_candidates(
            shield_id=shield_id,
            project_code=project_code_request,
        )

        center_int = ring_int_cache
        ring_window_count = MUD_CAKE_SEQUENCE_RING_COUNT

        required_fields = _risk_query_fields()

        multi_ring_data = query_consecutive_ring_data(
            ring_number,
            count=ring_window_count,
            measurement=measurement,
            fields=required_fields,
        ) or []

        ring_points_map = _group_points_by_ring(multi_ring_data)
        original_ring_data = ring_points_map.get(center_int) or []
        if not original_ring_data:
            ring_data = query_ring_data(ring_number, measurement=measurement, fields=required_fields)
            original_ring_data = ring_data if isinstance(ring_data, list) else ([ring_data] if ring_data else [])
            if original_ring_data:
                ring_points_map[center_int] = original_ring_data
                if isinstance(multi_ring_data, list):
                    multi_ring_data.extend(original_ring_data)

        preload_map = {}
        selected_rings = sorted(r for r, pts in ring_points_map.items() if pts and r <= center_int)[-ring_window_count:]
        for r in selected_rings:
            preload_map[r] = ring_points_map.get(r) or []
        preload_map[center_int] = original_ring_data

        all_key_fields = []
        seen_key_fields = set()
        for key in ("mud_cake_risk", "clog_risk", "mdr_seal_risk", "tail_seal_risk"):
            for field in _risk_key_parameter_fields(key):
                if field not in seen_key_fields:
                    seen_key_fields.add(field)
                    all_key_fields.append(field)
        all_key_parameters = collect_key_parameters(
            ring_number,
            all_key_fields,
            count=ring_window_count,
            preload=preload_map,
        )

        risk_results, point_results_by_type = aggregate_ring_risk_results(
            original_ring_data,
            current_ring_number=ring_number,
            multi_ring_data=multi_ring_data,
            measurement=measurement,
            source_candidates=source_candidates,
        )
        result = {
            "ring": int(ring_number),
            "project_code": project_code_request,
            "mud_cake_risk": {},
            "clog_risk": {},
            "mdr_seal_risk": {},
            "tail_seal_risk": {}
        }

        def _build_risk_item(risk_obj, risk_config_key):
            spec = RISK_OUTPUT_SPECS[risk_config_key]
            config = RISK_CONFIG[risk_config_key]
            fm = risk_obj.get("measures")
            fm = _clean_json_val(fm)
            display_fields = _risk_display_fields(risk_config_key)
            key_parameter_fields = _risk_key_parameter_fields(risk_config_key)
            imp = _risk_display_impact(risk_config_key)

            try:
                probability = float(risk_obj.get("probability") or 0.0)
            except Exception:
                probability = 0.0
            if probability < 0.0:
                probability = 0.0
            elif probability > 1.0:
                probability = 1.0
            probability = round(probability, 2)
            try:
                risk_score = round(float(risk_obj.get("risk_score")), 2)
            except Exception:
                risk_score = round(risk_assessor.map_probability_to_score(probability, spec["full_risk_type"]), 2)

            risk_out = {
                "risk_type": spec["risk_type_label"],
                "ring": int(result["ring"]),
                "project_code": project_code_request,
                "fault_measures": fm,
                "fault_reason": risk_obj.get("reason"),
                "fault_reason_analysis": get_fault_reason_analysis(spec["risk_type_label"], risk_obj["risk_level"]),
                "fault_cause": spec["fault_cause"],
                "impact_parameters": imp,
                "safety_level": risk_obj["risk_level"],
                "risk_level": map_safety_to_level(risk_obj["risk_level"]),
                "risk_score": risk_score,
                "probability": probability,
                "potential_risk": risk_obj["potential_risk"],
                "warning_time": "-",
                "warning_parameters": "-",
            }
            earliest_time_raw = "-"
            earliest_params = None
            if risk_config_key == "mud_cake_risk":
                earliest_time_raw = risk_obj.get("earliest_time") or "-"
            else:
                earliest_time_raw, earliest_params = _find_earliest_warning_from_point_results(
                    point_results_by_type.get(spec["full_risk_type"]),
                    CONSECUTIVE_TRIGGER_N,
                )

            if earliest_params:
                earliest_params = round_params(earliest_params)

            if earliest_time_raw and earliest_time_raw != "-":
                final_level_text = risk_obj.get("risk_level", "")
                should_warn = final_level_text in WARNING_LEVELS
                if risk_config_key != "mud_cake_risk" or should_warn:
                    risk_out["warning_time"] = format_time_utc_to_shanghai(earliest_time_raw)
            elif risk_config_key == "mud_cake_risk" and original_ring_data:
                t0 = original_ring_data[0].get("time")
                if t0:
                    risk_out["warning_time"] = format_time_utc_to_shanghai(t0)

            if earliest_params:
                wp_raw = {config["map"].get(k, k): earliest_params.get(k) for k in display_fields if
                          k in earliest_params}
                risk_out["warning_parameters"] = _append_units_to_map(wp_raw, config["units"])
            elif risk_config_key == "mud_cake_risk" and original_ring_data:
                first_p = original_ring_data[0]
                ep = {f: first_p.get(f) for f in display_fields}
                wp_raw = {config["map"].get(k, k): ep.get(k) for k in ep}
                risk_out["warning_parameters"] = _append_units_to_map(wp_raw, config["units"])
            else:
                risk_out["warning_parameters"] = "-"

            selected_key_parameters = [
                {"time": item.get("time"), **{field: item.get(field) for field in key_parameter_fields}}
                for item in all_key_parameters
            ]
            risk_out["key_parameters"] = _rename_keys(selected_key_parameters, config["map"])
            risk_out["key_parameters"] = _with_value_and_unit(risk_out["key_parameters"], config["units"])

            return risk_out

        for risk in risk_results:
            risk_type = risk['risk_type']

            if risk_type == "结泥饼风险":
                result["mud_cake_risk"] = _build_risk_item(risk, "mud_cake_risk")

            elif risk_type == "滞排风险":
                result["clog_risk"] = _build_risk_item(risk, "clog_risk")

            elif risk_type == "主驱动密封失效风险":
                result["mdr_seal_risk"] = _build_risk_item(risk, "mdr_seal_risk")

            elif risk_type == "盾尾密封失效风险":
                result["tail_seal_risk"] = _build_risk_item(risk, "tail_seal_risk")

        if cache_ttl > 0:
            _ttl_cache_set("risk_level_result", cache_key, result, ttl_sec=cache_ttl, max_items=2048)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": f"获取数据时发生错误: {str(e)}"
        }), 500


class _MySQLConnectionPool:
    def __init__(self, host, port, user, password, database, charset, cursorclass, maxsize=5, max_idle=0):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._charset = charset
        self._cursorclass = cursorclass
        self._maxsize = max(1, int(maxsize or 1))
        self._max_idle = max(0, min(int(max_idle or 0), self._maxsize))
        self._pool = queue.Queue(maxsize=self._maxsize)
        self._created = 0
        self._lock = threading.Lock()

    def _create_conn(self):
        read_timeout = Config.MYSQL_READ_TIMEOUT
        write_timeout = Config.MYSQL_WRITE_TIMEOUT
        return pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset=self._charset,
            cursorclass=self._cursorclass,
            autocommit=True,
            connect_timeout=10,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

    def _discard(self, conn):
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        with self._lock:
            if self._created > 0:
                self._created -= 1

    def _reset_for_reuse(self, conn):
        try:
            conn.ping(reconnect=False)
            try:
                conn.rollback()
            except Exception:
                pass
            return True
        except Exception:
            self._discard(conn)
            return False

    def acquire(self):
        while True:
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                break
            if self._reset_for_reuse(conn):
                return conn

        with self._lock:
            if self._created < self._maxsize:
                conn = self._create_conn()
                self._created += 1
                return conn

        wait_timeout = float(os.environ.get("MYSQL_POOL_WAIT", "5"))
        deadline = time.monotonic() + wait_timeout
        while True:
            with self._lock:
                if self._created < self._maxsize:
                    conn = self._create_conn()
                    self._created += 1
                    return conn
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("MySQL连接池等待超时")
            try:
                conn = self._pool.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if self._reset_for_reuse(conn):
                return conn

    def release(self, conn):
        if conn is None:
            return
        if not self._reset_for_reuse(conn):
            return
        if self._pool.qsize() >= self._max_idle:
            self._discard(conn)
            return
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            self._discard(conn)
        except Exception:
            self._discard(conn)

    def discard(self, conn):
        self._discard(conn)

    def close_all(self):
        while True:
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                return
            self._discard(conn)


class _PooledConnection:
    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._pool.release(self._conn)
        self._conn = None

    def __exit__(self, exc_type, exc, tb):
        if self._closed:
            return False
        self._closed = True
        if exc_type is not None:
            self._pool.discard(self._conn)
        else:
            self._pool.release(self._conn)
        self._conn = None
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_MYSQL_POOL = None
_MYSQL_POOL_LOCK = threading.Lock()


def _mysql_connect():
    global _MYSQL_POOL
    try:
        if _MYSQL_POOL is None:
            with _MYSQL_POOL_LOCK:
                if _MYSQL_POOL is None:
                    maxsize = _env_int("MYSQL_POOL_SIZE", 5)
                    _MYSQL_POOL = _MySQLConnectionPool(
                        host=Config.MYSQL_HOST,
                        port=Config.MYSQL_PORT,
                        user=Config.MYSQL_USER,
                        password=Config.MYSQL_PASSWORD,
                        database=Config.MYSQL_DATABASE,
                        charset="utf8",
                        cursorclass=pymysql.cursors.DictCursor,
                        maxsize=maxsize,
                        max_idle=_env_int("MYSQL_POOL_MAX_IDLE", 0),
                    )
        raw = _MYSQL_POOL.acquire()
        _set_session_timeout(raw)  # 每次获取连接时设置会话超时
        return _PooledConnection(_MYSQL_POOL, raw)
    except Exception as e:
        raise RuntimeError(f"MySQL连接失败: {e}")


def _set_session_timeout(conn):
    try:
        max_ms = Config.MYSQL_MAX_EXECUTION_MS
        with conn.cursor() as cur:
            cur.execute("SET SESSION MAX_EXECUTION_TIME=%s", (max_ms,))
    except Exception:
        pass


def _normalize_safety_level(level):
    try:
        if not level:
            return "无风险Ⅰ"
        s = str(level)
        if any(sym in s for sym in ("Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ")):
            return s
        if "高风险" in s:
            return "高风险Ⅳ"
        if "中风险" in s:
            return "中风险Ⅲ"
        if "低风险" in s:
            return "低风险Ⅱ"
        if "无风险" in s:
            return "无风险Ⅰ"
        return "无风险Ⅰ"
    except Exception:
        return "无风险Ⅰ"


def get_fault_reason_analysis(risk_type: str, risk_level: str) -> str:
    try:
        lvl = _normalize_safety_level(risk_level)
        config = _risk_config_by_type(risk_type) or {}
        mapping = config.get("fault_reason_analysis", {})
        if mapping:
            return mapping.get(lvl, "当前等级暂无更详细分析")
        return "当前等级暂无更详细分析"
    except Exception:
        return "当前等级暂无更详细分析"


def _fetch_latest_row(conn, table, project_code=None):
    _set_session_timeout(conn)
    table_sql = _quote_mysql_identifier(table)
    with conn.cursor() as cur:
        ttl_latest = float(os.environ.get("LATEST_ROW_CACHE_TTL_SEC", "2") or 2)
        latest_key = (str(table), str(project_code).strip() if project_code is not None else "")
        cached_row = _ttl_cache_get("mysql_latest_row", latest_key)
        if cached_row is not None:
            return cached_row

        cache_key = ("cols", str(table))
        existing_cols = _ttl_cache_get("mysql_table_columns", cache_key)
        if not existing_cols:
            try:
                cur.execute(f"SHOW COLUMNS FROM {table_sql}")
                rows = cur.fetchall() or []
                existing_cols = {row.get("Field") or row.get("field") for row in rows}
                existing_cols.discard(None)
                _ttl_cache_set("mysql_table_columns", cache_key, existing_cols, ttl_sec=600.0, max_items=256)
            except Exception:
                existing_cols = set()

        where_sql = ""
        params = ()
        if (
                project_code is not None
                and str(project_code).strip() != ""
                and "project_code" in existing_cols
        ):
            where_sql = " WHERE `project_code`=%s"
            params = (str(project_code).strip(),)

        desired_cols = [c for c in LATEST_RISK_ROW_COLUMNS if c in existing_cols]
        if not desired_cols:
            desired_cols = ["ring", "project_code", "risk_level", "risk_score"]
            desired_cols = [c for c in desired_cols if c in existing_cols] or ["ring"]
        column_sql = ", ".join([f"`{column}`" for column in desired_cols])

        try:
            if "id" in existing_cols:
                cur.execute(f"SELECT {column_sql} FROM {table_sql}{where_sql} ORDER BY `id` DESC LIMIT 1", params)
            else:
                raise RuntimeError("missing id")
            row = cur.fetchone()
            if row:
                return row
        except Exception:
            pass

        cur.execute(f"SELECT {column_sql} FROM {table_sql}{where_sql} ORDER BY `ring` DESC LIMIT 1", params)
        row = cur.fetchone()
        if row is not None:
            _ttl_cache_set("mysql_latest_row", latest_key, row, ttl_sec=ttl_latest, max_items=512)
        return row


@bp.route('/getLatestRiskLevel', methods=['POST'])
def get_latest_risk_level():
    try:
        data, err_resp, err_code = _require_json(
            [
                ("shield_id", "缺少 shield_id 参数"),
                ("risk_type", "缺少 risk_type 参数"),
            ]
        )
        if err_resp is not None:
            return err_resp, err_code
        shield_id = data.get("shield_id")
        risk_type_param = data.get("risk_type")

        rp = str(risk_type_param).strip()
        table = RISK_TABLES.get(rp)
        if not table:
            return jsonify({"status": "error", "error": "不支持的风险类型", "supported": list(RISK_TABLES.keys())}), 400

        try:
            _, project_code_request = _resolve_shield_context(shield_id)
        except InvalidShieldIdError as exc:
            return jsonify({"status": "error", "error": str(exc)}), 400

        with _mysql_connect() as conn:
            row = _fetch_latest_row(conn, table, project_code=project_code_request)
        if not row:
            return jsonify({"status": "error", "error": "MySQL未找到指定风险的最新环数据"}), 404

        def build_risk_out(row, risk_type_label):
            safety = row.get("safety_level")
            if safety is None or str(safety).strip() == "":
                safety = row.get("risk_level")
            ring_val = row.get("ring")
            try:
                ring_num = int(float(ring_val)) if ring_val is not None else "-"
            except Exception:
                ring_num = _clean_json_val(ring_val)

            fm = next((row.get(k) for k in ("fault_measures", "measures") if row.get(k) not in (None, "", "-")), None)
            fm = _clean_json_val(fm)
            ip = _clean_json_val(row.get("impact_parameters"))
            kp = _clean_json_val(row.get("key_parameters"), dash_if_empty=True)
            try:
                risk_score = round(float(row.get("risk_score")), 2)
            except Exception:
                risk_score = "-"
            potential_risk = row.get("potential_risk")
            if potential_risk in (None, "", "-"):
                potential_risk = "-"

            return {
                "risk_type": risk_type_label,
                "ring": ring_num,
                "project_code": _clean_json_val(row.get("project_code") or project_code_request),
                "key_parameters": kp,
                "safety_level": _clean_json_val(safety),
                "risk_score": risk_score,
                "potential_risk": _clean_json_val(potential_risk),
                "fault_reason_analysis": _clean_json_val(row.get("fault_reason_analysis")),
                "fault_cause": _clean_json_val(row.get("fault_cause")),
                "fault_measures": fm,
                "impact_parameters": ip,
            }

        return jsonify(build_risk_out(row, rp))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"获取最新环数据时发生错误: {str(e)}"}), 500


@bp.route('/getLatestRiskLevelSimple', methods=['GET'])
def get_latest_risk_level_simple():
    try:
        result = {}

        with _mysql_connect() as conn:
            for k, t in RISK_TABLES.items():
                # k 是中文风险类型 (例如 "滞排")
                # t 是对应的表名
                try:
                    row = _fetch_latest_row(conn, t)
                    if row:
                        level_raw = row.get("risk_level")
                        if level_raw is None or str(level_raw).strip() == "":
                            level_out = "-"
                        else:
                            level_out = level_raw

                        ring_val = row.get("ring")
                        try:
                            ring_num = int(float(ring_val)) if ring_val is not None else "-"
                        except Exception:
                            ring_num = "-" if ring_val is None else str(ring_val)

                        result[k] = {
                            "risk_level": level_out,
                            "ring": ring_num
                        }
                except Exception:
                    # 单个风险获取失败不应阻塞其他风险
                    continue

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"获取最新风险等级简报时发生错误: {str(e)}"}), 500


def query_consecutive_ring_data(center_ring, count=6, measurement=None, fields=None):
    try:
        ring_int = int(float(center_ring))
        target_count = int(count)
        if target_count <= 1:
            return query_ring_data(ring_int, measurement=measurement, fields=fields) or []

        attempt = 0
        combined = []
        while True:
            window = target_count * (2 ** attempt)
            start_ring = max(0, ring_int - (window - 1))
            points = query_ring_range_data(
                start_ring,
                ring_int + 1,
                measurement=measurement,
                fields=fields,
            )
            ring_map = _group_points_by_ring(points)
            selected_rings = sorted([r for r, pts in ring_map.items() if pts and r <= ring_int])[-target_count:]
            if selected_rings:
                combined = []
                selected_set = set(selected_rings)
                for p in points:
                    try:
                        pr = int(float(p.get("Ring.No", p.get("RING"))))
                    except Exception:
                        continue
                    if pr in selected_set:
                        combined.append(p)
            if len(selected_rings) >= target_count or start_ring <= 0:
                break
            attempt += 1
            if attempt > 30:
                break

        if not combined:
            return []
        seen = set()
        unique_points = []
        for p in combined:
            t = p.get('time')
            if t not in seen:
                seen.add(t)
                unique_points.append(p)
        unique_points.sort(key=lambda x: x.get('time'))
        return unique_points
    except Exception:
        return []


def _rename_keys(items, mapping):
    try:
        renamed = []
        for it in items or []:
            if not isinstance(it, dict):
                renamed.append(it)
                continue
            new_it = {}
            for k, v in it.items():
                mk = mapping.get(k, k)
                new_it[mk] = v
            renamed.append(new_it)
        return renamed
    except Exception:
        return items


def _with_value_and_unit(items, unit_map):
    try:
        out = []
        for it in items or []:
            if not isinstance(it, dict):
                out.append(it)
                continue
            ni = {}
            for k, v in it.items():
                if k == "time":
                    ni["time"] = v
                else:
                    u = unit_map.get(k, "-")
                    ni[k] = {"value": v, "unit": u}
            out.append(ni)
        return out
    except Exception:
        return items


def _append_units_to_map(data_map, units_map):
    try:
        out = {}
        for k, v in (data_map or {}).items():
            u = units_map.get(k, "")
            if v is None or (isinstance(v, str) and v.strip() in ("", "null", "none", "-")):
                out[k] = "-"
            else:
                out[k] = f"{v}{u}" if u else str(v)
        return out
    except Exception:
        return data_map


@bp.route('/getAllRiskRecords', methods=['POST'])
def get_all_risk_records():
    try:
        data, err_resp, err_code = _require_json(
            [
                ("shield_id", "缺少 shield_id 参数"),
                ("risk_type", "缺少 risk_type 参数"),
            ]
        )
        if err_resp is not None:
            return err_resp, err_code
        shield_id = data.get("shield_id")
        risk_type_param = data.get("risk_type")

        rp = str(risk_type_param).strip()
        table = RISK_TABLES.get(rp)
        if not table:
            return jsonify({"status": "error", "error": "不支持的风险类型", "supported": list(RISK_TABLES.keys())}), 400

        try:
            _, project_code_request = _resolve_shield_context(shield_id)
        except InvalidShieldIdError as exc:
            return jsonify({"status": "error", "error": str(exc)}), 400
        limit_env = _env_int("RISK_RECORDS_LIMIT", 1000)
        max_limit_env = _env_int("RISK_RECORDS_LIMIT_MAX", 5000)
        limit_arg = _normalize_limit(data.get("limit", limit_env), limit_env, max_limit_env)
        records = []
        high_count = 0
        mid_count = 0
        low_count = 0
        table_sql = _quote_mysql_identifier(table)
        with _mysql_connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"SELECT `ring`,`warning_time`,`warning_parameters`,`fault_reason`,`fault_measures`,`risk_level` "
                        f"FROM {table_sql} WHERE `project_code`=%s ORDER BY `id` DESC LIMIT %s",
                        (project_code_request, limit_arg),
                    )
                    rows = cur.fetchall() or []
                except Exception:
                    cur.execute(
                        f"SELECT `ring`,`warning_time`,`warning_parameters`,`fault_reason`,`fault_measures`,`risk_level` "
                        f"FROM {table_sql} WHERE `project_code`=%s ORDER BY `ring` DESC LIMIT %s",
                        (project_code_request, limit_arg),
                    )
                    rows = cur.fetchall() or []

                for row in rows:
                    ring_val = row.get("ring")
                    try:
                        ring_num = int(float(ring_val)) if ring_val is not None else "-"
                    except Exception:
                        ring_num = _clean_json_val(ring_val)

                    wp = _clean_json_val(row.get("warning_parameters"))
                    fm = _clean_json_val(row.get("fault_measures"))

                    level_raw = row.get("risk_level")
                    pri = _risk_priority(str(level_raw) if level_raw is not None else "")
                    if pri < 2:
                        continue
                    if pri == 4:
                        high_count += 1
                    elif pri == 3:
                        mid_count += 1
                    elif pri == 2:
                        low_count += 1
                    level_out = _clean_json_val(level_raw)
                    fault_reason_out = _clean_json_val(row.get("fault_reason"))
                    warning_time_out = _clean_json_val(row.get("warning_time"))
                    rec = {
                        "risk_type": rp,
                        "project_code": project_code_request,
                        "warning_time": warning_time_out,
                        "ring": ring_num,
                        "warning_parameters": wp,
                        "risk_level": level_out,
                        "fault_reason": fault_reason_out,
                        "fault_measures": fm,
                    }
                    records.append(rec)

        def _ring_num(rec):
            r = rec.get("ring")
            try:
                return int(float(r))
            except Exception:
                return -1

        records.sort(key=lambda rec: _ring_num(rec), reverse=True)

        return jsonify({
            "risk_type": rp,
            "project_code": project_code_request,
            "records": records,
            "high_risk_count": high_count,
            "mid_risk_count": mid_count,
            "low_risk_count": low_count,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"获取全部风险记录时发生错误: {str(e)}"}), 500


_POINT_INFO_POOL = None
_POINT_INFO_POOL_LOCK = threading.Lock()


def _get_point_info_connection():
    global _POINT_INFO_POOL
    if _POINT_INFO_POOL is None:
        with _POINT_INFO_POOL_LOCK:
            if _POINT_INFO_POOL is None:
                maxsize = _env_int("POINT_INFO_POOL_SIZE", 5)
                _POINT_INFO_POOL = _MySQLConnectionPool(
                    host=Config.POINT_INFO_MYSQL_HOST,
                    port=Config.POINT_INFO_MYSQL_PORT,
                    user=Config.POINT_INFO_MYSQL_USER,
                    password=Config.POINT_INFO_MYSQL_PASSWORD,
                    database=Config.POINT_INFO_MYSQL_DATABASE,
                    charset="utf8",
                    cursorclass=pymysql.cursors.DictCursor,
                    maxsize=maxsize,
                    max_idle=_env_int("POINT_INFO_POOL_MAX_IDLE", 0),
                )
    raw = _POINT_INFO_POOL.acquire()
    _set_session_timeout(raw)
    return _PooledConnection(_POINT_INFO_POOL, raw)


def _close_mysql_pools():
    for pool in (_MYSQL_POOL, _POINT_INFO_POOL):
        if pool is not None:
            pool.close_all()


atexit.register(_close_mysql_pools)

_POINT_INFO_EXCLUDED_TABLES = {"point_info_sum", "point_info_sum_old"}


def _list_point_info_tables(conn):
    cached = _ttl_cache_get("point_info_tables", "__tables__")
    if cached:
        return list(cached)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=%s AND table_name LIKE 'point_info\\_%%' ESCAPE '\\\\' "
            "ORDER BY table_name",
            ("point_info",),
        )
        rows = cur.fetchall() or []
    table_names = []
    for row in rows:
        table_name = row.get("table_name") or row.get("TABLE_NAME")
        if table_name and table_name not in _POINT_INFO_EXCLUDED_TABLES:
            table_names.append(table_name)
    _ttl_cache_set("point_info_tables", "__tables__", table_names, ttl_sec=600.0, max_items=8)
    return table_names


def _resolve_point_info_table(shield_id, available_tables):
    table_name = f"point_info_{shield_id}"
    if table_name not in available_tables:
        raise ValueError(f"Unsupported point info table: {table_name}")
    return table_name


@bp.route('/history/parameters', methods=['GET'])
def get_available_parameters():
    try:
        shield_id = str(request.args.get("shield_id") or "").strip()
        if not shield_id:
            return jsonify({"status": "error", "error": "缺少 shield_id 参数"}), 400
        cache_key = ("history_parameters", shield_id)
        cached = _ttl_cache_get("history_parameters", cache_key)
        if cached is not None:
            return jsonify(cached)
        with _get_point_info_connection() as conn:
            table_name = _resolve_point_info_table(shield_id, _list_point_info_tables(conn))
            table_sql = _quote_mysql_identifier(table_name)
            with conn.cursor() as cur:
                query = f"SELECT point_cn_name, point_code, point_unit FROM {table_sql} WHERE point_unit IS NOT NULL AND point_unit != '' AND point_code IS NOT NULL AND point_code != '' AND point_cn_name IS NOT NULL AND point_cn_name != ''"
                cur.execute(query)
                rows = cur.fetchall()
        result = [
            {"name": row["point_cn_name"], "code": row["point_code"], "unit": row["point_unit"]}
            for row in rows
        ]
        _ttl_cache_set("history_parameters", cache_key, result, ttl_sec=600.0, max_items=32)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"MySQL Query Error: {str(e)}"}), 500


@bp.route('/history/query', methods=['POST'])
def query_history_data():
    try:
        # 配置常量
        TARGET_DB_NAME = Config.HISTORY_TARGET_DB_NAME
        PROJECT_PREFIX = Config.HISTORY_PROJECT_PREFIX
        MYSQL_TABLE_NAME = f"point_info_{PROJECT_PREFIX}"
        mysql_table_sql = _quote_mysql_identifier(MYSQL_TABLE_NAME)

        # Get parameters from JSON body
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "error": "Missing JSON body"}), 400

        start_date_str = data.get("start_date")
        end_date_str = data.get("end_date")

        parameters = []
        params_raw = data.get("parameters")
        if isinstance(params_raw, list):
            parameters = params_raw
        elif isinstance(params_raw, str) and params_raw:
            params_raw = params_raw.strip()
            # Handle list format in string
            if params_raw.startswith("[") and params_raw.endswith("]"):
                try:
                    parameters = json.loads(params_raw)
                except Exception:
                    parameters = [p.strip() for p in params_raw.strip("[]").split(",") if p.strip()]
            else:
                parameters = [p.strip() for p in params_raw.split(",") if p.strip()]

        if not start_date_str or not end_date_str:
            return jsonify({"status": "error", "error": "Missing 'start_date' or 'end_date' parameter"}), 400

        if not parameters:
            return jsonify({"status": "error", "error": "Missing 'parameters' parameter"}), 400

        if not (1 <= len(parameters) <= 4):
            return jsonify({"status": "error", "error": "The number of parameters must be between 1 and 4"}), 400

        try:
            start_dt_local = pd.to_datetime(start_date_str).tz_localize('Asia/Shanghai')
            end_dt_local = pd.to_datetime(end_date_str).tz_localize('Asia/Shanghai') + timedelta(days=1)

            if start_dt_local >= end_dt_local:
                return jsonify({"status": "error", "error": "start_date cannot be later than end_date"}), 400

            start_dt_utc = start_dt_local.tz_convert('UTC')
            end_dt_utc = end_dt_local.tz_convert('UTC')

        except ValueError:
            return jsonify({"status": "error", "error": "Invalid date format. Use YYYY-MM-DD"}), 400

        point_mapping = {}

        cache_key = (MYSQL_TABLE_NAME, tuple(parameters))
        try:
            if cache_key in POINT_MAP_CACHE:
                point_mapping = POINT_MAP_CACHE.get(cache_key, {})
            else:
                with _get_point_info_connection() as conn:
                    with conn.cursor() as cur:
                        format_strings = ','.join(['%s'] * len(parameters))
                        query = f"SELECT point_code, point_cn_name, point_unit FROM {mysql_table_sql} WHERE point_cn_name IN ({format_strings})"
                        cur.execute(query, tuple(parameters))
                        rows = cur.fetchall()
                        for row in rows:
                            code_val = row['point_code']
                            if code_val:
                                code_val = code_val.strip()
                            point_mapping[row['point_cn_name']] = {
                                'code': code_val,
                                'unit': row['point_unit']
                            }
                POINT_MAP_CACHE[cache_key] = point_mapping
        except Exception as e:
            traceback.print_exc()
            return jsonify({"status": "error", "error": f"MySQL Query Error: {str(e)}"}), 500

        if not point_mapping:
            return jsonify({"status": "error", "error": "No matching parameters found in database"}), 404

        fields = []
        for p in parameters:
            if p in point_mapping:
                fields.append(_quote_influx_identifier(point_mapping[p]["code"]))

        if not fields:
            return jsonify({"status": "error", "error": "No valid fields to query"}), 400

        select_clause = ", ".join(fields)

        time_filter = f"time >= '{start_dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}' AND time < '{end_dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}'"

        all_points = []
        curr_date = start_dt_local.date() - timedelta(days=1)
        target_end_date = end_dt_local.date() + timedelta(days=1)

        measurement_names = []
        while curr_date < target_end_date:
            tn = f"{PROJECT_PREFIX}_DZ1360_{curr_date.strftime('%Y_%m_%d')}"
            measurement_names.append(tn)
            curr_date += timedelta(days=1)

        def _q(measurement_name):
            measurement_sql = _quote_influx_identifier(measurement_name)
            q = f'SELECT {select_clause} FROM {measurement_sql} WHERE {time_filter}'
            try:
                # 使用新的数据库名称 TARGET_DB_NAME
                res = client.query(q, database=TARGET_DB_NAME, epoch='ms')
                pts = list(res.get_points())
                return pts
            except Exception as e:
                return []

        max_workers = _env_int("HISTORY_QUERY_MAX_WORKERS", 8)
        with ThreadPoolExecutor(max_workers=min(max(1, max_workers), len(measurement_names) or 1)) as ex:
            futs = [ex.submit(_q, m) for m in measurement_names]
            for f in as_completed(futs):
                pts = f.result() or []
                if pts:
                    all_points.extend(pts)
        all_points.sort(key=lambda x: x.get('time', ''))

        mapping_info_list = []
        for cn_name in parameters:
            info = point_mapping.get(cn_name)
            if info:
                mapping_info_list.append((cn_name, info['code'], info['unit']))

        output_data = []
        tz_offset = timedelta(hours=8)

        for pt in all_points:
            ts = pt.get('time')
            # 快速时间格式化 (ts is milliseconds timestamp)
            dt = datetime.utcfromtimestamp(ts / 1000) + tz_offset
            time_str = dt.strftime("%Y/%m/%d %H:%M:%S")

            item = {"time": time_str}
            for cn_name, code, unit in mapping_info_list:
                val = pt.get(code)
                item[cn_name] = {
                    "value": val if val is not None else 0.0,
                    "unit": unit if unit else ""
                }
            output_data.append(item)

        return jsonify(output_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

import traceback

from flask import jsonify, request


def _get_param_value(keys):
    try:
        for key in keys:
            value = request.args.get(key)
            if value:
                return value

        body = request.get_json(silent=True)
        if isinstance(body, dict):
            for key in keys:
                value = body.get(key)
                if value:
                    return value
    except Exception:
        pass
    return None


def _risk_table_from_query(ctx):
    risk_type = _get_param_value(["risk_type", "risk", "type"])
    if not risk_type:
        return None, None

    risk_type = str(risk_type).strip()
    table = ctx.RISK_TABLES.get(risk_type)
    if table:
        return risk_type.replace("风险", ""), table

    short_name = risk_type.replace("风险", "")
    return short_name, ctx.RISK_TABLES.get(short_name)


def _project_code_from_query(ctx):
    project_code = _get_param_value(["project_code", "project"])
    if project_code:
        return str(project_code).strip()
    prefix = getattr(ctx.Config, "HISTORY_PROJECT_PREFIX", "")
    return str(prefix).upper() if prefix else "-"


def _score_to_probability(ctx, risk_score, full_risk_name):
    config = ctx._risk_config_by_type(full_risk_name) or {}
    points = config.get("score_points") or []
    if not points:
        return 0.0

    score = float(risk_score or 0.0)
    points = sorted(points, key=lambda item: float(item[1]), reverse=True)
    if score >= float(points[0][1]):
        return float(points[0][0])
    if score <= float(points[-1][1]):
        return float(points[-1][0])

    for (p0, s0), (p1, s1) in zip(points, points[1:]):
        p0, s0, p1, s1 = float(p0), float(s0), float(p1), float(s1)
        if s0 >= score >= s1:
            if s1 == s0:
                return p0
            probability = p0 + ((score - s0) / (s1 - s0)) * (p1 - p0)
            return max(0.0, min(1.0, probability))
    return float(points[-1][0])


def _probability_thresholds(ctx, full_risk_name):
    config = ctx._risk_config_by_type(full_risk_name) or {}
    points = sorted(config.get("score_points") or [], key=lambda item: float(item[0]))
    if len(points) >= 3:
        return float(points[2][0]), float(points[1][0])
    if len(points) >= 2:
        return float(points[-1][0]), float(points[1][0])
    return 0.5, 0.3


def register_misc_routes(bp, ctx):
    @bp.route("/filter/options", methods=["GET"])
    def get_filter_options():
        try:
            risk_name, table = _risk_table_from_query(ctx)
            if not table:
                return jsonify({"status": "error", "error": "不支持的风险类型"}), 400

            safety_level = "无风险Ⅰ"
            fault_measures = "-"
            project_code = _project_code_from_query(ctx)

            try:
                with ctx._mysql_connect() as conn:
                    row = ctx._fetch_latest_row(conn, table, project_code=project_code)
                    if row:
                        safety_level = row.get("safety_level") or row.get("risk_level") or "无风险Ⅰ"
                        fault_measures = row.get("fault_measures")
                        if fault_measures in (None, "", "-"):
                            fault_measures = row.get("measures")
                        fault_measures = ctx._clean_json_val(fault_measures)
            except Exception:
                traceback.print_exc()

            return jsonify(
                {
                    "project_code": project_code,
                    "safety_level": safety_level,
                    "potential_risk": (risk_name + "预警") if safety_level != "无风险Ⅰ" else "-",
                    "fault_measures": fault_measures,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(exc)}), 500

    @bp.route("/info/reverse", methods=["GET"])
    def get_reverse_risk_info():
        try:
            risk_name, table = _risk_table_from_query(ctx)
            if not table:
                return jsonify({"status": "error", "error": "不支持的风险类型"}), 400

            risk_score = 0.0
            try:
                with ctx._mysql_connect() as conn:
                    row = ctx._fetch_latest_row(conn, table, project_code=_project_code_from_query(ctx))
                    if row and row.get("risk_score") is not None:
                        risk_score = float(row.get("risk_score"))
            except Exception:
                traceback.print_exc()

            full_risk_name = risk_name + "风险"
            probability = _score_to_probability(ctx, risk_score, full_risk_name)
            potential_risk_threshold, normal_probability_threshold = _probability_thresholds(ctx, full_risk_name)

            return jsonify(
                {
                    "当前风险概率": round(probability, 3),
                    "潜在风险阈值": potential_risk_threshold,
                    "正常概率阈值": normal_probability_threshold,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(exc)}), 500

    @bp.route("/get_latest_state", methods=["GET"])
    def get_latest_state():
        try:
            target_db = ctx.Config.LATEST_STATE_TARGET_DB
            cached = ctx._ttl_cache_get("latest_state", ("latest_state", target_db))
            if isinstance(cached, dict):
                return jsonify(cached)

            result = ctx.client.query(
                "SHOW MEASUREMENTS WITH MEASUREMENT =~ /^tsjy_dz1360_DZ1360_/",
                database=target_db,
            )
            measurements = list(result.get_points()) if result is not None else []
            measurement_names = [item["name"] for item in measurements if item.get("name")]

            if not measurement_names:
                out = {"state": "-", "time": "-", "remark": "-"}
                ctx._ttl_cache_set("latest_state", ("latest_state", target_db), out, ttl_sec=3.0, max_items=32)
                return jsonify(out)

            latest_measurement = sorted(measurement_names)[-1]
            query = (
                f"SELECT state FROM {ctx._quote_influx_identifier(latest_measurement)} "
                f"ORDER BY time DESC LIMIT 1"
            )
            result = ctx.client.query(query, database=target_db)
            points = list(result.get_points())

            if not points:
                out = {"state": "-", "time": "-", "remark": "-"}
                ctx._ttl_cache_set("latest_state", ("latest_state", target_db), out, ttl_sec=3.0, max_items=32)
                return jsonify(out)

            state_val = points[0].get("state", "-")
            time_val = points[0].get("time", "-")
            remark = "-"
            try:
                remark = {0: "停机", 1: "掘进", 2: "拼环"}.get(int(state_val), "-")
            except (TypeError, ValueError):
                pass

            if time_val != "-":
                time_val = ctx.format_time_utc_to_shanghai(time_val)

            out = {"state": state_val, "time": time_val, "remark": remark}
            ctx._ttl_cache_set("latest_state", ("latest_state", target_db), out, ttl_sec=3.0, max_items=32)
            return jsonify(out)
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(exc)}), 500

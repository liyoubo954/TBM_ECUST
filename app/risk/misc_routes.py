# mport json
# import os
# import threading
# import traceback
# from flask import jsonify, request
#
#
# CORRELATION_DATA_CACHE = None
# CORRELATION_DATA_LOCK = threading.Lock()
# CORRELATION_MAPS_CACHE = None
#
# #
# def _get_correlation_data():
#     global CORRELATION_DATA_CACHE
#     with CORRELATION_DATA_LOCK:
#         if CORRELATION_DATA_CACHE is None:
#             json_path = os.path.join(os.path.dirname(__file__), "data", "监测参数关联.json")
#             if not os.path.exists(json_path):
#                 return None
#             try:
#                 with open(json_path, "r", encoding="utf-8") as f:
#                     CORRELATION_DATA_CACHE = json.load(f)
#             except Exception:
#                 traceback.print_exc()
#                 return None
#         return CORRELATION_DATA_CACHE
#
#
# def _get_correlation_maps():
#     global CORRELATION_MAPS_CACHE
#     if CORRELATION_MAPS_CACHE is not None:
#         return CORRELATION_MAPS_CACHE
#
#     data = _get_correlation_data()
#     if not data:
#         return None, None
#
#     with CORRELATION_DATA_LOCK:
#         if CORRELATION_MAPS_CACHE is not None:
#             return CORRELATION_MAPS_CACHE
#
#         param_system_map = {}
#         if "nodes" in data:
#             for sys_name, nodes in data["nodes"].items():
#                 for node in nodes:
#                     props = node.get("properties", {})
#                     p_name = props.get("name")
#                     if p_name:
#                         param_system_map[p_name] = sys_name
#
#         adjacency_map = {}
#         if "relationships" in data:
#             for rel in data["relationships"]:
#                 props = rel.get("properties", {})
#                 start_node = props.get("start_node")
#                 end_node = props.get("end_node")
#                 correlation = props.get("correlation")
#
#                 if not (start_node and end_node and correlation is not None):
#                     continue
#
#                 item = {
#                     "correlation": float(correlation),
#                     "abs_correlation": abs(float(correlation)),
#                     "direction": props.get("direction"),
#                     "level": props.get("level"),
#                 }
#
#                 item_start = item.copy()
#                 item_start["name"] = end_node
#                 adjacency_map.setdefault(start_node, []).append(item_start)
#
#                 item_end = item.copy()
#                 item_end["name"] = start_node
#                 adjacency_map.setdefault(end_node, []).append(item_end)
#
#         CORRELATION_MAPS_CACHE = (param_system_map, adjacency_map)
#         return CORRELATION_MAPS_CACHE
#
#
# def _get_param_value(keys):
#     try:
#         for k in keys:
#             v = request.args.get(k)
#             if v:
#                 return v
#         body = request.get_json(silent=True)
#         if isinstance(body, dict):
#             for k in keys:
#                 v = body.get(k)
#                 if v:
#                     return v
#     except Exception:
#         pass
#     return None
#
#
# def register_misc_routes(bp, ctx):
#     @bp.route("/filter/options", methods=["GET"])
#     def get_filter_options():
#         try:
#             rp, table = ctx._risk_table_from_query()
#             if not table:
#                 return jsonify({"status": "error", "error": "不支持的风险类型"}), 400
#
#             safety_level = "无风险Ⅰ"
#             fault_measures = "-"
#             potential_risk = "-"
#
#             try:
#                 with ctx._mysql_connect() as conn:
#                     row = ctx._fetch_latest_row(conn, table)
#                     if row:
#                         sl = row.get("safety_level")
#                         if sl and str(sl).strip():
#                             safety_level = sl
#                         else:
#                             safety_level = row.get("risk_level") or "无风险Ⅰ"
#
#                         fm = row.get("fault_measures")
#                         if fm in (None, "", "-"):
#                             fm = row.get("measures")
#
#                         fault_measures = ctx._clean_json_val(fm)
#             except Exception:
#                 traceback.print_exc()
#                 pass
#
#             potential_risk = (rp + "预警") if safety_level != "无风险Ⅰ" else "-"
#
#             return jsonify(
#                 {
#                     "project_code": ctx.PROJECT_CODE,
#                     "safety_level": safety_level,
#                     "potential_risk": potential_risk,
#                     "fault_measures": fault_measures,
#                 }
#             )
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"status": "error", "error": str(e)}), 500
#
#     @bp.route("/info/reverse", methods=["GET"])
#     def get_reverse_risk_info():
#         try:
#             rp, table = ctx._risk_table_from_query()
#             if not table:
#                 return jsonify({"status": "error", "error": "不支持的风险类型"}), 400
#
#             risk_score = 0.0
#             try:
#                 with ctx._mysql_connect() as conn:
#                     row = ctx._fetch_latest_row(conn, table)
#                     if row:
#                         rs = row.get("risk_score")
#                         if rs is not None:
#                             try:
#                                 risk_score = float(rs)
#                             except Exception:
#                                 pass
#             except Exception:
#                 traceback.print_exc()
#                 pass
#
#             full_risk_name = rp + "风险"
#             probability = ctx.risk_assessor.reverse_map_score_to_probability(risk_score, full_risk_name)
#             potential_risk_threshold, normal_probability_threshold = ctx.risk_assessor.get_probability_thresholds(
#                 full_risk_name
#             )
#
#             return jsonify(
#                 {
#                     "当前风险概率": round(probability, 3),
#                     "潜在风险阈值": potential_risk_threshold,
#                     "正常概率阈值": normal_probability_threshold,
#                 }
#             )
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"status": "error", "error": str(e)}), 500
#
#     @bp.route("/correlation/systems", methods=["GET"])
#     def get_correlation_systems():
#         try:
#             data = _get_correlation_data()
#             if not data or "nodes" not in data:
#                 return jsonify({"status": "error", "error": "关联数据不可用"}), 500
#
#             systems = list(data["nodes"].keys())
#             return jsonify({"systems": systems})
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"status": "error", "error": str(e)}), 500
#
#     @bp.route("/correlation/system_params", methods=["POST"])
#     def get_system_params():
#         try:
#             system_name = _get_param_value(["system", "system_name", "sys"])
#             if not system_name:
#                 return jsonify({"status": "error", "error": "Missing system"}), 400
#             system_name = str(system_name).strip()
#
#             data = _get_correlation_data()
#             if not data or "nodes" not in data:
#                 return jsonify({"status": "error", "error": "关联数据不可用"}), 500
#
#             nodes = data["nodes"].get(system_name)
#             if not nodes:
#                 return jsonify({"system": system_name, "parameters": []})
#
#             params = []
#             for node in nodes:
#                 props = node.get("properties", {})
#                 name = props.get("name")
#                 if name:
#                     params.append(name)
#
#             return jsonify({"parameters": params})
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"status": "error", "error": str(e)}), 500
#
#     @bp.route("/correlation/top5", methods=["POST"])
#     def get_top5_correlated():
#         try:
#             param_name = _get_param_value(["parameter", "parameter_name"])
#             if not param_name:
#                 return jsonify({"status": "error", "error": "Missing parameter"}), 400
#             param_name = str(param_name).strip()
#
#             param_system_map, adjacency_map = _get_correlation_maps()
#             if param_system_map is None or adjacency_map is None:
#                 return jsonify({"status": "error", "error": "关联数据不可用"}), 500
#
#             correlated = adjacency_map.get(param_name, [])
#             correlated.sort(key=lambda x: x["abs_correlation"], reverse=True)
#             top = correlated[:5]
#
#             result = {"parameter": param_name}
#             for idx, item in enumerate(top, start=1):
#                 p_name = item.get("name")
#                 result[str(idx)] = {
#                     "name": p_name,
#                     "correlation": item.get("correlation"),
#                     "direction": item.get("direction"),
#                     "level": item.get("level"),
#                     "system": param_system_map.get(p_name, "未知系统"),
#                 }
#             return jsonify(result)
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"status": "error", "error": str(e)}), 500
#
#     @bp.route("/get_latest_state", methods=["GET"])
#     def get_latest_state():
#         try:
#             target_db = ctx.Config.LATEST_STATE_TARGET_DB
#             cached = ctx._ttl_cache_get("latest_state", ("latest_state", target_db))
#             if isinstance(cached, dict):
#                 return jsonify(cached)
#
#             result = ctx.client.query(
#                 "SHOW MEASUREMENTS WITH MEASUREMENT =~ /^tsjy_dz1360_DZ1360_/",
#                 database=target_db,
#             )
#             measurements = list(result.get_points()) if result is not None else []
#             m_names = [m["name"] for m in measurements if m.get("name")]
#
#             if not m_names:
#                 out = {"state": "-", "time": "-", "remark": "-"}
#                 ctx._ttl_cache_set("latest_state", ("latest_state", target_db), out, ttl_sec=3.0, max_items=32)
#                 return jsonify(out)
#
#             latest_measurement = sorted(m_names)[-1]
#             query = (
#                 f"SELECT state FROM {ctx._quote_influx_identifier(latest_measurement)} ORDER BY time DESC LIMIT 1"
#             )
#             result = ctx.client.query(query, database=target_db)
#             points = list(result.get_points())
#
#             if points:
#                 state_val = points[0].get("state", "-")
#                 time_val = points[0].get("time", "-")
#
#                 remark = "-"
#                 try:
#                     state_int = int(state_val)
#                     if state_int == 0:
#                         remark = "停机"
#                     elif state_int == 1:
#                         remark = "掘进"
#                     elif state_int == 2:
#                         remark = "拼环"
#                 except (ValueError, TypeError):
#                     pass
#
#                 if time_val != "-":
#                     time_val = ctx.format_time_utc_to_shanghai(time_val)
#
#                 out = {"state": state_val, "time": time_val, "remark": remark}
#                 ctx._ttl_cache_set("latest_state", ("latest_state", target_db), out, ttl_sec=3.0, max_items=32)
#                 return jsonify(out)
#
#             out = {"state": "-", "time": "-", "remark": "-"}
#             ctx._ttl_cache_set("latest_state", ("latest_state", target_db), out, ttl_sec=3.0, max_items=32)
#             return jsonify(out)
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"status": "error", "error": str(e)}), 500

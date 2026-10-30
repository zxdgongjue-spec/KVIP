"""Bounded, shareable benchmark evidence; detailed JSON remains authoritative."""

import json


def render_record(reports, arguments, limit=14500):
    lines = [
        "# V9 服务器数据记录",
        "",
        "单位：延迟 us；设备计数 raw ticks，不假设频率。",
        "子阶段包含于父阶段，不可重复相加；各阶段中位数不保证可加。",
        "snapshot 不含恢复/校验/首次接管；stateful 诊断同步不适合性能比较。",
        "",
        "配置：" + json.dumps(arguments, ensure_ascii=False, default=str),
        "",
        "| B/Q/层/案例 | 状态 | fused p50/p95 | native p50/p95 | fused/native(p50) |",
        "|---|---|---|---|---|",
    ]
    details = []
    for r in reports:
        t = r.get("timing", {})
        f, n = t.get("fused_p50_us"), t.get("native_p50_us")
        label = f"{r.get('batch')}/{r.get('queries')}/{r.get('layer')}/{r.get('case')}"
        lines.append(
            f"| {label} | {r.get('status')} | {f}/{t.get('fused_p95_us')} | "
            f"{n}/{t.get('native_p95_us')} | {round(f / n, 3) if f and n else '-'} |"
        )
        evidence = {
            "case": label,
            "transport": r.get("transport", "host-model"),
            "binary_sha256": r.get("binary_sha256"),
            "build_options": r.get("build_options") if r is reports[0] else None,
            "device_resources": r.get("device_resources") if r is reports[0] else None,
            "workload": r.get("workload"),
            "software": r.get("software") if r is reports[0] else None,
            "native_configuration": r.get("native_configuration"),
            "native_graph_prewarm": r.get("native_graph_prewarm"),
            "timing": {
                k: v
                for k, v in t.items()
                if k
                in (
                    "api_wall",
                    "diagnostic_only",
                    "samples",
                    "baseline_execution",
                    "comparison_protocol",
                    "native_chain",
                    "score_layout",
                    "native_over_fused_speedup",
                    "phases",
                    "native_phases",
                    "li_service_windows",
                    "index_work_last_step",
                    "index_storage",
                    "mapping_validation",
                    "correctness_preflight",
                    "stateful_layout_diagnostics_timed",
                )
            },
            "device_ticks_p50_p95": {
                k: [v.get("p50_ticks"), v.get("p95_ticks")]
                for k, v in (t.get("device_stages") or {}).get("stages", {}).items()
            },
            "correctness": r.get("correctness_status"),
            "preflight": r.get("correctness_preflight"),
            "ownership_setup_us": r.get("index_cache_setup_us"),
            "h0_snapshot_seed_us": r.get("h0_snapshot_seed_us"),
            "error": r.get("error"),
            "completion": r.get("last_completion") if r.get("error") else None,
        }
        prep = r.get("caller_prepare_device_span_us", [])
        if prep:
            evidence["caller_prepare_us_min_max"] = [min(prep), max(prep)]
        details.append(
            json.dumps({k: v for k, v in evidence.items() if v is not None}, ensure_ascii=False, separators=(",", ":"))
        )
    suffix = "\n\n完整数据见同目录 summary.json 与各案例 report.json；本文件最多14500字符。\n"
    text = "\n".join(lines)
    omitted = 0
    for detail in details:
        block = "\n\n```json\n" + detail + "\n```"
        if len(text) + len(block) + len(suffix) + 100 <= limit:
            text += block
        else:
            omitted += 1
    if omitted:
        text += f"\n\n有 {omitted} 个案例的完整明细未展开，请查 JSON。"
    # Even very large case matrices/configurations cannot exceed the bound.
    return text[: limit - len(suffix)] + suffix

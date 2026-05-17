import argparse
import html
import json
from pathlib import Path


PERF_PREFIX = "PERF_JOB "


def add_timeline(job: dict) -> dict:
    wrapper_order = [
        ("desc_read", "Descriptor read", "work"),
        ("fetch_program", "Program fetch", "work"),
        ("fetch_input0", "Input0 fetch", "work"),
        ("fetch_input1", "Input1 fetch", "work"),
        ("start_core", "Core launch", "work"),
        ("wait_core", "Wait for core", "wait"),
        ("write_output", "Output writeback", "work"),
        ("done", "Done latch", "work"),
    ]
    core_order = [
        ("fetch", "Fetch/vector ops", "work"),
        ("matmul", "Matmul", "work"),
        ("done", "Done", "work"),
    ]

    wrapper_spans = []
    cursor = 0
    core_start = 0
    for key, label, kind in wrapper_order:
        value = int(job.get("wrapper", {}).get(key, 0))
        if key == "wait_core":
            core_start = cursor
        if value > 0:
            wrapper_spans.append(
                {
                    "label": label,
                    "start": cursor,
                    "end": cursor + value,
                    "cycles": value,
                    "kind": kind,
                }
            )
        cursor += value

    core_spans = []
    cursor = core_start
    for key, label, kind in core_order:
        value = int(job.get("core", {}).get(key, 0))
        if value > 0:
            core_spans.append(
                {
                    "label": label,
                    "start": cursor,
                    "end": cursor + value,
                    "cycles": value,
                    "kind": kind,
                }
            )
        cursor += value

    job["timeline"] = [
        {
            "module": "CPU firmware",
            "spans": [
                {"label": "MMIO start", "start": 0, "end": 1, "cycles": 1, "kind": "work"},
                {
                    "label": "Poll/wait for done",
                    "start": 1,
                    "end": int(job["total_cycles"]),
                    "cycles": max(0, int(job["total_cycles"]) - 1),
                    "kind": "wait",
                },
            ],
        },
        {"module": "NPU wrapper", "spans": wrapper_spans},
        {"module": "NPU core", "spans": core_spans},
    ]
    return job


def parse_perf_log(path: Path) -> dict:
    jobs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(PERF_PREFIX):
                jobs.append(add_timeline(json.loads(line[len(PERF_PREFIX) :])))
    if not jobs:
        raise ValueError(f"no {PERF_PREFIX.strip()} records found in {path}")
    return {
        "schema": "npu_perf_report_v0",
        "source_log": str(path),
        "summary": {
            "jobs": len(jobs),
            "total_cycles": sum(job["total_cycles"] for job in jobs),
            "max_job_cycles": max(job["total_cycles"] for job in jobs),
        },
        "jobs": jobs,
    }


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def write_html(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs_json = json.dumps(report["jobs"])
    report_json = json.dumps(report, indent=2)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NPU Cycle Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #647085;
      --line: #d9dee8;
      --accent: #2068d8;
      --accent-2: #1a9a7a;
      --accent-3: #c46b1f;
      --accent-4: #7b61d1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    header {{
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 650; letter-spacing: 0; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 16px; }}
    h3 {{ font-size: 14px; }}
    main {{ padding: 24px 28px 40px; max-width: 1200px; margin: 0 auto; }}
    .subtle {{ color: var(--muted); margin-top: 6px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric, .job, .raw {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric .value {{ font-size: 26px; font-weight: 700; margin-top: 8px; }}
    .grid {{ display: grid; gap: 16px; }}
    .job-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(110px, 160px) minmax(180px, 1fr) max-content;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
    }}
    .bar-value {{
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .bar-track {{
      height: 16px;
      background: #eef1f6;
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      min-width: 2px;
      border-radius: 4px;
    }}
    .module-title {{ margin-top: 12px; margin-bottom: 6px; color: var(--muted); }}
    .timeline-scroll {{
      width: 100%;
      overflow-x: auto;
      overflow-y: visible;
      padding-bottom: 4px;
    }}
    .timeline {{
      display: grid;
      grid-template-columns: max-content minmax(360px, 1fr) max-content;
      gap: 12px 12px;
      align-items: center;
      min-width: 680px;
      margin-top: 12px;
    }}
    .phase-timeline {{
      grid-template-columns: max-content minmax(360px, 1fr) max-content;
    }}
    .axis {{
      grid-column: 2;
      position: relative;
      height: 34px;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .axis-value-spacer {{ grid-column: 3; }}
    .axis::after {{
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: 4px;
      border-bottom: 1px solid var(--line);
    }}
    .tick {{
      position: absolute;
      bottom: 10px;
      transform: translateX(-50%);
      white-space: nowrap;
      background: var(--panel);
      padding: 0 4px;
    }}
    .lane-label {{ color: var(--ink); font-weight: 600; }}
    .timeline .lane-label {{
      white-space: nowrap;
    }}
    .phase-timeline .lane-label {{ font-weight: 500; color: #324056; }}
    .lane-value {{
      text-align: right;
      color: var(--ink);
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .lane {{
      position: relative;
      height: 34px;
      background: #eef1f6;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }}
    .span {{
      position: absolute;
      top: 4px;
      height: 24px;
      border-radius: 5px;
      color: #fff;
      display: flex;
      align-items: center;
      padding: 0 8px;
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 3px;
    }}
    .span.wait {{
      color: #253044;
      background: repeating-linear-gradient(
        45deg,
        #dce2ec,
        #dce2ec 6px,
        #cbd4e2 6px,
        #cbd4e2 12px
      );
      border: 1px solid #bcc7d8;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend i {{
      display: inline-block;
      width: 18px;
      height: 10px;
      border-radius: 3px;
      margin-right: 5px;
      vertical-align: -1px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.45;
    }}
    @media (max-width: 680px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .bar-row {{ grid-template-columns: minmax(140px, 1fr) max-content; gap: 5px; }}
      .bar-row > :first-child {{ grid-column: 1 / -1; }}
      .job-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>NPU Cycle Report</h1>
    <div class="subtle">Source: {html.escape(report["source_log"])}</div>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><h2>Jobs</h2><div class="value">{report["summary"]["jobs"]}</div></div>
      <div class="metric"><h2>Total Cycles</h2><div class="value">{report["summary"]["total_cycles"]}</div></div>
      <div class="metric"><h2>Max Job Cycles</h2><div class="value">{report["summary"]["max_job_cycles"]}</div></div>
    </section>
    <section class="grid" id="jobs"></section>
    <section class="raw" style="margin-top:16px">
      <h2 style="margin-bottom:10px">Raw JSON</h2>
      <pre>{html.escape(report_json)}</pre>
    </section>
  </main>
  <script>
    const jobs = {jobs_json};
    const colors = ["var(--accent)", "var(--accent-2)", "var(--accent-3)", "var(--accent-4)"];
    const timelineColors = {{
      "CPU firmware": "#7b61d1",
      "NPU wrapper": "#2068d8",
      "NPU core": "#1a9a7a"
    }};
    const wrappers = [
      ["desc_read", "Descriptor read"],
      ["fetch_program", "Program fetch"],
      ["fetch_input0", "Input0 fetch"],
      ["fetch_input1", "Input1 fetch"],
      ["start_core", "Core launch"],
      ["wait_core", "Core wait"],
      ["write_output", "Output writeback"],
      ["done", "Done latch"]
    ];
    const cores = [
      ["fetch", "Fetch/vector"],
      ["matmul", "Matmul"],
      ["done", "Done"]
    ];

    function row(key, label, value, total, colorIndex) {{
      const pct = total > 0 ? Math.max(0, Math.min(100, (value / total) * 100)) : 0;
      const div = document.createElement("div");
      div.className = "bar-row";
      div.innerHTML = `
        <div>${{label}}</div>
        <div class="bar-track"><div class="bar" style="width:${{pct}}%; background:${{colors[colorIndex % colors.length]}}"></div></div>
        <div class="bar-value">${{value}} cycles</div>
      `;
      return div;
    }}

    function renderModule(parent, title, entries, data, total, colorOffset) {{
      const h = document.createElement("h3");
      h.className = "module-title";
      h.textContent = title;
      parent.appendChild(h);
      entries.forEach(([key, label], idx) => parent.appendChild(row(key, label, data[key] || 0, total, idx + colorOffset)));
    }}

    function tickValues(total) {{
      if (total <= 10) return [0, total];
      const step = Math.max(1, Math.ceil(total / 4 / 10) * 10);
      const ticks = [0];
      for (let t = step; t < total; t += step) ticks.push(t);
      ticks.push(total);
      return ticks;
    }}

    function renderTimeline(parent, job) {{
      const title = document.createElement("h3");
      title.className = "module-title";
      title.textContent = "Cycle timeline";
      parent.appendChild(title);

      const scroll = document.createElement("div");
      scroll.className = "timeline-scroll";
      const timeline = document.createElement("div");
      timeline.className = "timeline";
      const spacer = document.createElement("div");
      const axis = document.createElement("div");
      axis.className = "axis";
      tickValues(job.total_cycles).forEach((tick) => {{
        const t = document.createElement("span");
        t.className = "tick";
        t.style.left = `${{(tick / job.total_cycles) * 100}}%`;
        t.textContent = tick;
        axis.appendChild(t);
      }});
      timeline.appendChild(spacer);
      timeline.appendChild(axis);
      const axisValueSpacer = document.createElement("div");
      axisValueSpacer.className = "axis-value-spacer";
      timeline.appendChild(axisValueSpacer);

      job.timeline.forEach((laneData) => {{
        const label = document.createElement("div");
        label.className = "lane-label";
        label.textContent = laneData.module;
        const lane = document.createElement("div");
        lane.className = "lane";
        laneData.spans.forEach((span) => {{
          const el = document.createElement("div");
          const width = ((span.end - span.start) / job.total_cycles) * 100;
          el.className = `span ${{span.kind}}`;
          el.style.left = `${{(span.start / job.total_cycles) * 100}}%`;
          el.style.width = `${{Math.max(width, 0.6)}}%`;
          if (span.kind !== "wait") el.style.background = timelineColors[laneData.module] || "var(--accent)";
          el.title = `${{laneData.module}}: ${{span.label}}\\n${{span.start}}-${{span.end}} cycles (${{span.cycles}})`;
          el.textContent = width >= 7 ? `${{span.label}} (${{span.cycles}})` : "";
          lane.appendChild(el);
        }});
        const laneValue = document.createElement("div");
        laneValue.className = "lane-value";
        const activeCycles = laneData.spans.reduce((sum, span) => sum + span.cycles, 0);
        laneValue.textContent = `${{activeCycles}} cycles`;
        timeline.appendChild(label);
        timeline.appendChild(lane);
        timeline.appendChild(laneValue);
      }});
      scroll.appendChild(timeline);
      parent.appendChild(scroll);

      const legend = document.createElement("div");
      legend.className = "legend";
      legend.innerHTML = `
        <span><i style="background:#2068d8"></i>active work</span>
        <span><i style="background:repeating-linear-gradient(45deg,#dce2ec,#dce2ec 6px,#cbd4e2 6px,#cbd4e2 12px); border:1px solid #bcc7d8"></i>wait/blocked</span>
      `;
      parent.appendChild(legend);
    }}

    function renderPhaseTimeline(parent, titleText, laneData, total, moduleColor) {{
      const title = document.createElement("h3");
      title.className = "module-title";
      title.textContent = titleText;
      parent.appendChild(title);

      const scroll = document.createElement("div");
      scroll.className = "timeline-scroll";
      const timeline = document.createElement("div");
      timeline.className = "timeline phase-timeline";
      const spacer = document.createElement("div");
      const axis = document.createElement("div");
      axis.className = "axis";
      tickValues(total).forEach((tick) => {{
        const t = document.createElement("span");
        t.className = "tick";
        t.style.left = `${{(tick / total) * 100}}%`;
        t.textContent = tick;
        axis.appendChild(t);
      }});
      timeline.appendChild(spacer);
      timeline.appendChild(axis);
      const axisValueSpacer = document.createElement("div");
      axisValueSpacer.className = "axis-value-spacer";
      timeline.appendChild(axisValueSpacer);

      laneData.spans.forEach((span) => {{
        const label = document.createElement("div");
        label.className = "lane-label";
        label.textContent = span.label;
        const lane = document.createElement("div");
        lane.className = "lane";
        const el = document.createElement("div");
        const width = ((span.end - span.start) / total) * 100;
        el.className = `span ${{span.kind}}`;
        el.style.left = `${{(span.start / total) * 100}}%`;
        el.style.width = `${{Math.max(width, 0.6)}}%`;
        if (span.kind !== "wait") el.style.background = moduleColor;
        el.title = `${{laneData.module}}: ${{span.label}}\\n${{span.start}}-${{span.end}} cycles (${{span.cycles}})`;
        el.textContent = width >= 7 ? `${{span.start}}-${{span.end}}` : "";
        lane.appendChild(el);
        const laneValue = document.createElement("div");
        laneValue.className = "lane-value";
        laneValue.textContent = `${{span.cycles}} cycles`;
        timeline.appendChild(label);
        timeline.appendChild(lane);
        timeline.appendChild(laneValue);
      }});
      scroll.appendChild(timeline);
      parent.appendChild(scroll);
    }}

    const root = document.getElementById("jobs");
    jobs.forEach((job) => {{
      const section = document.createElement("article");
      section.className = "job";
      section.innerHTML = `
        <div class="job-head">
          <h2>#${{job.id}} ${{job.name}}</h2>
          <div class="subtle">${{job.total_cycles}} cycles</div>
        </div>
      `;
      renderTimeline(section, job);
      renderPhaseTimeline(section, "Wrapper phases", job.timeline[1], job.total_cycles, timelineColors["NPU wrapper"]);
      renderPhaseTimeline(section, "Core phases", job.timeline[2], job.total_cycles, timelineColors["NPU core"]);
      root.appendChild(section);
    }});
  </script>
</body>
</html>
"""
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NPU cycle report UI from simulation log.")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--html-out", required=True, type=Path)
    args = parser.parse_args()

    report = parse_perf_log(args.log)
    write_json(report, args.json_out)
    write_html(report, args.html_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()

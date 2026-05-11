import json
import time

from IPython.display import HTML, Javascript, display


def render_sim_grid(data, viewer_id=None, graph_drop_initial=0, graph_drop_z=5.0):
    if viewer_id is None:
        viewer_id = f"simgrid_{int(time.time() * 1000)}"

    graph_drop_initial = max(0, int(graph_drop_initial))
    graph_drop_z = float(graph_drop_z)

    data_json = json.dumps(data)

    display(HTML(f'<div id="{viewer_id}"></div>'))
    display(Javascript(url="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"))

    js = f"""
(() => {{
  const ROOT_ID = {json.dumps(viewer_id)};
  const DATA = {data_json};
  const GRAPH_DROP_INITIAL = {graph_drop_initial};
  const GRAPH_DROP_Z = {graph_drop_z};

  function colabResize() {{
    try {{
      if (typeof google !== "undefined" && google.colab && google.colab.output) {{
        google.colab.output.setIframeHeight(document.body.scrollHeight, true);
      }}
    }} catch (e) {{}}
  }}

  function fmt(v) {{
    if (typeof v === "number" && Number.isFinite(v)) {{
      return Number(v).toFixed(6).replace(/\\.?0+$/, "").replace(/\\.$/, "");
    }}
    return String(v);
  }}

  function esc(v) {{
    return String(v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }}

  function uniqStrings(arr) {{
    if (!Array.isArray(arr)) return [];
    const seen = new Set();
    const out = [];
    for (const x of arr) {{
      if (typeof x === "string" && x.length > 0 && !seen.has(x)) {{
        seen.add(x);
        out.push(x);
      }}
    }}
    return out;
  }}

  function isFiniteNumber(v) {{
    return typeof v === "number" && Number.isFinite(v);
  }}

  function svgEl(tag) {{
    return document.createElementNS("http://www.w3.org/2000/svg", tag);
  }}

  function median(values) {{
    const arr = values.filter(isFiniteNumber).slice().sort((a, b) => a - b);
    if (arr.length === 0) return null;
    const mid = Math.floor(arr.length / 2);
    return arr.length % 2 ? arr[mid] : 0.5 * (arr[mid - 1] + arr[mid]);
  }}

  function mad(values, med) {{
    if (!isFiniteNumber(med)) return null;
    const dev = values
      .filter(isFiniteNumber)
      .map(v => Math.abs(v - med))
      .sort((a, b) => a - b);
    if (dev.length === 0) return null;
    const mid = Math.floor(dev.length / 2);
    return dev.length % 2 ? dev[mid] : 0.5 * (dev[mid - 1] + dev[mid]);
  }}

  function computeTrimStart(values, maxDrop, zThreshold) {{
    const n = values.length;
    if (!Number.isFinite(maxDrop) || maxDrop <= 0) return 0;

    const hardMax = Math.min(maxDrop, Math.max(0, n - 3));
    let start = 0;

    while (start < hardMax) {{
      const v = values[start];

      if (!isFiniteNumber(v)) {{
        start += 1;
        continue;
      }}

      const tail = values.slice(start + 1).filter(isFiniteNumber);
      if (tail.length < 4) break;

      const med = median(tail);
      const madVal = mad(tail, med);
      if (!isFiniteNumber(med)) break;

      const sigma = Math.max((isFiniteNumber(madVal) ? madVal * 1.4826 : 0), 1e-9);
      const rz = Math.abs(v - med) / sigma;

      if (rz > zThreshold) {{
        start += 1;
        continue;
      }}

      break;
    }}

    return start;
  }}

  if (window.__SIM_GRID_CLEANUP__) {{
    try {{ window.__SIM_GRID_CLEANUP__(); }} catch (e) {{}}
  }}

  const root = document.getElementById(ROOT_ID);
  if (!root) return;

  root.innerHTML = `
    <style>
      #${{ROOT_ID}} {{
        --bg: #ffffff;
        --panel: #ffffff;
        --panel-2: #f6f7fb;
        --panel-3: #eef1f6;
        --border: #d9dbe3;
        --text: #111111;
        --muted: #555b66;
        --title: #111111;
        --graph-line: #2d7ff9;
        --graph-fill: rgba(45, 127, 249, 0.14);
        --graph-grid: rgba(17, 17, 17, 0.10);
        --graph-axis: rgba(17, 17, 17, 0.22);
        --graph-dot: #2d7ff9;
        --graph-zero: rgba(220, 38, 38, 0.88);
        --graph-zero-soft: rgba(220, 38, 38, 0.14);
        --btn-bg: #ffffff;
        width: 100%;
      }}

      html[theme=dark] #${{ROOT_ID}} {{
        --bg: #1e1e1e;
        --panel: #252526;
        --panel-2: #2d2d30;
        --panel-3: #34343a;
        --border: #44474f;
        --text: #f1f3f4;
        --muted: #c7c9cc;
        --title: #ffffff;
        --graph-line: #7cb7ff;
        --graph-fill: rgba(124, 183, 255, 0.18);
        --graph-grid: rgba(241, 243, 244, 0.12);
        --graph-axis: rgba(241, 243, 244, 0.22);
        --graph-dot: #9dccff;
        --graph-zero: rgba(255, 120, 120, 0.92);
        --graph-zero-soft: rgba(255, 120, 120, 0.18);
        --btn-bg: #2d2d30;
      }}

      @media (prefers-color-scheme: dark) {{
        #${{ROOT_ID}} {{
          --bg: #1e1e1e;
          --panel: #252526;
          --panel-2: #2d2d30;
          --panel-3: #34343a;
          --border: #44474f;
          --text: #f1f3f4;
          --muted: #c7c9cc;
          --title: #ffffff;
          --graph-line: #7cb7ff;
          --graph-fill: rgba(124, 183, 255, 0.18);
          --graph-grid: rgba(241, 243, 244, 0.12);
          --graph-axis: rgba(241, 243, 244, 0.22);
          --graph-dot: #9dccff;
          --graph-zero: rgba(255, 120, 120, 0.92);
          --graph-zero-soft: rgba(255, 120, 120, 0.18);
          --btn-bg: #2d2d30;
        }}
      }}

      #${{ROOT_ID}} * {{
        box-sizing: border-box;
        min-width: 0;
      }}

      #${{ROOT_ID}} button,
      #${{ROOT_ID}} input[type="range"] {{
        accent-color: var(--graph-line);
      }}

      #${{ROOT_ID}} button {{
        border: 1px solid var(--border);
        background: var(--btn-bg);
        color: var(--text);
        padding: 6px 10px;
        border-radius: 8px;
        cursor: pointer;
      }}

      #${{ROOT_ID}} .simgrid-root {{
        width: 100%;
        max-width: 100%;
        font-family: system-ui, sans-serif;
        color: var(--text);
        background: var(--bg);
      }}

      #${{ROOT_ID}} .simgrid-toolbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        margin: 8px 0 12px 0;
        width: 100%;
      }}

      #${{ROOT_ID}} .simgrid-title {{
        font-weight: 700;
        color: var(--title);
      }}

      #${{ROOT_ID}} .simgrid-controls {{
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
        color: var(--text);
      }}

      #${{ROOT_ID}} .simgrid-grid {{
        display: grid;
        width: 100%;
        max-width: 100%;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
        align-items: start;
      }}

      #${{ROOT_ID}} .sim-card {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        width: 100%;
        max-width: 100%;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px;
        background: var(--panel);
        overflow: hidden;
      }}

      #${{ROOT_ID}} .sim-caption {{
        font-weight: 700;
        color: var(--title);
      }}

      #${{ROOT_ID}} .sim-host {{
        width: 100%;
        height: 260px;
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
        background: var(--panel-2);
      }}

      #${{ROOT_ID}} .sim-props {{
        display: grid;
        width: 100%;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 8px;
      }}

      #${{ROOT_ID}} .sim-prop {{
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--panel-2);
        padding: 8px 10px;
        min-width: 0;
      }}

      #${{ROOT_ID}} .sim-prop-key {{
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 2px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}

      #${{ROOT_ID}} .sim-prop-value {{
        display: block;
        font-size: 14px;
        font-weight: 700;
        color: var(--text);
        font-variant-numeric: tabular-nums;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}

      #${{ROOT_ID}} .sim-status {{
        font-size: 13px;
        color: var(--muted);
      }}

      #${{ROOT_ID}} .sim-graphs {{
        display: grid;
        width: 100%;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 8px;
        align-items: start;
      }}

      #${{ROOT_ID}} .sim-graph {{
        width: 100%;
        min-width: 0;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--panel-2);
        padding: 8px;
        overflow: hidden;
      }}

      #${{ROOT_ID}} .sim-graph-header {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 6px;
      }}

      #${{ROOT_ID}} .sim-graph-name {{
        font-size: 12px;
        color: var(--muted);
        font-weight: 700;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}

      #${{ROOT_ID}} .sim-graph-now {{
        font-size: 13px;
        color: var(--text);
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        flex: 0 0 auto;
      }}

      #${{ROOT_ID}} .sim-graph-layout {{
        display: grid;
        width: 100%;
        grid-template-columns: 52px minmax(0, 1fr);
        grid-template-rows: minmax(112px, auto) auto;
        grid-template-areas:
          "y chart"
          ". x";
        gap: 4px 8px;
        align-items: stretch;
      }}

      #${{ROOT_ID}} .sim-y-axis {{
        grid-area: y;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: flex-end;
        min-height: 112px;
        padding: 4px 0 8px 0;
        color: var(--muted);
        font-size: 11px;
        font-variant-numeric: tabular-nums;
        line-height: 1;
      }}

      #${{ROOT_ID}} .sim-chart-wrap {{
        grid-area: chart;
        width: 100%;
        min-width: 0;
        overflow: hidden;
      }}

      #${{ROOT_ID}} .sim-chart-svg {{
        display: block;
        width: 100%;
        height: auto;
        max-width: 100%;
      }}

      #${{ROOT_ID}} .sim-x-axis {{
        grid-area: x;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        width: 100%;
        min-width: 0;
        color: var(--muted);
        font-size: 11px;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
        overflow: hidden;
      }}

      #${{ROOT_ID}} .sim-x-axis > div {{
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}

      #${{ROOT_ID}} .sim-empty-graphs {{
        font-size: 12px;
        color: var(--muted);
      }}

      @media (max-width: 900px) {{
        #${{ROOT_ID}} .simgrid-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>

    <div class="simgrid-root">
      <div class="simgrid-toolbar">
        <div class="simgrid-title">${{esc(DATA.title || "Simulations")}}</div>
        <div class="simgrid-controls">
          <button id="${{ROOT_ID}}_play">Pause</button>
          <button id="${{ROOT_ID}}_reset">Reset</button>
          <label>
            Speed
            <input id="${{ROOT_ID}}_speed" type="range" min="0.1" max="3.0" value="1.0" step="0.1">
            <span id="${{ROOT_ID}}_speed_label">1.0x</span>
          </label>
        </div>
      </div>
      <div id="${{ROOT_ID}}_grid" class="simgrid-grid"></div>
    </div>
  `;

  const grid = document.getElementById(`${{ROOT_ID}}_grid`);
  const playBtn = document.getElementById(`${{ROOT_ID}}_play`);
  const resetBtn = document.getElementById(`${{ROOT_ID}}_reset`);
  const speedSlider = document.getElementById(`${{ROOT_ID}}_speed`);
  const speedLabel = document.getElementById(`${{ROOT_ID}}_speed_label`);

  const panels = [];
  const dummy = new THREE.Object3D();

  function maxBodies(sim) {{
    const frames = Array.isArray(sim?.frames) ? sim.frames : [];
    let m = 1;
    for (const f of frames) {{
      const n = Array.isArray(f?.bodies) ? f.bodies.length : 0;
      if (n > m) m = n;
    }}
    return m;
  }}

  function collectGraphMeta(sim) {{
    const frames = Array.isArray(sim?.frames) ? sim.frames : [];
    const meta = {{}};

    for (let i = 0; i < frames.length; i++) {{
      const frame = frames[i] || {{}};
      const props = (frame.props && typeof frame.props === "object") ? frame.props : {{}};
      const handles = uniqStrings(frame.graph);

      for (const handle of handles) {{
        if (!meta[handle]) {{
          meta[handle] = {{
            values: new Array(frames.length).fill(null),
            trimStart: 0,
            dataMin: Infinity,
            dataMax: -Infinity,
            labelMin: null,
            labelMax: null,
            plotMin: 0,
            plotMax: 1,
            isConstant: false,
            includesZero: false,
          }};
        }}

        const v = props[handle];
        if (isFiniteNumber(v)) {{
          meta[handle].values[i] = v;
        }}
      }}
    }}

    for (const handle of Object.keys(meta)) {{
      const m = meta[handle];
      m.trimStart = computeTrimStart(m.values, GRAPH_DROP_INITIAL, GRAPH_DROP_Z);

      const visible = m.values.slice(m.trimStart).filter(isFiniteNumber);

      if (visible.length === 0) {{
        m.labelMin = null;
        m.labelMax = null;
        m.plotMin = 0;
        m.plotMax = 1;
        m.includesZero = false;
        continue;
      }}

      m.dataMin = Infinity;
      m.dataMax = -Infinity;

      for (const v of visible) {{
        if (v < m.dataMin) m.dataMin = v;
        if (v > m.dataMax) m.dataMax = v;
      }}

      if (m.dataMin === m.dataMax) {{
        m.isConstant = true;
        m.labelMin = m.dataMin;
        m.labelMax = m.dataMax;
        const eps = Math.max(Math.abs(m.dataMin) * 1e-6, 1e-6);
        m.plotMin = m.dataMin - eps;
        m.plotMax = m.dataMax + eps;
      }} else {{
        m.isConstant = false;
        m.labelMin = m.dataMin;
        m.labelMax = m.dataMax;
        m.plotMin = m.dataMin;
        m.plotMax = m.dataMax;
      }}

      m.includesZero = m.labelMin !== null && m.labelMax !== null && m.labelMin <= 0 && 0 <= m.labelMax;
    }}

    return meta;
  }}

  function makeGraphCard(handle) {{
    const graph = document.createElement("div");
    graph.className = "sim-graph";

    const header = document.createElement("div");
    header.className = "sim-graph-header";

    const name = document.createElement("div");
    name.className = "sim-graph-name";
    name.textContent = handle;

    const nowVal = document.createElement("div");
    nowVal.className = "sim-graph-now";
    nowVal.textContent = "—";

    header.appendChild(name);
    header.appendChild(nowVal);
    graph.appendChild(header);

    const layout = document.createElement("div");
    layout.className = "sim-graph-layout";

    const yAxis = document.createElement("div");
    yAxis.className = "sim-y-axis";

    const yTop = document.createElement("div");
    yTop.textContent = "—";

    const yMid = document.createElement("div");
    yMid.textContent = "";

    const yBot = document.createElement("div");
    yBot.textContent = "—";

    yAxis.appendChild(yTop);
    yAxis.appendChild(yMid);
    yAxis.appendChild(yBot);

    const chartWrap = document.createElement("div");
    chartWrap.className = "sim-chart-wrap";

    const svg = svgEl("svg");
    svg.classList.add("sim-chart-svg");
    svg.setAttribute("viewBox", "0 0 320 112");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

    const defs = svgEl("defs");
    const clip = svgEl("clipPath");
    const clipId = `clip_${{ROOT_ID}}_${{Math.random().toString(36).slice(2)}}`;
    clip.setAttribute("id", clipId);

    const clipRect = svgEl("rect");
    clipRect.setAttribute("x", "0");
    clipRect.setAttribute("y", "0");
    clipRect.setAttribute("width", "320");
    clipRect.setAttribute("height", "112");
    clip.setAttribute("clipPathUnits", "userSpaceOnUse");
    clip.appendChild(clipRect);
    defs.appendChild(clip);
    svg.appendChild(defs);

    const g = svgEl("g");
    g.setAttribute("clip-path", `url(#${{clipId}})`);

    const grid1 = svgEl("line");
    grid1.setAttribute("x1", "0");
    grid1.setAttribute("y1", "8");
    grid1.setAttribute("x2", "320");
    grid1.setAttribute("y2", "8");
    grid1.setAttribute("stroke", "var(--graph-grid)");
    grid1.setAttribute("stroke-width", "1");

    const grid2 = svgEl("line");
    grid2.setAttribute("x1", "0");
    grid2.setAttribute("y1", "56");
    grid2.setAttribute("x2", "320");
    grid2.setAttribute("y2", "56");
    grid2.setAttribute("stroke", "var(--graph-grid)");
    grid2.setAttribute("stroke-width", "1");

    const grid3 = svgEl("line");
    grid3.setAttribute("x1", "0");
    grid3.setAttribute("y1", "104");
    grid3.setAttribute("x2", "320");
    grid3.setAttribute("y2", "104");
    grid3.setAttribute("stroke", "var(--graph-grid)");
    grid3.setAttribute("stroke-width", "1");

    const zeroLine = svgEl("line");
    zeroLine.setAttribute("x1", "0");
    zeroLine.setAttribute("y1", "0");
    zeroLine.setAttribute("x2", "320");
    zeroLine.setAttribute("y2", "0");
    zeroLine.setAttribute("stroke", "var(--graph-zero)");
    zeroLine.setAttribute("stroke-width", "1.5");
    zeroLine.setAttribute("stroke-dasharray", "4 3");
    zeroLine.style.display = "none";

    const area = svgEl("path");
    area.setAttribute("fill", "var(--graph-fill)");
    area.setAttribute("stroke", "none");

    const line = svgEl("path");
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "var(--graph-line)");
    line.setAttribute("stroke-width", "2.5");
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("stroke-linejoin", "round");

    const dot = svgEl("circle");
    dot.setAttribute("r", "3.5");
    dot.setAttribute("fill", "var(--graph-dot)");
    dot.setAttribute("stroke", "var(--panel)");
    dot.setAttribute("stroke-width", "1.5");
    dot.style.display = "none";

    const axisBase = svgEl("line");
    axisBase.setAttribute("x1", "0");
    axisBase.setAttribute("y1", "104");
    axisBase.setAttribute("x2", "320");
    axisBase.setAttribute("y2", "104");
    axisBase.setAttribute("stroke", "var(--graph-axis)");
    axisBase.setAttribute("stroke-width", "1");

    g.appendChild(grid1);
    g.appendChild(grid2);
    g.appendChild(grid3);
    g.appendChild(zeroLine);
    g.appendChild(area);
    g.appendChild(line);
    g.appendChild(dot);
    g.appendChild(axisBase);
    svg.appendChild(g);

    chartWrap.appendChild(svg);

    const xAxis = document.createElement("div");
    xAxis.className = "sim-x-axis";

    const xStart = document.createElement("div");
    xStart.textContent = "t=0";

    const xCenter = document.createElement("div");
    xCenter.textContent = "";

    const xEnd = document.createElement("div");
    xEnd.textContent = "";

    xAxis.appendChild(xStart);
    xAxis.appendChild(xCenter);
    xAxis.appendChild(xEnd);

    layout.appendChild(yAxis);
    layout.appendChild(chartWrap);
    layout.appendChild(xAxis);
    graph.appendChild(layout);

    return {{
      root: graph,
      nowVal,
      yTop,
      yMid,
      yBot,
      xStart,
      xCenter,
      xEnd,
      zeroLine,
      area,
      line,
      dot,
    }};
  }}

  function ensureGraphCards(panel, handles) {{
    const wanted = uniqStrings(handles);
    const wantedSet = new Set(wanted);
    let changed = false;

    for (const [handle, card] of Array.from(panel.graphCards.entries())) {{
      if (!wantedSet.has(handle)) {{
        card.root.remove();
        panel.graphCards.delete(handle);
        changed = true;
      }}
    }}

    for (const handle of wanted) {{
      let card = panel.graphCards.get(handle);
      if (!card) {{
        card = makeGraphCard(handle);
        panel.graphCards.set(handle, card);
        changed = true;
      }}
      panel.graphs.appendChild(card.root);
    }}

    panel.emptyGraphs.style.display = wanted.length ? "none" : "block";
    return changed;
  }}

  function updateGraph(panel, handle, frameIndex) {{
    const card = panel.graphCards.get(handle);
    const meta = panel.graphMeta[handle];
    if (!card || !meta) return;

    const values = meta.values || [];
    const n = values.length;
    const left = 0;
    const right = 320;
    const top = 8;
    const bottom = 104;
    const w = right - left;
    const h = bottom - top;

    const safeIndex = Math.max(0, Math.min(frameIndex, n - 1));
    const startIndex = Math.max(0, Math.min(meta.trimStart || 0, n));
    const denom = (meta.plotMax - meta.plotMin) || 1;

    let lineD = "";
    let areaD = "";
    let started = false;
    let lastX = left;
    let lastY = bottom;
    let lastValue = null;

    const visibleCount = Math.max(1, n - startIndex);

    for (let i = startIndex; i <= safeIndex; i++) {{
      const v = values[i];
      if (!isFiniteNumber(v)) {{
        started = false;
        continue;
      }}

      const t = visibleCount <= 1 ? 0 : (i - startIndex) / (visibleCount - 1);
      const x = left + t * w;
      const y = bottom - ((v - meta.plotMin) / denom) * h;

      if (!started) {{
        lineD += `M ${{x}} ${{y}}`;
        areaD += `M ${{x}} ${{bottom}} L ${{x}} ${{y}}`;
        started = true;
      }} else {{
        lineD += ` L ${{x}} ${{y}}`;
        areaD += ` L ${{x}} ${{y}}`;
      }}

      lastX = x;
      lastY = y;
      lastValue = v;
    }}

    if (started) {{
      areaD += ` L ${{lastX}} ${{bottom}} Z`;
      card.area.setAttribute("d", areaD);
      card.line.setAttribute("d", lineD);
      card.dot.setAttribute("cx", String(lastX));
      card.dot.setAttribute("cy", String(lastY));
      card.dot.style.display = "block";
      card.nowVal.textContent = fmt(lastValue);
    }} else {{
      card.area.setAttribute("d", "");
      card.line.setAttribute("d", "");
      card.dot.style.display = "none";
      card.nowVal.textContent = "—";
    }}

    if (meta.includesZero) {{
      const zeroY = bottom - ((0 - meta.plotMin) / denom) * h;
      card.zeroLine.setAttribute("y1", String(zeroY));
      card.zeroLine.setAttribute("y2", String(zeroY));
      card.zeroLine.style.display = "block";
    }} else {{
      card.zeroLine.style.display = "none";
    }}

    if (meta.labelMin === null || meta.labelMax === null) {{
      card.yTop.textContent = "";
      card.yMid.textContent = "";
      card.yBot.textContent = "";
    }} else if (meta.includesZero) {{
      card.yTop.textContent = fmt(meta.labelMax);
      card.yMid.textContent = "0";
      card.yBot.textContent = fmt(meta.labelMin);
    }} else {{
      card.yTop.textContent = fmt(meta.labelMax);
      card.yBot.textContent = fmt(meta.labelMin);
      card.yMid.textContent = meta.isConstant ? "constant" : fmt((meta.labelMin + meta.labelMax) * 0.5);
    }}

    const dt = Number(panel.sim?.dt);
    const totalTime = Number.isFinite(dt) && dt > 0 ? Math.max(0, (n - 1) * dt) : null;
    const startTime = Number.isFinite(dt) && dt > 0 ? startIndex * dt : null;
    const currentTime = Number.isFinite(dt) && dt > 0 ? safeIndex * dt : null;

    card.xStart.textContent = startTime === null ? `f=${{startIndex}}` : `t=${{fmt(startTime)}}`;
    card.xCenter.textContent = currentTime === null ? `frame ${{safeIndex}}` : `t=${{fmt(currentTime)}}`;
    card.xEnd.textContent = totalTime === null ? `f=${{Math.max(0, n - 1)}}` : `t=${{fmt(totalTime)}}`;
  }}

  function makePanel(sim, idx) {{
    const card = document.createElement("div");
    card.className = "sim-card";

    const caption = document.createElement("div");
    caption.className = "sim-caption";
    caption.textContent = sim.caption || `Simulation ${{idx + 1}}`;
    card.appendChild(caption);

    const host = document.createElement("div");
    host.className = "sim-host";
    card.appendChild(host);

    const props = document.createElement("div");
    props.className = "sim-props";
    card.appendChild(props);

    const graphs = document.createElement("div");
    graphs.className = "sim-graphs";
    card.appendChild(graphs);

    const emptyGraphs = document.createElement("div");
    emptyGraphs.className = "sim-empty-graphs";
    emptyGraphs.textContent = "No tracked graphs in this frame.";
    graphs.appendChild(emptyGraphs);

    grid.appendChild(card);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(host.clientWidth || 300, host.clientHeight || 260);
    renderer.setClearAlpha(0);
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const radius = sim.radius || 5.0;

    const camera = new THREE.PerspectiveCamera(
      45,
      (host.clientWidth || 300) / (host.clientHeight || 260),
      0.01,
      1000
    );
    camera.position.set(radius * 1.8, radius * 1.2, radius * 1.8);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.82));

    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(radius * 2.0, radius * 3.0, radius * 2.0);
    scene.add(sun);

    const gridHelper = new THREE.GridHelper(radius * 4.0, 12, 0x999999, 0xdddddd);
    gridHelper.position.y = -radius * 0.55;
    scene.add(gridHelper);

    const geom = new THREE.BoxGeometry(1, 1, 1);
    const mat = new THREE.MeshStandardMaterial({{
      color: 0x8fbcd4,
      roughness: 0.85,
      metalness: 0.05
    }});

    const mesh = new THREE.InstancedMesh(geom, mat, maxBodies(sim));
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    scene.add(mesh);

    return {{
      sim,
      host,
      props,
      graphs,
      emptyGraphs,
      renderer,
      scene,
      camera,
      mesh,
      graphMeta: collectGraphMeta(sim),
      graphCards: new Map(),
    }};
  }}

  function renderProps(panel, props) {{
    const entries = Object.entries((props && typeof props === "object") ? props : {{}});
    if (entries.length === 0) {{
      panel.props.innerHTML = `<div class="sim-status">No props</div>`;
      return;
    }}

    panel.props.innerHTML = entries.map(([k, v]) => `
      <div class="sim-prop">
        <span class="sim-prop-key">${{esc(k)}}</span>
        <span class="sim-prop-value">${{esc(fmt(v))}}</span>
      </div>
    `).join("");
  }}

  function applyFrame(panel, frameIndex) {{
    const sim = panel?.sim;
    const frames = Array.isArray(sim?.frames) ? sim.frames : [];

    if (!sim || frames.length === 0) {{
      panel.mesh.count = 0;
      panel.props.innerHTML = `<div class="sim-status">No frames</div>`;
      ensureGraphCards(panel, []);
      return;
    }}

    const safeIndex = Number.isFinite(frameIndex)
      ? ((frameIndex % frames.length) + frames.length) % frames.length
      : 0;

    const frame = frames[safeIndex];
    if (!frame) {{
      panel.mesh.count = 0;
      panel.props.innerHTML = `<div class="sim-status">Missing frame</div>`;
      ensureGraphCards(panel, []);
      return;
    }}

    const bodies = Array.isArray(frame.bodies) ? frame.bodies : [];
    panel.mesh.count = bodies.length;

    for (let i = 0; i < bodies.length; i++) {{
      const b = bodies[i];
      if (!b) continue;

      dummy.position.set(
        (b.pos && b.pos[0]) ?? 0,
        (b.pos && b.pos[1]) ?? 0,
        (b.pos && b.pos[2]) ?? 0
      );

      dummy.quaternion.set(
        (b.q && b.q[1]) ?? 0,
        (b.q && b.q[2]) ?? 0,
        (b.q && b.q[3]) ?? 0,
        (b.q && b.q[0]) ?? 1
      );

      dummy.scale.set(
        2 * ((b.half && b.half[0]) ?? 0.5),
        2 * ((b.half && b.half[1]) ?? 0.5),
        2 * ((b.half && b.half[2]) ?? 0.5)
      );

      dummy.updateMatrix();
      panel.mesh.setMatrixAt(i, dummy.matrix);
      panel.mesh.setColorAt(i, new THREE.Color(b.color || "#8fbcd4"));
    }}

    if (bodies.length === 0) {{
      dummy.position.set(1e9, 1e9, 1e9);
      dummy.quaternion.set(0, 0, 0, 1);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      panel.mesh.setMatrixAt(0, dummy.matrix);
    }}

    panel.mesh.instanceMatrix.needsUpdate = true;
    if (panel.mesh.instanceColor) panel.mesh.instanceColor.needsUpdate = true;
    panel.mesh.computeBoundingBox();
    panel.mesh.computeBoundingSphere();

    const props = (frame.props && typeof frame.props === "object") ? frame.props : {{}};
    renderProps(panel, props);

    const handles = uniqStrings(frame.graph);
    const changed = ensureGraphCards(panel, handles);

    for (const handle of handles) {{
      updateGraph(panel, handle, safeIndex);
    }}

    if (changed) setTimeout(colabResize, 0);
  }}

  function resize() {{
    for (const p of panels) {{
      const w = p.host.clientWidth || 300;
      const h = p.host.clientHeight || 260;
      p.camera.aspect = w / h;
      p.camera.updateProjectionMatrix();
      p.renderer.setSize(w, h);
    }}
    colabResize();
  }}

  for (let i = 0; i < (Array.isArray(DATA.sims) ? DATA.sims.length : 0); i++) {{
    panels.push(makePanel(DATA.sims[i], i));
  }}

  let playing = true;
  let speed = 1.0;
  let timeSec = 0.0;
  let lastT = performance.now();
  let raf = null;

  speedSlider.oninput = () => {{
    speed = parseFloat(speedSlider.value);
    speedLabel.textContent = speed.toFixed(1) + "x";
  }};

  playBtn.onclick = () => {{
    playing = !playing;
    playBtn.textContent = playing ? "Pause" : "Play";
  }};

  resetBtn.onclick = () => {{
    timeSec = 0.0;
  }};

  function loop(now) {{
    const dtReal = (now - lastT) * 0.001;
    lastT = now;

    if (playing) timeSec += dtReal * speed;

    for (const p of panels) {{
      const sim = p?.sim;
      const dt = Number(sim?.dt);
      const frameCount = Array.isArray(sim?.frames) ? sim.frames.length : 0;

      if (!sim || !Number.isFinite(dt) || dt <= 0 || frameCount === 0) {{
        p.mesh.count = 0;
        p.props.innerHTML = `<div class="sim-status">Invalid sim data</div>`;
        ensureGraphCards(p, []);
        p.renderer.render(p.scene, p.camera);
        continue;
      }}

      const frameIndex = Math.floor(timeSec / dt);
      applyFrame(p, frameIndex);
      p.renderer.render(p.scene, p.camera);
    }}

    raf = requestAnimationFrame(loop);
  }}

  window.addEventListener("resize", resize);
  resize();
  raf = requestAnimationFrame(loop);

  window.__SIM_GRID_CLEANUP__ = function() {{
    if (raf !== null) cancelAnimationFrame(raf);
    window.removeEventListener("resize", resize);

    for (const p of panels) {{
      try {{ p.renderer.dispose(); }} catch (e) {{}}
      if (p.host) p.host.innerHTML = "";
      if (p.graphs) p.graphs.innerHTML = "";
    }}
  }};
}})();
"""
    display(Javascript(js))

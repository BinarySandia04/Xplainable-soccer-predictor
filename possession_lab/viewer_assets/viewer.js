(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);

  const LABELS = ["H", "A", "D", "X"];
  const COLORS = {
    H: "#4ea8ff",
    A: "#ff6b5f",
    D: "#ffc857",
    X: "#607286",
  };
  const NAMES = {
    H: "Barcelona",
    A: "Athletic Club",
    D: "Balón muerto",
    X: "Sin evaluar",
  };
  const TEAM_NAMES = {
    "-1": "N/A",
    0: "Barcelona",
    1: "Athletic Club",
  };
  const MONO = '"SFMono-Regular", "Cascadia Code", "Roboto Mono", monospace';
  const BALL_LAYERS = ["raw", "yolo", "video", "combined", "fusion"];
  const LAYER_NAMES = {
    raw: "ATD bruto",
    combined: "Combinado",
    video: "Balón vídeo",
    yolo: "YOLO balón",
    fusion: "Fusión",
  };
  const LAYER_SHORT_NAMES = {
    raw: "ATD",
    combined: "comb.",
    video: "vídeo",
    yolo: "YOLO",
    fusion: "fusión",
  };
  const SOURCE_STATUS_NAMES = {
    observed: "observado",
    estimated: "estimado",
    abstain: "sin datos",
    event: "evento",
    unknown: "?",
  };
  const ui = {
    video: $("#video"),
    overlay: $("#overlay"),
    overlayToggle: $("#overlayToggle"),
    yoloPlayersToggle: $("#yoloPlayersToggle"),
    yoloCandidatesToggle: $("#yoloCandidatesToggle"),
    layerToggles: Array.from(document.querySelectorAll("[data-layer]")),
    pitchMap: $("#pitchMap"),
    pitchStatus: $("#pitchStatus"),
    pitchEmpty: $("#pitchEmpty"),
    videoError: $("#videoError"),
    videoBuffer: $("#videoBuffer"),
    videoBufferText: $("#videoBufferText"),
    videoBuffered: $("#videoBuffered"),
    syncState: $("#syncState"),
    clock: $("#clock"),
    frameMeta: $("#frameMeta"),
    frameNumber: $("#frameNumber"),
    timecode: $("#timecode"),
    predBadge: $("#predBadge"),
    predName: $("#predName"),
    gtBadge: $("#gtBadge"),
    gtName: $("#gtName"),
    utBadge: $("#utBadge"),
    utName: $("#utName"),
    agreement: $("#agreement"),
    utAgreement: $("#utAgreement"),
    fusionAgreement: $("#fusionAgreement"),
    fusionBadge: $("#fusionBadge"),
    fusionName: $("#fusionName"),
    fusedBadge: $("#fusedBadge"),
    fusedName: $("#fusedName"),
    fusionConfidence: $("#fusionConfidence"),
    fusionSummary: $("#fusionSummary"),
    fusionStatus: $("#fusionStatus"),
    sourceList: $("#sourceList"),
    spellText: $("#spellText"),
    spellDuration: $("#spellDuration"),
    tagText: $("#tagText"),
    tagDetail: $("#tagDetail"),
    gtWindow: $("#gtWindow"),
    trackingStatus: $("#trackingStatus"),
    ballState: $("#ballState"),
    ballPosition: $("#ballPosition"),
    ballSpeed: $("#ballSpeed"),
    teamSpeed: $("#teamSpeed"),
    nearPlayer: $("#nearPlayer"),
    ballEdge: $("#ballEdge"),
    ballGap: $("#ballGap"),
    touchState: $("#touchState"),
    eventState: $("#eventState"),
    videoDuration: $("#videoDuration"),
    metricsCard: $("#metricsCard"),
    metrics: $("#metrics"),
    utMetrics: $("#utMetrics"),
    utStatus: $("#utStatus"),
    utAnchor: $("#utAnchor"),
    utClear: $("#utClear"),
    playBtn: $("#playBtn"),
    frameInput: $("#frameInput"),
    goFrame: $("#goFrame"),
    gotoPublic: $("#gotoPublic"),
    gotoStart: $("#gotoStart"),
    prevChange: $("#prevChange"),
    nextChange: $("#nextChange"),
    prevSpell: $("#prevSpell"),
    nextSpell: $("#nextSpell"),
    prevDiff: $("#prevDiff"),
    nextDiff: $("#nextDiff"),
    overview: $("#overview"),
    detail: $("#detail"),
    overviewHint: $("#overviewHint"),
    detailHint: $("#detailHint"),
    hoverInfo: $("#hoverInfo"),
    explainCard: $("#explainCard"),
    switchDetail: $("#switchDetail"),
    switchList: $("#switchList"),
    switchMeta: $("#switchMeta"),
    explainFilters: $("#explainFilters"),
    fatal: $("#fatal"),
    fatalMessage: $("#fatalMessage"),
  };

  const state = {
    data: null,
    video: ui.video,
    frame: -1,
    initialFrame: 0,
    scrubbing: false,
    pointerId: null,
    scrubSource: null,
    scrubWindow: null,
    hoverFrame: null,
    hoverSource: null,
    loadedMetadata: false,
    syncWarning: false,
    lastRenderAt: 0,
    pendingSeekFrame: null,
    scrubRaf: null,
    pendingScrubFrame: null,
    utAnchors: [],
    utAnchorsSrc: null,
    utAnchor: null,
    utSaving: false,
    anchorDrag: null,
    selectedSwitch: -1,
    switchFilter: "all",
    switchFrames: [],
    overlayEnabled: true,
    layers: { raw: true, combined: true, video: true, yolo: true, fusion: true },
    yoloPlayersEnabled: true,
    yoloCandidatesEnabled: true,
    frameData: null,
    frameDataRequest: null,
    frameCache: new Map(),
    frameDataError: false,
  };

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function codeAt(text, frame) {
    if (!text) return null;
    return text.charAt(frame);
  }

  function codeClass(code) {
    return LABELS.includes(code) ? code : "X";
  }

  function setBadge(element, code) {
    const c = codeClass(code);
    element.className = `badge badge-${c}`;
    element.textContent = code || "—";
  }

  function setSync(text, kind) {
    ui.syncState.textContent = text;
    ui.syncState.className = `status-pill ${kind || ""}`.trim();
  }

  function pad(value, length) {
    return String(value).padStart(length, "0");
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    const totalMs = Math.round(seconds * 1000);
    const hours = Math.floor(totalMs / 3_600_000);
    const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
    const secs = Math.floor((totalMs % 60_000) / 1000);
    const millis = totalMs % 1000;
    return `${hours}:${pad(minutes, 2)}:${pad(secs, 2)}.${pad(millis, 3)}`;
  }

  function formatShort(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    const total = Math.floor(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (hours > 0) return `${hours}:${pad(minutes, 2)}:${pad(secs, 2)}`;
    return `${minutes}:${pad(secs, 2)}`;
  }

  function formatFrame(frame) {
    return frame.toLocaleString("es-ES");
  }

  function frameFromTime(seconds) {
    if (!state.data) return 0;
    const raw = Math.floor(seconds * state.data.fps + 1e-6);
    return clamp(raw, 0, state.data.n_frames - 1);
  }

  function timeForFrame(frame) {
    return (frame + 0.5) / state.data.fps;
  }

  function lowerBound(values, target) {
    let lo = 0;
    let hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (values[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function findInterval(intervals, frame) {
    if (!intervals || intervals.length === 0) return null;
    let lo = 0;
    let hi = intervals.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (intervals[mid][0] <= frame) lo = mid + 1;
      else hi = mid;
    }
    const candidate = intervals[lo - 1];
    return candidate && frame >= candidate[0] && frame < candidate[1] ? candidate : null;
  }

  function previousValue(values, frame) {
    if (!values || values.length === 0) return null;
    const index = lowerBound(values, frame);
    return index > 0 ? values[index - 1] : null;
  }

  function nextValue(values, frame) {
    if (!values || values.length === 0) return null;
    const index = lowerBound(values, frame + 1);
    return index < values.length ? values[index] : null;
  }

  function findTag(frame) {
    const tags = state.data ? state.data.tags : null;
    if (!tags) return null;
    for (const tag of tags) {
      if (tag.start <= frame && frame < tag.stop) return tag;
    }
    return null;
  }

  function utCodeAt(frame) {
    const segment = findInterval(state.data ? state.data.ut_segments : null, frame);
    return segment ? LABELS[segment[2]] : "X";
  }

  function metricText(value) {
    if (typeof value !== "number") return String(value);
    return Number.isInteger(value) ? value.toLocaleString("es-ES") : value.toFixed(3);
  }

  function numberText(value, digits = 2, suffix = "") {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return `${Number(value).toFixed(digits)}${suffix}`;
  }

  function teamText(team) {
    return TEAM_NAMES[String(team)] || "N/A";
  }

  function sourceText(source) {
    if (source === 0) return "observado";
    if (source === 1) return "estimado";
    if (source === 3) return "anclado ATD";
    return "perdido";
  }

  function setText(element, text, title) {
    if (element.textContent !== text) element.textContent = text;
    if (title !== undefined && element.title !== title) element.title = title;
  }

  function layerData(frameData, layer) {
    if (!frameData) return null;
    if (frameData.layers && frameData.layers[layer]) return frameData.layers[layer];
    if (layer === "combined") return frameData.ball || null;
    return null;
  }

  function layerSummary(frameData, compact = false, includeGap = true) {
    if (!frameData) return "cargando seguimiento…";
    const parts = [];
    for (const layer of ["fusion", "combined", "video", "yolo", "raw"]) {
      if (!state.layers[layer]) continue;
      const data = layerData(frameData, layer);
      const source = data ? Number(data.source) : 2;
      const gap = data ? Number(data.gap_s) : NaN;
      const gapText =
        includeGap && Number.isFinite(gap) && gap > 0 && source !== 0
          ? ` (${gap.toFixed(2)} s)`
          : "";
      const name = compact ? LAYER_SHORT_NAMES[layer] : LAYER_NAMES[layer];
      parts.push(`${name} ${sourceText(source)}${gapText}`);
    }
    return parts.length ? parts.join(" · ") : "sin capas de balón";
  }

  function layerColor(layer, source, alpha = 1) {
    if (layer === "raw") return `rgba(244, 248, 251, ${alpha})`;
    if (layer === "video") {
      if (source === 1) return `rgba(255, 200, 87, ${alpha})`;
      if (source === 3) return `rgba(85, 209, 135, ${alpha})`;
      return `rgba(46, 220, 150, ${alpha})`;
    }
    if (layer === "yolo") return `rgba(255, 180, 92, ${alpha})`;
    if (layer === "fusion") return `rgba(232, 121, 249, ${alpha})`;
    return source === 1
      ? `rgba(255, 200, 87, ${alpha})`
      : `rgba(244, 248, 251, ${alpha})`;
  }

  function frameCachePut(frame, payload) {
    state.frameCache.set(frame, payload);
    while (state.frameCache.size > 400) {
      state.frameCache.delete(state.frameCache.keys().next().value);
    }
  }

  function clearExplorer(message) {
    setText(ui.trackingStatus, message, message);
    for (const element of [
      ui.ballState,
      ui.ballPosition,
      ui.ballSpeed,
      ui.teamSpeed,
      ui.nearPlayer,
      ui.ballEdge,
      ui.ballGap,
      ui.touchState,
      ui.eventState,
    ]) {
      element.textContent = "—";
    }
  }

  function updateExplorer() {
    const data = state.data;
    if (!data) return;
    if (!data.tracking_available) {
      clearExplorer("Seguimiento no disponible: faltan los datos gated o el ATD.");
      return;
    }
    const frameData = state.frameData;
    if (!frameData || frameData.frame !== state.frame) {
      const loading = `frame ${formatFrame(state.frame)} · cargando seguimiento…`;
      setText(ui.trackingStatus, loading, loading);
      return;
    }
    const ball = frameData.ball || {};
    const features = frameData.features || {};
    const status = `frame ${formatFrame(frameData.frame)} · ${frameData.players.length} jugadores · ${layerSummary(frameData, false, false)}`;
    setText(ui.trackingStatus, status, status);
    ui.ballState.textContent = `${sourceText(ball.source)}${ball.out_of_play ? " · fuera" : ""}`;
    ui.ballPosition.textContent =
      ball.ex === null || ball.ex === undefined
        ? "—"
        : `${numberText(ball.ex, 2)} , ${numberText(ball.ey, 2)} m`;
    ui.ballSpeed.textContent = numberText(features.ball_speed, 2, " m/s");
    ui.teamSpeed.textContent = numberText(features.team_speed, 2, " m/s");
    ui.nearPlayer.textContent =
      features.near_distance === null || features.near_distance === undefined
        ? "—"
        : `${teamText(features.near_team)} · ${numberText(features.near_distance, 2, " m")}`;
    ui.ballEdge.textContent =
      features.edge === null || features.edge === undefined
        ? "—"
        : numberText(features.edge, 2, " m");
    ui.ballGap.textContent =
      features.gap === null || features.gap === undefined
        ? "—"
        : features.gap > 0
          ? `${numberText(features.gap / data.fps, 2, " s")} (${features.gap} f)`
          : "sin hueco";
    ui.touchState.textContent =
      features.last_touch === null || features.last_touch === undefined || features.last_touch < 0
        ? "sin toque"
        : `${teamText(features.last_touch)} · ${numberText(features.last_touch_age / data.fps, 2, " s")}`;
    const events = frameData.active_events || [];
    ui.eventState.textContent =
      events.length === 0
        ? "juego abierto"
        : events
            .map((event) =>
              event.type === "goal"
                ? `GOL · ${event.kind}`
                : `PARADA · ${event.trigger} · ${formatFrame(event.start)}–${formatFrame(event.end)}${
                    event.restart === null || event.restart === undefined
                      ? ""
                      : ` · saque ${formatFrame(event.restart)}`
                  }`,
            )
            .join(" · ");
  }

  function requestFrameData(frame) {
    const data = state.data;
    if (!data || !data.tracking_available) {
      updateExplorer();
      return;
    }
    const cached = state.frameCache.get(frame);
    if (cached) {
      state.frameData = cached;
      updateExplorer();
      updateFusionPanel();
      return;
    }
    if (state.frameData && state.frameData.frame === frame) {
      updateExplorer();
      return;
    }
    if (state.frameDataRequest) state.frameDataRequest.abort();
    const controller = new AbortController();
    state.frameDataRequest = controller;
    const loading = `frame ${formatFrame(frame)} · cargando seguimiento…`;
    setText(ui.trackingStatus, loading, loading);
    fetch(`/api/frame?frame=${frame}`, { signal: controller.signal, cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        frameCachePut(frame, payload);
        if (state.frame !== frame) return;
        state.frameData = payload;
        state.frameDataError = false;
        updateExplorer();
        updateFusionPanel();
        drawOverlay();
      })
      .catch((error) => {
        if (error && error.name === "AbortError") return;
        if (state.frame !== frame) return;
        state.frameDataError = true;
        clearExplorer(`No se pudo cargar el seguimiento del frame (${error instanceof Error ? error.message : String(error)}).`);
      })
      .finally(() => {
        if (state.frameDataRequest === controller) state.frameDataRequest = null;
      });
  }

  function fitOverlay() {
    const rect = ui.video.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * dpr);
    const height = Math.round(rect.height * dpr);
    if (ui.overlay.width !== width || ui.overlay.height !== height) {
      ui.overlay.width = width;
      ui.overlay.height = height;
    }
    const ctx = ui.overlay.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function fitPitchMap() {
    const rect = ui.pitchMap.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * dpr);
    const height = Math.round(rect.height * dpr);
    if (ui.pitchMap.width !== width || ui.pitchMap.height !== height) {
      ui.pitchMap.width = width;
      ui.pitchMap.height = height;
    }
    const ctx = ui.pitchMap.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function visibleBallLayers() {
    return BALL_LAYERS.filter((layer) => state.layers[layer]);
  }

  function layerTrail(frameData, layer, coordinates) {
    if (!frameData) return [];
    const trails = frameData.trails && frameData.trails[layer];
    if (trails && Array.isArray(trails[coordinates])) return trails[coordinates];
    if (layer === "combined") {
      const legacy = coordinates === "pitch" ? frameData.trail_pitch : frameData.trail;
      return Array.isArray(legacy) ? legacy : [];
    }
    return [];
  }

  function drawLayerTrails(ctx, frameData, xOf, yOf, coordinates) {
    for (const layer of visibleBallLayers()) {
      const points = layerTrail(frameData, layer, coordinates);
      if (points.length < 2) continue;
      ctx.save();
      ctx.lineWidth = layer === "combined" ? 1.8 : 1.3;
      for (let index = 1; index < points.length; index++) {
        const previous = points[index - 1];
        const current = points[index];
        const alpha = 0.1 + (0.45 * index) / points.length;
        ctx.strokeStyle = layerColor(layer, current[2], alpha);
        ctx.beginPath();
        ctx.moveTo(xOf(previous[0]), yOf(previous[1]));
        ctx.lineTo(xOf(current[0]), yOf(current[1]));
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  function drawLayerBall(ctx, x, y, layer, data, scale = 1) {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !data) return;
    const source = Number(data.source);
    if (source === 2) return;
    const outside = Boolean(data.out_of_play);
    const color = outside ? "#ff6b5f" : layerColor(layer, source, 0.96);
    if (layer === "raw") {
      ctx.beginPath();
      ctx.arc(x, y, 2.6 * scale, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
      ctx.stroke();
      return;
    }
    if (layer === "yolo") {
      const radius = 3.4 * scale;
      const recovered = Boolean(data.recovered);
      ctx.save();
      if (recovered) {
        ctx.strokeStyle = "rgba(255, 200, 87, 0.95)";
        ctx.fillStyle = "rgba(255, 200, 87, 0.95)";
        ctx.setLineDash([2.5, 2]);
      }
      ctx.beginPath();
      ctx.rect(x - radius, y - radius, radius * 2, radius * 2);
      ctx.lineWidth = recovered ? 1.6 : 1.3;
      ctx.strokeStyle = recovered ? "rgba(255, 200, 87, 0.95)" : color;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(x, y, 1.1 * scale, 0, Math.PI * 2);
      ctx.fillStyle = recovered ? "rgba(255, 200, 87, 0.95)" : color;
      ctx.fill();
      ctx.restore();
      return;
    }
    ctx.save();
    if (layer === "fusion") {
      // Diamond for the consensus/fused ball; it should read as a decision, not a detection.
      const radius = 8.6 * scale;
      ctx.beginPath();
      ctx.moveTo(x, y - radius);
      ctx.lineTo(x + radius, y);
      ctx.lineTo(x, y + radius);
      ctx.lineTo(x - radius, y);
      ctx.closePath();
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 2.6 * scale, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
      return;
    }
    const radius = (layer === "video" ? 7.2 : 8.2) * scale;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.lineWidth = layer === "combined" || layer === "video" ? 2 : 1.7;
    if (source === 1) ctx.setLineDash([3, 2]);
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.setLineDash([]);
    if (layer === "video" && source === 3) {
      // ATD anchor: a small tick above the ring marks a video track that is currently pinned.
      ctx.beginPath();
      ctx.moveTo(x, y - radius - 2.5 * scale);
      ctx.lineTo(x, y - radius + 2.5 * scale);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.3;
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(x, y, 2.2 * scale, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
  }

  function updatePitchStatus(frameData) {
    const data = state.data;
    if (!data) {
      ui.pitchEmpty.textContent = "Cargando datos del partido…";
      ui.pitchEmpty.classList.remove("hidden");
      setText(ui.pitchStatus, "Cargando seguimiento…", "Cargando seguimiento…");
      return;
    }
    if (!data.tracking_available) {
      ui.pitchEmpty.textContent = "Seguimiento no disponible: faltan el ATD o sus metadatos.";
      ui.pitchEmpty.classList.remove("hidden");
      setText(ui.pitchStatus, "Seguimiento no disponible", "Seguimiento no disponible");
      return;
    }
    if (!frameData) {
      // Do not flash a loading overlay on every ordinary frame fetch.  During playback the
      // previous map stays visible until the matching payload arrives.
      if (!state.frameData) {
        ui.pitchEmpty.textContent = `Cargando frame ${formatFrame(state.frame)}…`;
        ui.pitchEmpty.classList.remove("hidden");
      } else {
        ui.pitchEmpty.classList.add("hidden");
      }
      const loading = `frame ${formatFrame(state.frame)} · cargando seguimiento…`;
      setText(ui.pitchStatus, loading, loading);
      return;
    }

    ui.pitchEmpty.classList.add("hidden");
    const status = `frame ${formatFrame(frameData.frame)} · ${frameData.players.length} slots · ${layerSummary(frameData, true)}`;
    setText(ui.pitchStatus, status, status);
    ui.pitchMap.setAttribute(
      "aria-label",
      `Vista cenital frame ${frameData.frame}: ${frameData.players.length} slots, ${layerSummary(frameData)}.`,
    );
  }

  function drawYoloPlayerMarkers(ctx, players, xOf, yOf, coordinates, radius) {
    if (!state.yoloPlayersEnabled || !Array.isArray(players) || players.length === 0) return;
    ctx.save();
    for (const player of players) {
      if (player.official) continue; // referees are not players: no ghost dots on the markings
      const px = coordinates === "video" ? player.px : player.x;
      const py = coordinates === "video" ? player.py : player.y;
      if (px === null || px === undefined || py === null || py === undefined) continue;
      if (!Number.isFinite(Number(px)) || !Number.isFinite(Number(py))) continue;
      const x = xOf(Number(px));
      const y = yOf(Number(py));
      const color = player.team === 0 ? COLORS.H : player.team === 1 ? COLORS.A : "#c8d6e5";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.58;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y - radius - 1.5);
      ctx.lineTo(x, y - radius + 1.5);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 1.1, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.78;
      ctx.fill();
    }
    ctx.restore();
  }

  // Decision codes mirror BALL_CAND_* in possession_lab/perception.py.  The accepted winner is drawn
  // by drawLayerBall, so this overlay only draws the *other* candidates to make the physical gate
  // visible: what YOLO saw, and why the tracker kept or discarded each one.
  const YOLO_CAND = {
    low: 0,
    accepted: 1,
    recovered: 2,
    unreachable: 3,
    occluded: 4,
    unselected: 5,
  };

  function drawBallCandidates(ctx, frameData, xOf, yOf, coordinates, scale = 1) {
    if (!state.layers.yolo || !state.yoloCandidatesEnabled) return;
    const candidates = frameData && frameData.yolo_candidates;
    if (!Array.isArray(candidates) || candidates.length === 0) return;
    ctx.save();
    ctx.lineWidth = 1.2;
    for (const c of candidates) {
      const cx = coordinates === "video" ? c.px : c.x;
      const cy = coordinates === "video" ? c.py : c.y;
      if (cx === null || cx === undefined || cy === null || cy === undefined) continue;
      if (!Number.isFinite(Number(cx)) || !Number.isFinite(Number(cy))) continue;
      const x = xOf(Number(cx));
      const y = yOf(Number(cy));
      const r = 3.2 * scale;
      if (c.decision === YOLO_CAND.accepted || c.decision === YOLO_CAND.recovered) continue;
      if (c.decision === YOLO_CAND.unreachable) {
        ctx.strokeStyle = "rgba(255, 107, 95, 0.9)";
        ctx.beginPath();
        ctx.moveTo(x - r, y - r);
        ctx.lineTo(x + r, y + r);
        ctx.moveTo(x + r, y - r);
        ctx.lineTo(x - r, y + r);
        ctx.stroke();
      } else if (c.decision === YOLO_CAND.occluded) {
        // A low-confidence detection sitting on a player: kept out as a likely shirt patch.
        ctx.strokeStyle = "rgba(198, 120, 255, 0.95)";
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x - r * 0.7, y + r * 0.7);
        ctx.lineTo(x + r * 0.7, y - r * 0.7);
        ctx.stroke();
      } else if (c.decision === YOLO_CAND.unselected) {
        ctx.strokeStyle = "rgba(120, 200, 230, 0.5)";
        ctx.beginPath();
        ctx.arc(x, y, r * 0.85, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.fillStyle = "rgba(150, 160, 170, 0.55)";
        ctx.beginPath();
        ctx.arc(x, y, 1.3 * scale, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  function drawPitchMap() {
    const data = state.data;
    const frameData =
      state.frameData && state.frameData.frame === state.frame ? state.frameData : null;
    updatePitchStatus(frameData);
    if (data && data.tracking_available && !frameData) return;

    const fitted = fitPitchMap();
    if (!fitted) return;
    const { ctx, width, height } = fitted;
    ctx.clearRect(0, 0, width, height);

    const pitch = (frameData && frameData.pitch) || { length: 105.3, width: 68 };
    const length = Number(pitch.length) || 105.3;
    const pitchWidth = Number(pitch.width) || 68;
    const pad = Math.max(12, Math.min(width, height) * 0.035);
    const scale = Math.min((width - 2 * pad) / length, (height - 2 * pad) / pitchWidth);
    const drawWidth = length * scale;
    const drawHeight = pitchWidth * scale;
    const left = (width - drawWidth) / 2;
    const top = (height - drawHeight) / 2;
    const xOf = (x) => left + x * scale;
    const yOf = (y) => top + y * scale;
    const line = "rgba(220, 238, 228, 0.56)";

    ctx.fillStyle = "#08160e";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#0d2919";
    ctx.fillRect(left, top, drawWidth, drawHeight);
    for (let stripe = 0; stripe < 12; stripe += 2) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.018)";
      ctx.fillRect(left + (stripe * drawWidth) / 12, top, drawWidth / 12, drawHeight);
    }

    ctx.save();
    ctx.strokeStyle = line;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(left + 0.5, top + 0.5, drawWidth - 1, drawHeight - 1);
    ctx.beginPath();
    ctx.moveTo(xOf(length / 2), top);
    ctx.lineTo(xOf(length / 2), top + drawHeight);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(xOf(length / 2), yOf(pitchWidth / 2), 9.15 * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(xOf(length / 2), yOf(pitchWidth / 2), 0.7 * scale, 0, Math.PI * 2);
    ctx.fillStyle = line;
    ctx.fill();

    const box = (x, boxWidth, boxHeight) => {
      ctx.strokeRect(
        xOf(x),
        yOf((pitchWidth - boxHeight) / 2),
        boxWidth * scale,
        boxHeight * scale,
      );
    };
    box(0, 16.5, 40.32);
    box(length - 16.5, 16.5, 40.32);
    box(0, 5.5, 18.32);
    box(length - 5.5, 5.5, 18.32);
    for (const spotX of [11, length - 11]) {
      ctx.beginPath();
      ctx.arc(xOf(spotX), yOf(pitchWidth / 2), 0.45 * scale, 0, Math.PI * 2);
      ctx.fillStyle = line;
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(240, 248, 255, 0.85)";
    ctx.lineWidth = 2.4;
    for (const goalX of [0, length]) {
      ctx.beginPath();
      ctx.moveTo(xOf(goalX), yOf(pitchWidth / 2 - 3.66));
      ctx.lineTo(xOf(goalX), yOf(pitchWidth / 2 + 3.66));
      ctx.stroke();
    }
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, drawWidth, drawHeight);
    ctx.clip();

    drawLayerTrails(ctx, frameData, xOf, yOf, "pitch");
    drawBallCandidates(ctx, frameData, xOf, yOf, "pitch", 1);

    const playerRadius = clamp(width * 0.0042, 3.5, 6);
    for (const player of (frameData && frameData.players) || []) {
      const x = xOf(player.x);
      const y = yOf(player.y);
      if (x < left - 8 || x > left + drawWidth + 8 || y < top - 8 || y > top + drawHeight + 8) continue;
      const color = player.team === 0 ? COLORS.H : player.team === 1 ? COLORS.A : "#c8d6e5";
      ctx.beginPath();
      ctx.arc(x, y, playerRadius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.72)";
      ctx.stroke();
    }
    drawYoloPlayerMarkers(
      ctx,
      frameData && frameData.video_players,
      xOf,
      yOf,
      "pitch",
      playerRadius * 0.92,
    );

    ctx.restore();

    // Draw ball markers after the pitch clip so an out-of-play ball remains visible in the
    // padding around the pitch instead of disappearing at the touchline.
    const markerScale = clamp(width / 620, 0.8, 1.25);
    for (const layer of visibleBallLayers()) {
      const layerInfo = layerData(frameData, layer);
      if (
        !layerInfo ||
        layerInfo.source === 2 ||
        layerInfo.x === null ||
        layerInfo.y === null ||
        !Number.isFinite(Number(layerInfo.x)) ||
        !Number.isFinite(Number(layerInfo.y))
      ) {
        continue;
      }
      drawLayerBall(
        ctx,
        xOf(layerInfo.x),
        yOf(layerInfo.y),
        layer,
        layerInfo,
        markerScale,
      );
    }

    const lengthLabel = length.toFixed(1).replace(".", ",");
    const widthLabel = pitchWidth.toFixed(1).replace(".", ",");
    drawText(ctx, `x: 0 → ${lengthLabel} m`, left + 5, top + 12, {
      size: 8,
      weight: 500,
      color: "rgba(220, 238, 228, 0.48)",
    });
    drawText(ctx, `y: 0 → ${widthLabel} m`, left + 5, top + drawHeight - 7, {
      size: 8,
      weight: 500,
      color: "rgba(220, 238, 228, 0.48)",
    });
  }

  function drawOverlay() {
    drawPitchMap();
    const fitted = fitOverlay();
    if (!fitted) return;
    const { ctx, width, height } = fitted;
    const frameData =
      state.frameData && state.frameData.frame === state.frame ? state.frameData : null;
    // During playback the matching payload can arrive a few milliseconds after the video frame.
    // Keep the previous drawing in that interval instead of clearing to an empty overlay.
    if (state.data && state.data.tracking_available && !frameData) return;
    ctx.clearRect(0, 0, width, height);
    if (!state.overlayEnabled || !state.data || !state.data.tracking_available || !frameData) return;
    const scaleX = width / (frameData.video_width || 1920);
    const scaleY = height / (frameData.video_height || 1080);
    const xOf = (px) => px * scaleX;
    const yOf = (py) => py * scaleY;

    drawLayerTrails(ctx, frameData, xOf, yOf, "video");
    drawBallCandidates(ctx, frameData, xOf, yOf, "video", 1);

    for (const player of frameData.players || []) {
      if (player.px === null || player.px === undefined || player.py === null || player.py === undefined) {
        continue;
      }
      const color =
        player.team === 0 ? COLORS.H : player.team === 1 ? COLORS.A : "#c8d6e5";
      ctx.beginPath();
      ctx.arc(xOf(player.px), yOf(player.py), 3.6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.82;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.stroke();
    }
    drawYoloPlayerMarkers(
      ctx,
      frameData.video_players,
      xOf,
      yOf,
      "video",
      4.4,
    );

    for (const layer of visibleBallLayers()) {
      const layerInfo = layerData(frameData, layer);
      if (
        !layerInfo ||
        layerInfo.source === 2 ||
        layerInfo.px === null ||
        layerInfo.py === null ||
        !Number.isFinite(Number(layerInfo.px)) ||
        !Number.isFinite(Number(layerInfo.py))
      ) {
        continue;
      }
      drawLayerBall(
        ctx,
        xOf(layerInfo.px),
        yOf(layerInfo.py),
        layer,
        layerInfo,
        0.9,
      );
    }

    const combined = layerData(frameData, "combined");
    if (state.layers.combined && (!combined || combined.source === 2)) {
      const label = "balón perdido";
      ctx.save();
      ctx.font = `600 11px ${MONO}`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(8,16,24,0.72)";
      ctx.fillRect(10, 10, textWidth + 14, 22);
      ctx.fillStyle = "#8ca0b4";
      ctx.fillText(label, 17, 25);
      ctx.restore();
    }
  }

  function updateUTStatus(prefix) {
    const data = state.data;
    if (!data) return;
    const frames = Number(data.ut_labeled_frames) || 0;
    let anchor = "";
    if (state.utAnchor !== null) {
      anchor = state.utAnchor >= data.n_frames ? " · ancla al final" : ` · ancla en ${formatFrame(state.utAnchor)}`;
    }
    ui.utStatus.textContent = `${prefix || ""}UT: ${formatFrame(frames)} frames etiquetados${anchor}`;
  }

  function utAnchorFrames() {
    const segments = state.data ? state.data.ut_segments : null;
    if (state.utAnchorsSrc !== segments) {
      const frames = new Set();
      for (const [start, stop] of segments || []) {
        frames.add(start);
        frames.add(stop);
      }
      state.utAnchors = Array.from(frames).sort((a, b) => a - b);
      state.utAnchorsSrc = segments;
    }
    return state.utAnchors;
  }

  function frameToX(canvas, source, frame) {
    const rect = canvas.getBoundingClientRect();
    if (source === "overview") return (frame / state.data.n_frames) * rect.width;
    const { start, stop } = detailWindow();
    return ((frame - start) / Math.max(1, stop - start)) * rect.width;
  }

  function findDraggableAnchor(clientX, clientY, source) {
    if (!state.data) return null;
    const canvas = source === "overview" ? ui.overview : ui.detail;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const top = source === "overview" ? 54 : 82;
    const bottom = source === "overview" ? 82 : 118;
    const y = clientY - rect.top;
    if (y < top || y > bottom) return null;
    const x = clientX - rect.left;
    let best = null;
    let bestDistance = 7;
    for (const frame of utAnchorFrames()) {
      const distance = Math.abs(frameToX(canvas, source, frame) - x);
      if (distance <= bestDistance) {
        best = { frame, pending: false };
        bestDistance = distance;
      }
    }
    if (state.utAnchor !== null) {
      const distance = Math.abs(frameToX(canvas, source, state.utAnchor) - x);
      if (distance < bestDistance) best = { frame: state.utAnchor, pending: true };
    }
    return best;
  }

  function drawAnchors(ctx, xOf, top, bottom, width) {
    const frames = utAnchorFrames();
    ctx.save();
    for (const frame of frames) {
      const x = xOf(frame);
      if (x < -4 || x > width + 4) continue;
      ctx.strokeStyle = "rgba(232,240,247,0.85)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, top);
      ctx.lineTo(x + 0.5, bottom);
      ctx.stroke();
      ctx.fillStyle = "#e8f0f7";
      ctx.beginPath();
      ctx.moveTo(x - 3, top - 5);
      ctx.lineTo(x + 3, top - 5);
      ctx.lineTo(x, top);
      ctx.closePath();
      ctx.fill();
    }

    const pending = state.utAnchor;
    if (pending !== null && !frames.includes(pending)) {
      const x = xOf(pending);
      if (x >= -4 && x <= width + 4) {
        ctx.strokeStyle = "rgba(255,200,87,0.9)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x + 0.5, top);
        ctx.lineTo(x + 0.5, bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#ffc857";
        ctx.beginPath();
        ctx.moveTo(x - 3, top - 5);
        ctx.lineTo(x + 3, top - 5);
        ctx.lineTo(x, top);
        ctx.closePath();
        ctx.fill();
      }
    }

    if (state.anchorDrag && state.anchorDrag.from !== state.anchorDrag.to) {
      const x = xOf(state.anchorDrag.to);
      if (x >= -4 && x <= width + 4) {
        ctx.strokeStyle = "rgba(255,200,87,0.95)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(x + 0.5, top);
        ctx.lineTo(x + 0.5, bottom);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  function spellStarts() {
    return state.data.pred_spells.map((spell) => spell[0]);
  }

  function groupConsecutive(values) {
    const runs = [];
    for (const value of values || []) {
      const last = runs[runs.length - 1];
      if (last && value === last[1]) last[1] = value + 1;
      else runs.push([value, value + 1]);
    }
    return runs;
  }

  function updateURL(frame) {
    if (!Number.isFinite(frame)) return;
    const url = new URL(window.location.href);
    url.searchParams.set("frame", String(frame));
    window.history.replaceState(null, "", url);
  }

  function isInsidePublic(frame) {
    return Boolean(state.data) && frame >= state.data.public[0] && frame < state.data.public[1];
  }

  function probabilityBars(probability) {
    const bars = document.createElement("div");
    bars.className = "source-bars";
    for (let index = 0; index < 3; index++) {
      const value = Math.max(0, Math.min(1, Number(probability[index]) || 0));
      const segment = document.createElement("i");
      segment.className = `bar-${LABELS[index]}`;
      segment.style.flex = `${value} 1 0`;
      bars.appendChild(segment);
    }
    return bars;
  }

  function updateFusionPanel() {
    const data = state.data;
    if (!data) return;
    const frameData =
      state.frameData && state.frameData.frame === state.frame ? state.frameData : null;
    const fusion = frameData && frameData.fusion ? frameData.fusion : null;

    if (!fusion || !fusion.available) {
      if (data.fusion_available && state.fusionRendered) {
        // The payload for this frame is still in flight while playback already moved:
        // keep the last rendered panel. Blanking it every tick is what made the card
        // flicker (same anti-flash rule as updatePitchStatus for the pitch map).
        return;
      }
      setBadge(ui.fusionBadge, "X");
      ui.fusionName.textContent = data.fusion_available ? "Cargando…" : "Sin cache";
      setBadge(ui.fusedBadge, "X");
      ui.fusedName.textContent = data.fusion_available ? "Cargando…" : "Sin cache";
      ui.fusionConfidence.textContent = "—";
      ui.fusionAgreement.textContent = data.fusion_available
        ? "Frame de fusión cargando"
        : "Cache de fusión no cargado · ejecuta `fusion` y reinicia el viewer";
      ui.fusionAgreement.className = "agreement na";
      ui.fusionStatus.textContent = data.fusion_available
        ? `frame ${formatFrame(state.frame)} · cargando poll…`
        : "Fusión no disponible";
      ui.fusionSummary.textContent = data.fusion_available
        ? "Esperando los votos del frame."
        : "No hay votos que mostrar.";
      ui.sourceList.replaceChildren();
      const empty = document.createElement("div");
      empty.className = "fusion-empty";
      empty.textContent = data.fusion_available
        ? "Cargando fuentes…"
        : "Ejecuta `python -m possession_lab fusion` para generar outputs/fusion.npz.";
      ui.sourceList.appendChild(empty);
      return;
    }

    state.fusionRendered = true;
    const code = LABELS[fusion.code] || "X";
    setBadge(ui.fusionBadge, code);
    ui.fusionName.textContent = NAMES[code] || "—";
    setBadge(ui.fusedBadge, code);
    ui.fusedName.textContent = NAMES[code] || "—";
    ui.fusionConfidence.textContent = `${(Number(fusion.confidence || 0) * 100).toFixed(0)}%`;
    const baseline = fusion.baseline_label || null;
    const mainPred = codeAt(data.labels, state.frame);
    let agreementText;
    let agreementClass;
    if (!baseline) {
      agreementText = "HMM base no disponible en este frame";
      agreementClass = "agreement na";
    } else if (baseline === code) {
      agreementText = `✓ fusión coincide con el HMM base (${baseline})`;
      agreementClass = "agreement ok";
    } else {
      agreementText = `⚠ fusión difiere del HMM base (${baseline} → ${code})`;
      agreementClass = "agreement warn";
    }
    if (mainPred && mainPred !== code) {
      agreementText += ` · predicción principal ${mainPred}`;
    }
    ui.fusionAgreement.textContent = agreementText;
    ui.fusionAgreement.className = agreementClass;
    const sourceCount = Array.isArray(fusion.sources) ? fusion.sources.length : 0;
    setText(
      ui.fusionStatus,
      `frame ${formatFrame(state.frame)} · ${sourceCount} fuentes · modo ${
        fusion.combine_mode || "log"
      }${baseline ? ` · HMM base ${baseline}` : ""}`,
    );

    const probability = fusion.fused_probability || [0, 0, 0];
    const summary = [
      `H ${(Number(probability[0]) * 100).toFixed(0)}%`,
      `A ${(Number(probability[1]) * 100).toFixed(0)}%`,
      `D ${(Number(probability[2]) * 100).toFixed(0)}%`,
      fusion.event_dead ? "evento D" : "",
    ].filter(Boolean).join(" · ");
    ui.fusionSummary.replaceChildren(probabilityBars(probability));
    const summaryText = document.createElement("span");
    summaryText.textContent = summary;
    ui.fusionSummary.appendChild(summaryText);

    ui.sourceList.replaceChildren();
    for (const source of fusion.sources || []) {
      const row = document.createElement("div");
      row.className = "source-row";
      const statusName = source.status_name || "unknown";
      if (statusName === "abstain") row.classList.add("abstain");
      if (statusName === "estimated") row.classList.add("estimated");
      if (statusName === "event") row.classList.add("event");

      const head = document.createElement("div");
      head.className = "source-row-head";
      const name = document.createElement("span");
      name.className = "source-name";
      name.textContent = source.label || source.key || "fuente";
      const status = document.createElement("span");
      status.className = "source-status";
      status.textContent = SOURCE_STATUS_NAMES[statusName] || statusName;
      head.append(name, status);

      const weight = document.createElement("span");
      weight.className = "source-weight";
      const numericWeight = Number(source.weight || 0);
      if (source.role === "prior") weight.textContent = "prior";
      else if (source.role === "override") weight.textContent = "override";
      else if (source.role === "redundant") weight.textContent = `diagnóstico ${numericWeight.toFixed(2)}`;
      else weight.textContent = numericWeight >= 0.001 ? `peso ${numericWeight.toFixed(2)}` : "peso 0";
      weight.title = `Peso efectivo ${numericWeight.toFixed(4)}`;

      row.append(head, weight, probabilityBars(source.probability || [0, 0, 0]));
      const reason = document.createElement("span");
      reason.className = "source-reason";
      reason.textContent = source.reason || "";
      reason.title = source.reason || "";
      row.appendChild(reason);
      ui.sourceList.appendChild(row);
    }
    if (!ui.sourceList.childElementCount) {
      const empty = document.createElement("div");
      empty.className = "fusion-empty";
      empty.textContent = "El poll no tiene fuentes activas en este frame.";
      ui.sourceList.appendChild(empty);
    }
  }

  function updateUI() {
    const data = state.data;
    if (!data || state.frame < 0) return;
    const frame = state.frame;
    const pred = codeAt(data.labels, frame);
    const gt = codeAt(data.gt, frame);
    const inside = isInsidePublic(frame) && gt !== null && gt !== "X";
    const seconds = frame / data.fps;

    ui.frameNumber.textContent = formatFrame(frame);
    ui.timecode.textContent = `${formatTime(seconds)} · ${seconds.toFixed(3)} s`;
    ui.clock.textContent = formatTime(seconds);
    ui.frameMeta.textContent = `frame ${formatFrame(frame)} / ${formatFrame(data.n_frames - 1)}`;

    setBadge(ui.predBadge, pred);
    ui.predName.textContent = NAMES[codeClass(pred)] || "—";

    if (data.gt_available && gt !== null) {
      setBadge(ui.gtBadge, gt);
      ui.gtName.textContent = NAMES[codeClass(gt)] || "—";
    } else {
      setBadge(ui.gtBadge, "X");
      ui.gtName.textContent = data.gt_available ? "—" : "GT no cargada";
    }

    if (!data.gt_available) {
      ui.agreement.textContent = "GT pública no cargada";
      ui.agreement.className = "agreement na";
    } else if (!inside) {
      ui.agreement.textContent = "GT pública no disponible en este frame";
      ui.agreement.className = "agreement na";
    } else if (pred === gt) {
      ui.agreement.textContent = "✓ coincide con la GT pública";
      ui.agreement.className = "agreement ok";
    } else {
      ui.agreement.textContent = `✗ no coincide · GT ${gt} · pred ${pred}`;
      ui.agreement.className = "agreement bad";
    }

    const ut = utCodeAt(frame);
    setBadge(ui.utBadge, ut);
    ui.utName.textContent = ut === "X" ? "Sin etiqueta" : NAMES[ut] || "—";
    if (ut === "X") {
      ui.utAgreement.textContent = "UT sin etiqueta en este frame";
      ui.utAgreement.className = "agreement na";
    } else if (pred === ut) {
      ui.utAgreement.textContent = "✓ coincide con la UT";
      ui.utAgreement.className = "agreement ok";
    } else {
      ui.utAgreement.textContent = `✗ no coincide · UT ${ut} · pred ${pred}`;
      ui.utAgreement.className = "agreement bad";
    }

    const spell = findInterval(data.pred_spells, frame);
    if (spell) {
      ui.spellText.textContent = `${spell[2]} · ${formatFrame(spell[0])}–${formatFrame(spell[1] - 1)}`;
      ui.spellDuration.textContent = `${((spell[1] - spell[0]) / data.fps).toFixed(2)} s`;
    } else {
      ui.spellText.textContent = "Fuera de una posesión H/A";
      ui.spellDuration.textContent = "—";
    }

    const tag = findTag(frame);
    if (tag) {
      const team = TEAM_NAMES[String(tag.team)] || "N/A";
      ui.tagText.textContent = `${tag.code || tag.type || "tag"} · ${team}`;
      const extras = Object.entries(tag.labels || {})
        .filter(([key]) => key !== "Team")
        .map(([key, value]) => `${key}: ${value}`)
        .join(" · ");
      ui.tagDetail.textContent = extras || tag.type || "—";
    } else {
      ui.tagText.textContent = "—";
      ui.tagDetail.textContent = "Sin smart tag en este frame";
    }

    if (document.activeElement !== ui.frameInput) {
      ui.frameInput.value = String(frame);
    }
    updateFusionPanel();
  }

  function renderMetrics() {
    const data = state.data;
    if (!data) return;
    if (data.metrics) {
      const m = data.metrics;
      ui.metricsCard.classList.remove("hidden");
      ui.metrics.innerHTML = [
        ["Spell F1", m.spell_f1],
        ["Macro F1", m.macro_f1],
        ["D-F1", m.d_f1],
        ["Accuracy", m.accuracy],
        ["Spells pred.", m.n_pred_spells],
        ["Spells GT", m.n_gt_spells],
      ].map(([label, value]) => `<div>${label} <strong>${metricText(value)}</strong></div>`).join("");
    }

    const utMetrics = data.ut_metrics;
    if (utMetrics) {
      ui.utMetrics.innerHTML = [
        ["Spell F1", utMetrics.spell_f1],
        ["Macro F1", utMetrics.macro_f1],
        ["D-F1", utMetrics.d_f1],
        ["Accuracy", utMetrics.accuracy],
        ["Frames UT", data.ut_labeled_frames],
        ["Spells UT", utMetrics.n_gt_spells],
        ["Spells pred.", utMetrics.n_pred_spells],
      ].map(([label, value]) => `<div>${label} <strong>${metricText(value)}</strong></div>`).join("");
    } else {
      ui.utMetrics.textContent = "Sin etiquetas UT.";
    }

    updateSwitchDistance();
    autoSelectSwitch();
  }

  function drawTimelines() {
    if (!state.data) return;
    drawOverview();
    drawDetail();
  }

  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * dpr);
    const height = Math.round(rect.height * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function paintSegments(ctx, segments, xOf, y, height, width, alpha = 1) {
    if (!segments || segments.length === 0) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    for (const [start, stop, code] of segments) {
      const x1 = xOf(start);
      const x2 = xOf(stop);
      if (x2 < 0 || x1 > width) continue;
      const left = Math.max(0, x1);
      const right = Math.min(width, x2);
      if (right <= left) continue;
      ctx.fillStyle = COLORS[LABELS[code]] || COLORS.X;
      ctx.fillRect(left, y, Math.max(1, right - left), height);
    }
    ctx.restore();
  }

  function drawText(ctx, text, x, y, options = {}) {
    const {
      size = 10,
      weight = 600,
      color = "#8ca0b4",
      align = "left",
      baseline = "middle",
    } = options;
    ctx.save();
    ctx.font = `${weight} ${size}px ${MONO}`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  function drawEventMarkers(ctx, xOf, width, height, source) {
    const data = state.data;
    if (!data) return;
    const minGap = data.fps * (source === "overview" ? 1.0 : 0.4);
    const gapLimit = source === "overview" ? 600 : 2000;
    let gapCount = 0;
    ctx.save();
    for (const gap of data.ball_gaps || []) {
      if (gapCount >= gapLimit) break;
      if (gap[1] - gap[0] < minGap) continue;
      const x1 = xOf(gap[0]);
      const x2 = xOf(gap[1]);
      if (x2 < 0 || x1 > width) continue;
      ctx.strokeStyle = "rgba(140,160,180,0.22)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.max(0, x1) + 0.5, 0);
      ctx.lineTo(Math.max(0, x1) + 0.5, height);
      ctx.stroke();
      gapCount += 1;
    }
    for (const event of data.events || []) {
      if (event.type === "stoppage") {
        const x = xOf(event.start);
        if (x >= 0 && x <= width) {
          ctx.strokeStyle = "rgba(255,200,87,0.5)";
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 4]);
          ctx.beginPath();
          ctx.moveTo(x + 0.5, 0);
          ctx.lineTo(x + 0.5, height);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        const end = xOf(event.end);
        if (end >= 0 && end <= width) {
          ctx.strokeStyle = "rgba(255,200,87,0.28)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(end + 0.5, 0);
          ctx.lineTo(end + 0.5, height);
          ctx.stroke();
          ctx.fillStyle = "rgba(255,200,87,0.6)";
          ctx.fillRect(end - 1.5, height - 4, 3, 4);
        }
        const kick =
          event.restart === null || event.restart === undefined ? null : xOf(event.restart);
        if (kick !== null && kick >= 0 && kick <= width) {
          ctx.strokeStyle = "rgba(140,214,255,0.35)";
          ctx.lineWidth = 1;
          ctx.setLineDash([1, 3]);
          ctx.beginPath();
          ctx.moveTo(kick + 0.5, 0);
          ctx.lineTo(kick + 0.5, height);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "rgba(140,214,255,0.9)";
          ctx.beginPath();
          ctx.moveTo(kick, 1);
          ctx.lineTo(kick + 3.5, 5);
          ctx.lineTo(kick, 9);
          ctx.lineTo(kick - 3.5, 5);
          ctx.closePath();
          ctx.fill();
        }
      } else if (event.type === "goal") {
        const x = xOf(event.frame);
        if (x < 0 || x > width) continue;
        ctx.strokeStyle = "rgba(78,230,139,0.8)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x + 0.5, 0);
        ctx.lineTo(x + 0.5, height);
        ctx.stroke();
        ctx.fillStyle = "#4ee68b";
        ctx.beginPath();
        ctx.moveTo(x - 4, 0);
        ctx.lineTo(x + 4, 0);
        ctx.lineTo(x, 6);
        ctx.closePath();
        ctx.fill();
      }
    }
    ctx.restore();
  }

  function drawOverview() {
    const fitted = fitCanvas(ui.overview);
    if (!fitted) return;
    const { ctx, width, height } = fitted;
    const data = state.data;
    const n = data.n_frames;
    const xOf = (frame) => (frame / n) * width;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#081018";
    ctx.fillRect(0, 0, width, height);

    for (const [start, stop] of data.halves) {
      ctx.fillStyle = "rgba(255,255,255,0.025)";
      ctx.fillRect(xOf(start), 0, Math.max(1, xOf(stop) - xOf(start)), height);
    }

    const publicX1 = xOf(data.public[0]);
    const publicX2 = xOf(data.public[1]);
    ctx.fillStyle = "rgba(78,168,255,0.08)";
    ctx.fillRect(publicX1, 0, Math.max(1, publicX2 - publicX1), height);

    paintSegments(ctx, data.pred_segments, xOf, 12, 22, width);
    if (data.gt_available) {
      paintSegments(ctx, data.gt_segments, xOf, 40, 16, width, 0.9);
    } else {
      ctx.fillStyle = "rgba(96,114,134,0.18)";
      ctx.fillRect(0, 40, width, 16);
      drawText(ctx, "GT no cargada", 8, 48, { color: "#6d8093", size: 9 });
    }
    ctx.fillStyle = "rgba(96,114,134,0.12)";
    ctx.fillRect(0, 60, width, 16);
    paintSegments(ctx, data.ut_segments, xOf, 60, 16, width);
    drawAnchors(ctx, xOf, 60, 76, width);
    drawEventMarkers(ctx, xOf, width, height, "overview");
    drawDecisionMarkers(ctx, xOf, width, "overview");

    drawText(ctx, "PRED", 6, 8, { size: 8, color: "#8ca0b4" });
    drawText(ctx, "GT", 6, 36, { size: 8, color: "#8ca0b4" });
    drawText(ctx, "UT", 6, 56, { size: 8, color: "#8ca0b4" });

    const tickFrames = 15 * 60 * data.fps;
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    for (let frame = 0; frame <= n; frame += tickFrames) {
      const x = xOf(frame);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
      if (frame < n) {
        drawText(ctx, formatShort(frame / data.fps), x + 4, height - 10, {
          size: 8,
          color: "#607286",
        });
      }
    }

    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(78,168,255,0.55)";
    for (const x of [publicX1, publicX2]) {
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    }
    ctx.restore();

    if (publicX2 - publicX1 > 120) {
      drawText(ctx, "GT pública", publicX1 + 6, 88, {
        size: 8,
        color: "rgba(168,207,243,0.8)",
      });
    }

    if (state.hoverFrame !== null && state.hoverSource === "overview") {
      const hx = xOf(state.hoverFrame);
      ctx.strokeStyle = "rgba(255,255,255,0.32)";
      ctx.beginPath();
      ctx.moveTo(hx + 0.5, 0);
      ctx.lineTo(hx + 0.5, height);
      ctx.stroke();
    }

    const x = xOf(state.frame);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(x - 4, 0);
    ctx.lineTo(x + 4, 0);
    ctx.lineTo(x, 6);
    ctx.closePath();
    ctx.fill();
  }

  function detailWindow() {
    const data = state.data;
    const span = 40 * data.fps;
    let start = state.frame - Math.floor(span / 2);
    start = clamp(start, 0, Math.max(0, data.n_frames - span));
    return { start, stop: Math.min(data.n_frames, start + span) };
  }

  function drawDetail() {
    const fitted = fitCanvas(ui.detail);
    if (!fitted) return;
    const { ctx, width, height } = fitted;
    const data = state.data;
    const { start, stop } = detailWindow();
    const span = Math.max(1, stop - start);
    const xOf = (frame) => ((frame - start) / span) * width;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#081018";
    ctx.fillRect(0, 0, width, height);

    for (const [halfStart, halfStop] of data.halves) {
      const x1 = xOf(halfStart);
      const x2 = xOf(halfStop);
      if (x2 < 0 || x1 > width) continue;
      ctx.fillStyle = "rgba(255,255,255,0.025)";
      ctx.fillRect(Math.max(0, x1), 0, Math.max(1, Math.min(width, x2) - Math.max(0, x1)), height);
    }

    const publicX1 = xOf(data.public[0]);
    const publicX2 = xOf(data.public[1]);
    if (publicX2 >= 0 && publicX1 <= width) {
      ctx.fillStyle = "rgba(78,168,255,0.08)";
      ctx.fillRect(
        Math.max(0, publicX1),
        0,
        Math.max(1, Math.min(width, publicX2) - Math.max(0, publicX1)),
        height,
      );
    }

    const secondsStart = Math.ceil(start / data.fps);
    const secondsStop = Math.floor(stop / data.fps);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    for (let second = secondsStart; second <= secondsStop; second++) {
      const frame = second * data.fps;
      const x = xOf(frame);
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
      if (second % 5 === 0) {
        drawText(ctx, formatShort(second), x + 3, height - 9, {
          size: 8,
          color: "#607286",
        });
      }
    }

    paintSegments(ctx, data.pred_segments, xOf, 10, 30, width);
    if (data.gt_available) {
      paintSegments(ctx, data.gt_segments, xOf, 52, 24, width, 0.9);
    } else {
      ctx.fillStyle = "rgba(96,114,134,0.18)";
      ctx.fillRect(0, 52, width, 24);
    }
    ctx.fillStyle = "rgba(96,114,134,0.12)";
    ctx.fillRect(0, 88, width, 24);
    paintSegments(ctx, data.ut_segments, xOf, 88, 24, width);
    drawAnchors(ctx, xOf, 88, 112, width);
    drawEventMarkers(ctx, xOf, width, height, "detail");
    drawDecisionMarkers(ctx, xOf, width, "detail");

    drawText(ctx, "PRED", 6, 18, { size: 8, color: "rgba(232,240,247,0.72)" });
    drawText(ctx, data.gt_available ? "GT" : "GT no cargada", 6, 60, {
      size: 8,
      color: data.gt_available ? "rgba(232,240,247,0.72)" : "#6d8093",
    });
    drawText(ctx, "UT", 6, 96, { size: 8, color: "rgba(232,240,247,0.72)" });

    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(78,168,255,0.55)";
    for (const x of [publicX1, publicX2]) {
      if (x < 0 || x > width) continue;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    }
    ctx.restore();

    if (state.hoverFrame !== null && state.hoverSource === "detail") {
      const hx = xOf(state.hoverFrame);
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx + 0.5, 0);
      ctx.lineTo(hx + 0.5, height);
      ctx.stroke();
    }

    const x = xOf(state.frame);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, height);
    ctx.stroke();

    drawText(
      ctx,
      `frame ${formatFrame(state.frame)} · ${formatTime(state.frame / data.fps)}`,
      width - 6,
      10,
      { size: 9, align: "right", color: "rgba(232,240,247,0.72)" },
    );
  }

  function renderFrame(frame, { force = false } = {}) {
    if (!state.data) return;
    const next = clamp(Math.round(frame), 0, state.data.n_frames - 1);
    const changed = next !== state.frame;
    state.frame = next;
    updateUI();
    requestFrameData(next);
    if (changed || force) {
      drawTimelines();
      if (!state.data.tracking_available || (state.frameData && state.frameData.frame === next)) {
        drawOverlay();
      }
    }
  }

  function renderFromTime(seconds) {
    if (!state.data || state.scrubbing || !Number.isFinite(seconds)) return;
    const next = frameFromTime(seconds);
    if (next !== state.frame) {
      renderFrame(next);
    }
  }

  function seekToFrame(frame, { pause = false, updateHistory = true } = {}) {
    if (!state.data) return;
    const target = clamp(Math.round(frame), 0, state.data.n_frames - 1);
    if (pause) state.video.pause();
    if (state.video.readyState >= 1) {
      const seconds = timeForFrame(target);
      if (Math.abs(state.video.currentTime - seconds) > 0.0005) {
        try {
          state.video.currentTime = seconds;
        } catch (error) {
          // The video element can reject a seek while metadata is still settling.
        }
      }
    } else {
      state.pendingSeekFrame = target;
    }
    renderFrame(target, { force: true });
    if (updateHistory) updateURL(target);
  }

  function jump(delta) {
    seekToFrame(state.frame + delta, { pause: true });
  }

  function jumpToValue(value) {
    if (value === null || value === undefined) return;
    seekToFrame(value, { pause: true });
  }

  function updateHover(frame, source) {
    if (!state.data) return;
    state.hoverFrame = frame;
    state.hoverSource = source;
    if (frame === null) {
      setText(ui.hoverInfo, "—");
      return;
    }
    const pred = codeAt(state.data.labels, frame);
    const gt = codeAt(state.data.gt, frame);
    const ut = utCodeAt(frame);
    const gtText = gt !== null && gt !== "X" ? ` · GT ${gt}` : "";
    const utText = ut !== "X" ? ` · UT ${ut}` : "";
    setText(ui.hoverInfo, `frame ${formatFrame(frame)} · ${formatTime(frame / state.data.fps)} · pred ${pred || "—"}${gtText}${utText}`);
  }

  function eventFrame(event, source) {
    const canvas = source === "overview" ? ui.overview : ui.detail;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const fraction = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    if (source === "overview") {
      return clamp(Math.floor(fraction * state.data.n_frames), 0, state.data.n_frames - 1);
    }
    // During a scrub the window is frozen at pointerdown: the detail window follows
    // state.frame, so re-deriving it mid-gesture would map pixels against a window
    // the previous seek already moved (double jumps, self-referential drags).
    const { start, stop } = state.scrubWindow || detailWindow();
    return clamp(start + Math.floor(fraction * (stop - start)), 0, state.data.n_frames - 1);
  }

  function scheduleScrub(frame) {
    state.pendingScrubFrame = frame;
    if (state.scrubRaf !== null) return;
    state.scrubRaf = requestAnimationFrame(() => {
      state.scrubRaf = null;
      const target = state.pendingScrubFrame;
      state.pendingScrubFrame = null;
      if (target !== null && state.scrubbing) {
        seekToFrame(target, { pause: true, updateHistory: false });
      }
    });
  }

  function attachScrubber(canvas, source) {
    canvas.addEventListener("pointerdown", (event) => {
      if (!state.data) return;
      event.preventDefault();
      state.pointerId = event.pointerId;
      canvas.setPointerCapture(event.pointerId);
      const hit = findDraggableAnchor(event.clientX, event.clientY, source);
      if (hit) {
        state.video.pause();
        state.anchorDrag = { from: hit.frame, to: hit.frame, pending: hit.pending };
        updateHover(hit.frame, source);
        drawTimelines();
        return;
      }
      const markerHit = findMarkerHit(event.clientX, event.clientY, source);
      if (markerHit) {
        state.video.pause();
        selectSwitch(markerHit.idx, { seek: true });
        return;
      }
      state.scrubbing = true;
      state.scrubSource = source;
      state.scrubWindow = source === "overview" ? null : detailWindow();
      const frame = eventFrame(event, source);
      if (frame !== null) {
        updateHover(frame, source);
        seekToFrame(frame, { pause: true, updateHistory: false });
      }
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!state.data) return;
      const frame = eventFrame(event, source);
      if (frame === null) return;
      if (state.anchorDrag && event.pointerId === state.pointerId) {
        state.anchorDrag.to = frame;
        updateHover(frame, source);
        drawTimelines();
      } else if (state.scrubbing && event.pointerId === state.pointerId) {
        updateHover(frame, source);
        scheduleScrub(frame);
      } else if (!state.scrubbing) {
        const markerHit = findMarkerHit(event.clientX, event.clientY, source);
        canvas.style.cursor = markerHit
          ? "pointer"
          : findDraggableAnchor(event.clientX, event.clientY, source)
            ? "ew-resize"
            : "crosshair";
        if (markerHit) setText(ui.hoverInfo, describeMarkerHit(markerHit));
        else updateHover(frame, source);
        if (source === "overview") drawOverview();
        else drawDetail();
      }
    });

    const finish = (event) => {
      if (event.pointerId !== state.pointerId) return;
      const drag = state.anchorDrag;
      const wasScrubbing = state.scrubbing;
      const lastScrub = state.pendingScrubFrame;
      state.anchorDrag = null;
      state.scrubbing = false;
      state.pointerId = null;
      state.scrubSource = null;
      state.scrubWindow = null;
      if (state.scrubRaf !== null) {
        cancelAnimationFrame(state.scrubRaf);
        state.scrubRaf = null;
      }
      state.pendingScrubFrame = null;
      try {
        canvas.releasePointerCapture(event.pointerId);
      } catch (error) {
        // Pointer capture may already have been released by the browser.
      }
      if (drag) {
        if (drag.pending) {
          state.utAnchor = clamp(drag.to, 0, state.data.n_frames);
          updateUTStatus("Ancla movida. ");
          drawTimelines();
        } else if (drag.to !== drag.from) {
          drawTimelines();
          postUserTruth({ boundary: drag.from, to: drag.to });
        } else {
          drawTimelines();
        }
        return;
      }
      if (wasScrubbing) {
        if (event.type === "pointerup" && lastScrub !== null) {
          // Flush the last scrubbed frame: the press already sought, and
          // re-deriving the frame from pixels here would seek a second time.
          seekToFrame(lastScrub, { pause: true, updateHistory: false });
        }
        updateURL(state.frame);
      }
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);

    canvas.addEventListener("pointerleave", () => {
      if (state.scrubbing || state.anchorDrag) return;
      canvas.style.cursor = "crosshair";
      state.hoverFrame = null;
      state.hoverSource = null;
      updateHover(null, null);
      drawOverview();
      drawDetail();
    });
  }

  function setPlayButton() {
    ui.playBtn.textContent = state.video.paused ? "Reproducir" : "Pausa";
  }

  function setBuffering(visible, text) {
    if (text) ui.videoBufferText.textContent = text;
    ui.videoBuffer.classList.toggle("hidden", !visible);
  }

  function bufferedAhead() {
    const video = state.video;
    const time = video.currentTime;
    let ahead = 0;
    for (let index = 0; index < video.buffered.length; index++) {
      if (video.buffered.start(index) <= time + 0.5 && video.buffered.end(index) > time) {
        ahead = video.buffered.end(index) - time;
      }
    }
    return ahead;
  }

  function updateBufferedLabel() {
    const ahead = bufferedAhead();
    ui.videoBuffered.textContent = ahead > 0 ? `buffer ${formatShort(ahead)}` : "buffer —";
  }

  function togglePlay() {
    if (state.video.paused) {
      const promise = state.video.play();
      if (promise && typeof promise.catch === "function") {
        promise.catch(() => {
          ui.videoError.textContent = "El navegador bloqueó la reproducción. Pulsa Reproducir de nuevo.";
          ui.videoError.classList.remove("hidden");
        });
      }
    } else {
      state.video.pause();
    }
    setPlayButton();
  }

  function showFatal(message) {
    ui.fatalMessage.textContent = message;
    ui.fatal.classList.remove("hidden");
    setSync("error", "error");
  }

  async function postUserTruth(body, message) {
    if (!state.data || state.utSaving) return;
    state.utSaving = true;
    try {
      const response = await fetch("/api/user-truth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status} al guardar la UT`);
      Object.assign(state.data, await response.json());
      if (body.clear || body.boundary !== undefined) {
        state.utAnchor = null;
      } else if (body.stop !== undefined) {
        state.utAnchor = clamp(body.stop, 0, state.data.n_frames);
      }
      drawTimelines();
      updateUI();
      renderMetrics();
      updateUTStatus(
        message ||
          (body.clear
            ? "UT borrada. "
            : body.boundary !== undefined
              ? "Ancla movida. "
              : "Ancla añadida. ")
      );
    } catch (error) {
      ui.utStatus.textContent = `No se pudo guardar la UT: ${error instanceof Error ? error.message : String(error)}`;
    } finally {
      state.utSaving = false;
    }
  }

  function paintUserTruth(label) {
    if (!state.data || state.utSaving) return;
    const frame = state.frame;
    const anchor = state.utAnchor;
    const start = anchor === null ? frame : Math.min(anchor, frame);
    const stop = (anchor === null ? frame : Math.max(anchor, frame)) + 1;
    postUserTruth({ start, stop, label });
  }

  function toggleUserTruthAnchor() {
    if (!state.data) return;
    state.utAnchor = state.utAnchor === state.frame ? null : state.frame;
    updateUTStatus(state.utAnchor === null ? "Ancla quitada. " : "Ancla fijada. ");
  }

  function toggleUserTruthAnchor() {
    if (!state.data) return;
    state.utAnchor = state.utAnchor === state.frame ? null : state.frame;
    updateUTStatus(state.utAnchor === null ? "Ancla quitada. " : "Ancla fijada. ");
  }

  // ---------------------------------------------------------------------------
  // Decision annotations: the "why" behind every state switch. Markers are
  // clickable on both canvases; the side card lists every annotated switch
  // with timestamps.
  // ---------------------------------------------------------------------------

  const KIND_LABELS = {
    out_of_bounds: "fuera",
    parked: "parado",
    restart: "reanudación",
    teams_slow: "lentitud",
    ball_missing: "balón perdido",
    tag: "etiqueta",
    geometry: "geometría",
    prior: "prior",
  };
  const KIND_COLORS = {
    out_of_bounds: "#ff6b5f",
    parked: "#d2a8ff",
    restart: "#4ee68b",
    teams_slow: "#9ad0ff",
    ball_missing: "#8ca0b4",
    tag: "#4ea8ff",
    geometry: "#c8d6e5",
    prior: "#8ca0b4",
  };
  const SWITCH_PLACEHOLDER =
    '<p class="muted small">Sin cambio seleccionado. Haz clic en una marca <strong>│</strong> de la línea temporal o en la lista de abajo.</p>';
  const SWITCH_LIST_WINDOW = 160;

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function stamp(seconds) {
    if (!Number.isFinite(seconds)) return "—";
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
  }

  function switchListData() {
    return state.data && Array.isArray(state.data.switches) ? state.data.switches : [];
  }

  function switchMatchesFilter(sw) {
    const filter = state.switchFilter;
    if (filter === "toD") return sw.to === "D";
    if (filter === "fromD") return sw.from === "D";
    if (filter === "team") return sw.from !== "D" && sw.to !== "D";
    return true;
  }

  function filteredSwitchEntries() {
    const entries = [];
    switchListData().forEach((sw, idx) => {
      if (switchMatchesFilter(sw)) entries.push({ sw, idx });
    });
    return entries;
  }

  function distanceLine(offsetS) {
    const abs = Math.abs(offsetS);
    if (abs < 0.05) return "estás en este frame";
    return offsetS > 0 ? `este cambio está ${abs.toFixed(1)} s por delante del reproductor` : `este cambio quedó ${abs.toFixed(1)} s atrás`;
  }

  function updateSwitchDistance() {
    const el = document.getElementById("swDistance");
    const sw = state.selectedSwitch >= 0 ? switchListData()[state.selectedSwitch] : null;
    if (el && sw && state.data) el.textContent = distanceLine((sw.frame - state.frame) / state.data.fps);
  }

  function renderSwitchDetail() {
    if (!state.data) return;
    const sw = state.selectedSwitch >= 0 ? switchListData()[state.selectedSwitch] : null;
    if (sw) {
      const confPct = Math.round((Number(sw.confidence) || 0) * 100);
      const reasons = (sw.reasons || [])
        .map((r) => {
          const color = KIND_COLORS[r.kind] || "#8ca0b4";
          const label = KIND_LABELS[r.kind] || r.kind;
          const evFrame = r.evidence && Number.isFinite(r.evidence.frame) ? r.evidence.frame : null;
          const jump =
            evFrame !== null && Math.abs(evFrame - sw.frame) > 2
              ? ` <button type="button" class="mini-seek" data-seek-frame="${evFrame}" title="Ir al frame ${evFrame}">${stamp(evFrame / state.data.fps)}</button>`
              : "";
          return `<li><span class="reason-chip" style="--chip:${color}">${escapeHtml(label)}</span><span class="reason-text">${escapeHtml(r.text)}${jump}</span></li>`;
        })
        .join("");
      ui.switchDetail.innerHTML = `
        <div class="switch-head">
          <span class="badge badge-${sw.from}">${sw.from}</span>
          <span class="switch-arrow">→</span>
          <span class="badge badge-${sw.to}">${sw.to}</span>
          <button type="button" class="mini-seek stamp-btn" data-seek-frame="${sw.frame}" title="Ir a este frame">${stamp(sw.t_s)}</button>
          <span class="mono muted">f ${formatFrame(sw.frame)}</span>
        </div>
        <div class="conf-row" title="Probabilidad de emisión del estado entrante en el frame del cambio (antes del suavizado temporal)">
          <div class="conf-bar"><i style="width:${confPct}%"></i></div><span class="mono muted">${confPct}%</span>
        </div>
        <ul class="reason-list">${reasons}</ul>
        <div id="swDistance" class="muted small">${escapeHtml(distanceLine((sw.frame - state.frame) / state.data.fps))}</div>`;
      return;
    }
    ui.switchDetail.innerHTML = SWITCH_PLACEHOLDER;
  }

  function renderSwitchList() {
    const entries = filteredSwitchEntries();
    if (!entries.length) {
      ui.switchList.innerHTML = '<li class="muted small">Sin cambios con este filtro.</li>';
      return;
    }
    let center = entries.findIndex((e) => e.idx === state.selectedSwitch);
    if (center < 0) {
      center = 0;
      let best = Infinity;
      entries.forEach((e, i) => {
        const d = Math.abs(e.sw.frame - state.frame);
        if (d < best) {
          best = d;
          center = i;
        }
      });
    }
    const half = Math.floor(SWITCH_LIST_WINDOW / 2);
    const from = clamp(center - half, 0, Math.max(0, entries.length - SWITCH_LIST_WINDOW));
    const to = Math.min(entries.length, from + SWITCH_LIST_WINDOW);
    const rows = [];
    if (from > 0) rows.push(`<li class="muted small">… ${from} cambios anteriores ocultos</li>`);
    for (let i = from; i < to; i += 1) {
      const { sw, idx } = entries[i];
      rows.push(`
        <li class="switch-row${idx === state.selectedSwitch ? " selected" : ""}" data-idx="${idx}" title="${escapeHtml(sw.summary)}">
          <span class="stamp">${stamp(sw.t_s)}</span>
          <span class="sw-label"><span class="badge badge-${sw.from}">${sw.from}</span>→<span class="badge badge-${sw.to}">${sw.to}</span></span>
          <span class="sw-summary">${escapeHtml(sw.summary)}</span>
        </li>`);
    }
    if (to < entries.length) rows.push(`<li class="muted small">… ${entries.length - to} cambios posteriores ocultos</li>`);
    ui.switchList.innerHTML = rows.join("");
  }

  function renderSwitchMeta() {
    const total = switchListData().length;
    const filtered = filteredSwitchEntries().length;
    if (!total) {
      ui.switchMeta.textContent = "Sin anotaciones: ejecuta «python -m possession_lab explain» para generarlas.";
      return;
    }
    ui.switchMeta.textContent = `${filtered} de ${total} cambios anotados`;
  }

  function renderDecisions() {
    renderSwitchDetail();
    renderSwitchList();
    renderSwitchMeta();
  }

  function selectSwitch(idx, { seek = false } = {}) {
    if (!Number.isInteger(idx) || idx < 0 || idx >= switchListData().length) return;
    state.selectedSwitch = idx;
    renderSwitchDetail();
    renderSwitchList();
    if (seek) seekToFrame(switchListData()[idx].frame, { pause: true });
  }

  function jumpSwitch(direction) {
    const entries = filteredSwitchEntries();
    if (!entries.length) return;
    const frames = entries.map((e) => e.sw.frame);
    const target = direction < 0 ? previousValue(frames, state.frame) : nextValue(frames, state.frame);
    if (target === null || target === undefined) return;
    const entry = entries.find((e) => e.sw.frame === target);
    if (entry) selectSwitch(entry.idx, { seek: true });
  }

  function autoSelectSwitch() {
    const frames = state.switchFrames;
    if (!frames || !frames.length || !state.data) return;
    const i = lowerBound(frames, state.frame);
    let best = -1;
    let bestDist = 1.5 * state.data.fps;
    for (const j of [i - 1, i]) {
      if (j >= 0 && j < frames.length) {
        const d = Math.abs(frames[j] - state.frame);
        if (d <= bestDist) {
          best = j;
          bestDist = d;
        }
      }
    }
    if (best >= 0 && best !== state.selectedSwitch) {
      state.selectedSwitch = best;
      renderSwitchDetail();
      renderSwitchList();
    }
  }

  function findMarkerHit(clientX, clientY, source) {
    if (!state.data) return null;
    const canvas = source === "overview" ? ui.overview : ui.detail;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const switchTop = source === "overview" ? 10 : 8;
    const switchBottom = source === "overview" ? 36 : 46;
    let best = null;
    let bestDist = 6;
    if (y >= switchTop && y <= switchBottom) {
      switchListData().forEach((sw, idx) => {
        const d = Math.abs(frameToX(canvas, source, sw.frame) - x);
        if (d <= bestDist) {
          best = { kind: "switch", idx, frame: sw.frame, distance: d };
          bestDist = d;
        }
      });
    }
    return best;
  }

  function describeMarkerHit(hit) {
    const sw = switchListData()[hit.idx];
    return `cambio ${sw.from}→${sw.to} · ${stamp(sw.t_s)} · ${sw.summary}`;
  }

  function drawDecisionMarkers(ctx, xOf, width, source) {
    ctx.save();
    const tickTop = source === "overview" ? 12 : 10;
    const tickBottom = source === "overview" ? 34 : 44;
    const switches = switchListData();
    switches.forEach((sw, i) => {
      const x = xOf(sw.frame);
      if (x < -2 || x > width + 2) return;
      const selected = i === state.selectedSwitch;
      ctx.strokeStyle = selected ? "#ffc857" : "rgba(232,240,247,0.55)";
      ctx.lineWidth = selected ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, tickTop);
      ctx.lineTo(x + 0.5, tickBottom);
      ctx.stroke();
    });
    if (source === "detail" && state.selectedSwitch >= 0) {
      const sw = switches[state.selectedSwitch];
      if (sw) {
        const x = xOf(sw.frame);
        if (x >= -2 && x <= width + 2) {
          drawText(ctx, `${sw.from}→${sw.to}`, clamp(x + 5, 0, width - 30), 18, { size: 8, color: "#ffc857" });
        }
      }
    }
    ctx.restore();
  }

  function setup() {
    const data = state.data;
    document.title = `Possession Review · ${data.game}`;
    ui.frameInput.max = String(data.n_frames - 1);
    ui.videoDuration.textContent = `duración ${formatTime(data.duration_s)}`;

    const publicStart = data.public[0];
    const publicStop = data.public[1];
    ui.gtWindow.textContent = data.gt_available
      ? `${formatTime(publicStart / data.fps)} – ${formatTime(publicStop / data.fps)} · ${formatFrame(publicStop - publicStart)} frames`
      : "No disponible";

    const params = new URLSearchParams(window.location.search);
    const requested = Number.parseInt(params.get("frame") || "", 10);
    state.initialFrame = Number.isFinite(requested)
      ? clamp(requested, 0, data.n_frames - 1)
      : clamp(Number(data.start_frame) || publicStart, 0, data.n_frames - 1);
    state.frame = -1;
    state.pendingSeekFrame = state.initialFrame;
    renderFrame(state.initialFrame, { force: true });

    const disagreementStarts = groupConsecutive(data.disagreements).map((run) => run[0]);
    ui.prevDiff.disabled = !data.gt_available || disagreementStarts.length === 0;
    ui.nextDiff.disabled = !data.gt_available || disagreementStarts.length === 0;
    ui.gotoPublic.disabled = !data.gt_available;

    attachScrubber(ui.overview, "overview");
    attachScrubber(ui.detail, "detail");

    let seekTimer = null;
    const clearSeekTimer = () => {
      if (seekTimer !== null) {
        window.clearTimeout(seekTimer);
        seekTimer = null;
      }
    };

    state.video.addEventListener("loadstart", () => {
      setBuffering(true, "Cargando video…");
      ui.videoBuffered.textContent = "buffer —";
    });
    state.video.addEventListener("waiting", () => setBuffering(true, "Buffering…"));
    state.video.addEventListener("stalled", () => setBuffering(true, "Buffering…"));
    state.video.addEventListener("canplay", () => {
      clearSeekTimer();
      setBuffering(false);
    });
    state.video.addEventListener("playing", () => {
      clearSeekTimer();
      setBuffering(false);
    });
    state.video.addEventListener("seeking", () => {
      clearSeekTimer();
      seekTimer = window.setTimeout(() => setBuffering(true, "Buscando…"), 300);
    });
    state.video.addEventListener("seeked", () => {
      clearSeekTimer();
      setBuffering(false);
    });
    state.video.addEventListener("progress", updateBufferedLabel);
    state.video.addEventListener("timeupdate", updateBufferedLabel);

    state.video.addEventListener("loadedmetadata", () => {
      state.loadedMetadata = true;
      const duration = state.video.duration;
      const durationText = Number.isFinite(duration) ? formatTime(duration) : "—";
      ui.videoDuration.textContent = `duración ${durationText} · ${data.fps} fps`;
      updateBufferedLabel();
      if (Number.isFinite(duration) && Math.abs(duration - data.duration_s) > 0.15) {
        state.syncWarning = true;
        setSync("duración distinta", "warn");
      } else {
        setSync("video sincronizado", "ok");
      }
      const target = state.pendingSeekFrame === null ? state.initialFrame : state.pendingSeekFrame;
      state.pendingSeekFrame = null;
      seekToFrame(target, { pause: true, updateHistory: false });
      updateURL(target);
      setPlayButton();
    });

    state.video.addEventListener("play", setPlayButton);
    state.video.addEventListener("pause", () => {
      setPlayButton();
      updateBufferedLabel();
    });
    state.video.addEventListener("ended", setPlayButton);
    state.video.addEventListener("timeupdate", () => renderFromTime(state.video.currentTime));
    state.video.addEventListener("seeked", () => {
      if (!state.scrubbing) renderFromTime(state.video.currentTime);
    });
    state.video.addEventListener("error", () => {
      setBuffering(false);
      const code = state.video.error ? state.video.error.code : "?";
      ui.videoError.textContent = `No se pudo reproducir el MP4 táctico (código ${code}). Comprueba que el servidor y el archivo existen.`;
      ui.videoError.classList.remove("hidden");
      setSync("error de video", "error");
    });

    if (state.video.requestVideoFrameCallback) {
      const onVideoFrame = (now, metadata) => {
        if (!state.scrubbing) renderFromTime(metadata.mediaTime);
        state.video.requestVideoFrameCallback(onVideoFrame);
      };
      state.video.requestVideoFrameCallback(onVideoFrame);
    }

    ui.playBtn.addEventListener("click", togglePlay);
    ui.overlayToggle.checked = state.overlayEnabled;
    ui.overlayToggle.addEventListener("change", () => {
      state.overlayEnabled = ui.overlayToggle.checked;
      drawOverlay();
    });
    if (ui.yoloPlayersToggle) {
      ui.yoloPlayersToggle.checked = state.yoloPlayersEnabled;
      ui.yoloPlayersToggle.addEventListener("change", () => {
        state.yoloPlayersEnabled = ui.yoloPlayersToggle.checked;
        drawOverlay();
      });
    }
    if (ui.yoloCandidatesToggle) {
      ui.yoloCandidatesToggle.checked = state.yoloCandidatesEnabled;
      ui.yoloCandidatesToggle.addEventListener("change", () => {
        state.yoloCandidatesEnabled = ui.yoloCandidatesToggle.checked;
        drawOverlay();
      });
    }
    for (const toggle of ui.layerToggles) {
      const layer = toggle.dataset.layer;
      if (!Object.prototype.hasOwnProperty.call(state.layers, layer)) continue;
      state.layers[layer] = toggle.checked;
      toggle.addEventListener("change", () => {
        state.layers[layer] = toggle.checked;
        drawOverlay();
        updateExplorer();
      });
    }
    for (const button of document.querySelectorAll("[data-seek]")) {
      button.addEventListener("click", () => jump(Number(button.dataset.seek)));
    }
    ui.prevChange.addEventListener("click", () => jumpToValue(previousValue(data.changes, state.frame)));
    ui.nextChange.addEventListener("click", () => jumpToValue(nextValue(data.changes, state.frame)));
    ui.prevSpell.addEventListener("click", () => jumpToValue(previousValue(spellStarts(), state.frame)));
    ui.nextSpell.addEventListener("click", () => jumpToValue(nextValue(spellStarts(), state.frame)));
    ui.prevDiff.addEventListener("click", () => jumpToValue(data.gt_available ? previousValue(disagreementStarts, state.frame) : null));
    ui.nextDiff.addEventListener("click", () => jumpToValue(data.gt_available ? nextValue(disagreementStarts, state.frame) : null));
    ui.gotoPublic.addEventListener("click", () => seekToFrame(publicStart, { pause: true }));
    ui.gotoStart.addEventListener("click", () => seekToFrame(0, { pause: true }));
    ui.goFrame.addEventListener("click", () => {
      const value = Number.parseInt(ui.frameInput.value, 10);
      if (Number.isFinite(value)) seekToFrame(value, { pause: true });
    });
    ui.frameInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        ui.goFrame.click();
      }
    });

    for (const button of document.querySelectorAll("[data-ut-label]")) {
      button.addEventListener("click", () => paintUserTruth(button.dataset.utLabel));
    }
    ui.utAnchor.addEventListener("click", toggleUserTruthAnchor);
    ui.utClear.addEventListener("click", () =>
      postUserTruth({ eraseNearest: state.frame }, "Tramo UT borrado. "),
    );

    state.switchFrames = switchListData().map((sw) => sw.frame);
    for (const button of ui.explainFilters.querySelectorAll("button")) {
      button.addEventListener("click", () => {
        state.switchFilter = button.dataset.which;
        for (const other of ui.explainFilters.querySelectorAll("button")) {
          other.classList.toggle("active", other === button);
        }
        renderSwitchList();
        renderSwitchMeta();
      });
    }
    ui.switchList.addEventListener("click", (event) => {
      const row = event.target instanceof Element ? event.target.closest("li[data-idx]") : null;
      if (row) selectSwitch(Number.parseInt(row.dataset.idx, 10), { seek: true });
    });
    ui.switchDetail.addEventListener("click", (event) => {
      const seekButton = event.target instanceof Element ? event.target.closest("[data-seek-frame]") : null;
      if (seekButton) {
        const frame = Number.parseInt(seekButton.dataset.seekFrame, 10);
        if (Number.isFinite(frame)) seekToFrame(frame, { pause: true });
      }
    });
    renderDecisions();

    window.addEventListener("keydown", (event) => {
      const target = event.target;
      const typing =
        target instanceof Element && target.closest("textarea, select, input:not([type='number'])");
      if (!typing && !event.ctrlKey && !event.metaKey && !event.altKey) {
        const key = event.key.length === 1 ? event.key.toUpperCase() : "";
        if (!event.repeat && ["H", "A", "D", "X"].includes(key)) {
          event.preventDefault();
          paintUserTruth(key);
          return;
        }
      }
      if (target instanceof Element && target.closest("input, textarea, select, button")) return;
      if (event.code === "Space") {
        event.preventDefault();
        togglePlay();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const delta = (event.shiftKey ? data.fps : 1) * (event.key === "ArrowLeft" ? -1 : 1);
        jump(delta);
      } else if (event.key === "[") {
        event.preventDefault();
        jumpToValue(previousValue(data.changes, state.frame));
      } else if (event.key === "]") {
        event.preventDefault();
        jumpToValue(nextValue(data.changes, state.frame));
      } else if (event.key === "j") {
        event.preventDefault();
        jumpSwitch(-1);
      } else if (event.key === "k") {
        event.preventDefault();
        jumpSwitch(1);
      } else if (event.key === ",") {
        event.preventDefault();
        jumpToValue(previousValue(spellStarts(), state.frame));
      } else if (event.key === ".") {
        event.preventDefault();
        jumpToValue(nextValue(spellStarts(), state.frame));
      } else if (event.key === "n") {
        event.preventDefault();
        jumpToValue(data.gt_available ? nextValue(disagreementStarts, state.frame) : null);
      } else if (event.key === "N") {
        event.preventDefault();
        jumpToValue(data.gt_available ? previousValue(disagreementStarts, state.frame) : null);
      } else if (event.key.toLowerCase() === "g" && data.gt_available) {
        event.preventDefault();
        seekToFrame(publicStart, { pause: true });
      } else if (event.key === "Home") {
        event.preventDefault();
        seekToFrame(0, { pause: true });
      } else if (event.key === "End") {
        event.preventDefault();
        seekToFrame(data.n_frames - 1, { pause: true });
      }
    });

    const resize = () => {
      drawTimelines();
      drawOverlay();
    };
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(resize);
      observer.observe(ui.overview.parentElement);
      observer.observe(ui.detail.parentElement);
      observer.observe(ui.video);
    }
    window.addEventListener("resize", resize);

    ui.video.src = "/video";
    ui.video.load();
    setPlayButton();
    drawTimelines();
    updateUI();
    renderMetrics();
    updateBufferedLabel();
    updateUTStatus();
  }

  async function boot() {
    try {
      const response = await fetch("/api/viewer-data", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status} al cargar /api/viewer-data`);
      state.data = await response.json();
      if (!state.data || !Array.isArray(state.data.pred_spells)) {
        throw new Error("El payload del visor no tiene el formato esperado.");
      }
      setup();
    } catch (error) {
      showFatal(error instanceof Error ? error.message : String(error));
    }
  }

  boot();
})();

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

interface GraphNode {
  id: string;
  name: string;
  sector: string;
  country: string;
  region: string;
  tier: string;
  community: number;
  degree: number;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

interface GraphPayload {
  version: string;
  computed_at: string;
  start: string;
  end: string;
  n_observations: number;
  n_communities: number;
  min_corr: number;
  top_k: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface SimNode extends d3.SimulationNodeDatum, GraphNode {
  radius?: number;
  fillColor?: string;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  weight: number;
}

const COMMUNITY_COLORS = [
  "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
  "#0891b2", "#db2777", "#65a30d", "#b91c1c", "#0d9488",
];
const FALLBACK_COLOR = "#6b7280";

function communityColor(c: number): string {
  return COMMUNITY_COLORS[c] ?? FALLBACK_COLOR;
}

function nodeRadius(n: GraphNode): number {
  // log-scale degree → 3-10px
  return 3 + Math.min(7, Math.log2(Math.max(1, n.degree)) * 1.6);
}

export default function GlobalContagionGraph() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const nodesRef = useRef<SimNode[]>([]);
  const linksRef = useRef<SimLink[]>([]);
  const transformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);
  const dprRef = useRef(typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);

  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 560 });

  // Filters
  const [country, setCountry] = useState<string>("all");
  const [sector, setSector] = useState<string>("all");
  const [minCorr, setMinCorr] = useState<number>(0.30);
  const [search, setSearch] = useState<string>("");

  // Selection
  const [selected, setSelected] = useState<SimNode | null>(null);
  const [hover, setHover] = useState<SimNode | null>(null);

  // ── Load JSON ──────────────────────────────────────────────
  useEffect(() => {
    fetch("/data/contagion_graph.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: GraphPayload) => {
        setPayload(d);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  // ── Resize observer ────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width: Math.max(400, width), height: Math.max(400, height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── Filter options derived from payload ────────────────────
  const countries = useMemo(() => {
    if (!payload) return [];
    return ["all", ...Array.from(new Set(payload.nodes.map((n) => n.country))).sort()];
  }, [payload]);
  const sectors = useMemo(() => {
    if (!payload) return [];
    return ["all", ...Array.from(new Set(payload.nodes.map((n) => n.sector))).sort()];
  }, [payload]);

  // ── Filtered subgraph ──────────────────────────────────────
  const filtered = useMemo(() => {
    if (!payload) return { nodes: [] as SimNode[], edges: [] as SimLink[] };
    const q = search.trim().toLowerCase();
    const keptNodes = payload.nodes.filter((n) => {
      if (country !== "all" && n.country !== country) return false;
      if (sector !== "all" && n.sector !== sector) return false;
      if (q && !n.id.toLowerCase().includes(q) && !n.name.toLowerCase().includes(q))
        return false;
      return true;
    });
    const keptIds = new Set(keptNodes.map((n) => n.id));
    const keptEdges = payload.edges.filter(
      (e) =>
        keptIds.has(e.source as string) &&
        keptIds.has(e.target as string) &&
        Math.abs(e.weight) >= minCorr,
    );
    return {
      nodes: keptNodes.map((n) => ({ ...n, radius: nodeRadius(n), fillColor: communityColor(n.community) })),
      edges: keptEdges.map((e) => ({ source: e.source, target: e.target, weight: e.weight })),
    };
  }, [payload, country, sector, search, minCorr]);

  // ── Build / update simulation ──────────────────────────────
  useEffect(() => {
    if (!filtered.nodes.length) {
      nodesRef.current = [];
      linksRef.current = [];
      simRef.current?.stop();
      return;
    }
    const { width, height } = dimensions;

    // Preserve positions across filter changes
    const existing = new Map(nodesRef.current.map((n) => [n.id, n]));
    const merged: SimNode[] = filtered.nodes.map((n) => {
      const e = existing.get(n.id);
      if (e) {
        e.fillColor = n.fillColor;
        e.radius = n.radius;
        return e;
      }
      return {
        ...n,
        x: width / 2 + (Math.random() - 0.5) * 200,
        y: height / 2 + (Math.random() - 0.5) * 200,
      };
    });
    nodesRef.current = merged;
    linksRef.current = filtered.edges as SimLink[];

    if (!simRef.current) {
      simRef.current = d3
        .forceSimulation<SimNode, SimLink>(merged)
        .force(
          "link",
          d3.forceLink<SimNode, SimLink>(linksRef.current)
            .id((d) => d.id)
            .distance((l) => 30 + (1 - Math.abs(l.weight)) * 60)
            .strength((l) => Math.abs(l.weight) * 0.6),
        )
        .force("charge", d3.forceManyBody().strength(-90))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide<SimNode>().radius((d) => (d.radius ?? 6) + 2))
        .alphaDecay(0.025)
        .alpha(0.7);
    } else {
      const sim = simRef.current;
      sim.nodes(merged);
      (sim.force("link") as d3.ForceLink<SimNode, SimLink>).links(linksRef.current);
      sim.force("center", d3.forceCenter(width / 2, height / 2));
      sim.alpha(0.5).restart();
    }
  }, [filtered, dimensions]);

  // ── Zoom ───────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const sel = d3.select(canvas);
    const zoom = d3
      .zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.3, 4])
      .on("zoom", (e) => {
        transformRef.current = e.transform;
      });
    sel.call(zoom);
    return () => {
      sel.on(".zoom", null);
    };
  }, []);

  // ── Render loop ────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = dprRef.current;
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    canvas.style.width = `${dimensions.width}px`;
    canvas.style.height = `${dimensions.height}px`;

    const sim = simRef.current;
    if (!sim) return;

    let raf = 0;
    const neighborIds = new Set<string>();
    if (selected) {
      for (const l of linksRef.current) {
        const s = l.source as SimNode;
        const t = l.target as SimNode;
        if (s.id === selected.id) neighborIds.add(t.id);
        else if (t.id === selected.id) neighborIds.add(s.id);
      }
    }

    const draw = () => {
      const { width, height } = dimensions;
      const t = transformRef.current;
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);
      ctx.translate(t.x, t.y);
      ctx.scale(t.k, t.k);

      // ── Edges ──────────────────────────────────
      ctx.lineCap = "round";
      const dimEdges = !!selected;
      for (const l of linksRef.current) {
        const s = l.source as SimNode;
        const tg = l.target as SimNode;
        if (typeof s.x !== "number" || typeof s.y !== "number" ||
            typeof tg.x !== "number" || typeof tg.y !== "number") continue;
        const isAdj = selected && (s.id === selected.id || tg.id === selected.id);
        const alpha = isAdj ? 0.85 : dimEdges ? 0.05 : 0.18;
        const color = l.weight >= 0 ? "120,120,140" : "220,90,90";
        ctx.strokeStyle = `rgba(${color},${alpha})`;
        ctx.lineWidth = isAdj ? 1.6 : Math.min(1.4, 0.4 + Math.abs(l.weight) * 1.2);
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(tg.x, tg.y);
        ctx.stroke();
      }

      // ── Nodes ──────────────────────────────────
      for (const n of nodesRef.current) {
        if (typeof n.x !== "number" || typeof n.y !== "number") continue;
        const r = n.radius ?? 6;
        const dim = selected && n.id !== selected.id && !neighborIds.has(n.id);
        ctx.globalAlpha = dim ? 0.18 : 1.0;
        ctx.fillStyle = n.fillColor ?? FALLBACK_COLOR;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fill();
        if (selected && n.id === selected.id) {
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 2;
          ctx.stroke();
        } else if (hover && hover.id === n.id) {
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // ── Labels for big nodes / selected / neighbors ────
      ctx.font = "11px ui-sans-serif, system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const n of nodesRef.current) {
        if (typeof n.x !== "number" || typeof n.y !== "number") continue;
        const showLabel =
          (selected && (n.id === selected.id || neighborIds.has(n.id))) ||
          (!selected && (n.degree >= 10 || (n.radius ?? 0) >= 8));
        if (!showLabel) continue;
        const r = n.radius ?? 6;
        ctx.fillStyle = "rgba(15,23,42,0.85)";
        ctx.fillText(n.id, n.x, n.y + r + 2);
      }
      ctx.restore();
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [dimensions, selected, hover]);

  // ── Hit-testing for hover/click ───────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const t = transformRef.current;
      const x = (e.clientX - rect.left - t.x) / t.k;
      const y = (e.clientY - rect.top - t.y) / t.k;
      let hit: SimNode | null = null;
      for (const n of nodesRef.current) {
        if (typeof n.x !== "number" || typeof n.y !== "number") continue;
        const r = n.radius ?? 6;
        const dx = n.x - x;
        const dy = n.y - y;
        if (dx * dx + dy * dy <= (r + 2) * (r + 2)) {
          hit = n;
          break;
        }
      }
      setHover(hit);
      canvas.style.cursor = hit ? "pointer" : "default";
    };
    const onClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const t = transformRef.current;
      const x = (e.clientX - rect.left - t.x) / t.k;
      const y = (e.clientY - rect.top - t.y) / t.k;
      for (const n of nodesRef.current) {
        if (typeof n.x !== "number" || typeof n.y !== "number") continue;
        const r = n.radius ?? 6;
        const dx = n.x - x;
        const dy = n.y - y;
        if (dx * dx + dy * dy <= (r + 2) * (r + 2)) {
          setSelected((curr) => (curr?.id === n.id ? null : n));
          return;
        }
      }
      setSelected(null);
    };
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("click", onClick);
    return () => {
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("click", onClick);
    };
  }, []);

  // ── Top-10 neighbors of selected ──────────────────────────
  const selectedNeighbors = useMemo(() => {
    if (!selected || !payload) return [];
    const out: { node: GraphNode; weight: number }[] = [];
    const byId = new Map(payload.nodes.map((n) => [n.id, n]));
    for (const e of payload.edges) {
      let other: string | null = null;
      if (e.source === selected.id) other = e.target as string;
      else if (e.target === selected.id) other = e.source as string;
      if (other) {
        const n = byId.get(other);
        if (n) out.push({ node: n, weight: e.weight });
      }
    }
    out.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    return out.slice(0, 10);
  }, [selected, payload]);

  if (loading) {
    return (
      <div className="border border-ki-border rounded p-6 bg-ki-surface-sunken">
        <span className="text-[12px] text-ki-on-surface-muted">
          Loading contagion graph<span className="cursor-blink">_</span>
        </span>
      </div>
    );
  }
  if (error || !payload) {
    return (
      <div className="border border-ki-border rounded p-6 bg-ki-surface-sunken">
        <span className="text-[12px] text-ki-on-surface-muted">
          Contagion graph unavailable: {error ?? "no data"}
        </span>
      </div>
    );
  }

  return (
    <section className="border border-ki-border rounded bg-ki-surface">
      {/* Header */}
      <div className="border-b border-ki-border px-5 py-3 flex items-baseline justify-between">
        <div>
          <div className="eyebrow text-ki-primary mb-1">Cross-Market Contagion</div>
          <h3 className="text-[15px] font-medium text-ki-on-surface">
            {payload.nodes.length} tickers · {payload.n_communities} communities · {payload.start}→{payload.end.slice(0, 10)}
          </h3>
        </div>
        <div className="text-[11px] font-data text-ki-on-surface-muted">
          {payload.n_observations.toLocaleString()} trading days · top-{payload.top_k} edges
        </div>
      </div>

      {/* Filters */}
      <div className="px-5 py-3 border-b border-ki-border flex flex-wrap items-center gap-3 text-[12px]">
        <label className="flex items-center gap-2">
          <span className="text-ki-on-surface-muted">Country</span>
          <select
            value={country}
            onChange={(e) => setCountry(e.target.value)}
            className="border border-ki-border rounded px-2 py-0.5 bg-ki-surface font-data"
          >
            {countries.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ki-on-surface-muted">Sector</span>
          <select
            value={sector}
            onChange={(e) => setSector(e.target.value)}
            className="border border-ki-border rounded px-2 py-0.5 bg-ki-surface"
          >
            {sectors.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ki-on-surface-muted">|r| ≥</span>
          <input
            type="range"
            min={0.2}
            max={0.9}
            step={0.05}
            value={minCorr}
            onChange={(e) => setMinCorr(parseFloat(e.target.value))}
            className="w-28"
          />
          <span className="font-data tabular-nums w-10">{minCorr.toFixed(2)}</span>
        </label>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search ticker or name…"
          className="border border-ki-border rounded px-2 py-0.5 ml-auto w-56 text-[12px] font-data"
        />
        <span className="text-ki-on-surface-muted text-[11px]">
          {filtered.nodes.length} nodes · {filtered.edges.length} edges
        </span>
      </div>

      {/* Body: canvas + sidebar */}
      <div className="flex flex-col lg:flex-row">
        <div ref={containerRef} className="flex-1 relative bg-ki-surface-sunken min-h-[560px]">
          <canvas ref={canvasRef} />
          {hover && !selected && (
            <div className="absolute top-3 left-3 bg-ki-surface border border-ki-border rounded px-3 py-2 text-[11px] shadow-sm pointer-events-none">
              <div className="font-medium font-data">{hover.id}</div>
              <div className="text-ki-on-surface-muted">{hover.name}</div>
              <div className="text-ki-on-surface-muted">
                {hover.country} · {hover.sector} · community #{hover.community}
              </div>
            </div>
          )}
          {/* Legend */}
          <div className="absolute bottom-3 left-3 bg-ki-surface/80 backdrop-blur border border-ki-border rounded px-3 py-2 text-[10px] flex flex-wrap gap-2 max-w-md">
            <span className="text-ki-on-surface-muted mr-1">Communities</span>
            {Array.from({ length: Math.min(payload.n_communities, COMMUNITY_COLORS.length) }).map((_, i) => (
              <span key={i} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full" style={{ background: COMMUNITY_COLORS[i] }} />#{i}
              </span>
            ))}
          </div>
        </div>

        {/* Right sidebar: selection details */}
        <aside className="lg:w-72 border-t lg:border-t-0 lg:border-l border-ki-border p-4 bg-ki-surface">
          {selected ? (
            <>
              <div className="eyebrow text-ki-primary mb-1">Selected</div>
              <div className="font-data text-[14px] font-medium">{selected.id}</div>
              <div className="text-[12px] text-ki-on-surface-muted">{selected.name}</div>
              <div className="text-[11px] text-ki-on-surface-muted mt-1">
                {selected.country} · {selected.sector} · community #{selected.community}
              </div>
              <div className="mt-4 mb-2 eyebrow">Top-10 correlated</div>
              <ul className="space-y-1">
                {selectedNeighbors.map((nb) => (
                  <li
                    key={nb.node.id}
                    className="flex items-center justify-between text-[12px] cursor-pointer hover:bg-ki-surface-sunken rounded px-1"
                    onClick={() => {
                      const sn = nodesRef.current.find((n) => n.id === nb.node.id);
                      if (sn) setSelected(sn);
                    }}
                  >
                    <span className="font-data flex items-center gap-2">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ background: communityColor(nb.node.community) }}
                      />
                      {nb.node.id}
                    </span>
                    <span className="text-ki-on-surface-muted">{nb.node.country}</span>
                    <span
                      className={`font-data tabular-nums w-10 text-right ${
                        nb.weight >= 0 ? "text-emerald-700" : "text-red-700"
                      }`}
                    >
                      {nb.weight >= 0 ? "+" : ""}{nb.weight.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ul>
              <button
                className="mt-4 text-[11px] text-ki-on-surface-muted hover:text-ki-on-surface underline"
                onClick={() => setSelected(null)}
              >
                Clear selection
              </button>
            </>
          ) : (
            <div className="text-[12px] text-ki-on-surface-muted leading-relaxed">
              Click any ticker to see its top-10 most correlated names across all global markets
              (2018–today, daily log returns).
              <br /><br />
              Communities are detected with Louvain on the |r|-weighted graph; resolution 1.1.
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

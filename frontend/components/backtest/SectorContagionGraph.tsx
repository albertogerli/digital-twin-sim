"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

interface Node {
  id: string;
  n_tickers: number;
  out_degree: number;
  in_degree: number;
}

interface Edge {
  source: string;
  target: string;
  beta: number;
  t_stat: number;
}

interface Payload {
  version: string;
  computed_at: string;
  start: string;
  end: string;
  min_beta: number;
  min_t_stat: number;
  n_observations: number;
  nodes: Node[];
  edges: Edge[];
}

interface SimNode extends d3.SimulationNodeDatum, Node {
  radius?: number;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  beta: number;
  t_stat: number;
}

const POS_COLOR = "#059669";  // emerald — positive spillover (co-move)
const NEG_COLOR = "#dc2626";  // red — negative spillover (anti-correlation)
const NODE_COLOR = "#0f172a";

function nodeRadius(n: Node): number {
  // n_tickers in [1, 41]; map log → [6, 22]
  return 6 + Math.min(16, Math.log2(Math.max(1, n.n_tickers)) * 3);
}

export default function SectorContagionGraph() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const nodesRef = useRef<SimNode[]>([]);
  const linksRef = useRef<SimLink[]>([]);
  const transformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);
  const dprRef = useRef(typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);

  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 480 });
  const [minBeta, setMinBeta] = useState<number>(0.05);
  const [selected, setSelected] = useState<SimNode | null>(null);
  const [hover, setHover] = useState<SimNode | null>(null);

  useEffect(() => {
    fetch("/data/sector_contagion.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: Payload) => {
        setPayload(d);
        setMinBeta(d.min_beta);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width: Math.max(400, width), height: Math.max(360, height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const filtered = useMemo(() => {
    if (!payload) return { nodes: [] as SimNode[], edges: [] as SimLink[] };
    const edges = payload.edges.filter((e) => Math.abs(e.beta) >= minBeta);
    const usedIds = new Set<string>();
    edges.forEach((e) => {
      usedIds.add(e.source);
      usedIds.add(e.target);
    });
    const nodes = payload.nodes
      .filter((n) => usedIds.has(n.id))
      .map((n) => ({ ...n, radius: nodeRadius(n) }));
    return {
      nodes,
      edges: edges.map((e) => ({ source: e.source, target: e.target, beta: e.beta, t_stat: e.t_stat })),
    };
  }, [payload, minBeta]);

  // ── Simulation ────────────────────────────────────────────
  useEffect(() => {
    if (!filtered.nodes.length) {
      nodesRef.current = [];
      linksRef.current = [];
      simRef.current?.stop();
      return;
    }
    const { width, height } = dimensions;
    const existing = new Map(nodesRef.current.map((n) => [n.id, n]));
    const merged: SimNode[] = filtered.nodes.map((n) => {
      const e = existing.get(n.id);
      if (e) {
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
        .force("link", d3.forceLink<SimNode, SimLink>(linksRef.current)
          .id((d) => d.id)
          .distance((l) => 70 + (1 - Math.abs(l.beta)) * 80)
          .strength((l) => Math.min(0.6, Math.abs(l.beta) * 4)))
        .force("charge", d3.forceManyBody().strength(-220))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide<SimNode>().radius((d) => (d.radius ?? 8) + 4))
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

  // ── Zoom ──
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const sel = d3.select(canvas);
    const zoom = d3.zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.4, 4])
      .on("zoom", (e) => { transformRef.current = e.transform; });
    sel.call(zoom);
    return () => { sel.on(".zoom", null); };
  }, []);

  // ── Render ──
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
    let raf = 0;
    const draw = () => {
      const { width, height } = dimensions;
      const t = transformRef.current;
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);
      ctx.translate(t.x, t.y);
      ctx.scale(t.k, t.k);

      // Edges with arrowheads
      for (const l of linksRef.current) {
        const s = l.source as SimNode;
        const tg = l.target as SimNode;
        if (typeof s.x !== "number" || typeof s.y !== "number" ||
            typeof tg.x !== "number" || typeof tg.y !== "number") continue;
        const isAdj = selected && (s.id === selected.id || tg.id === selected.id);
        const dim = !!selected && !isAdj;
        const baseAlpha = dim ? 0.06 : 0.65;
        const color = l.beta >= 0 ? POS_COLOR : NEG_COLOR;
        ctx.strokeStyle = `${color}${Math.round(baseAlpha * 255).toString(16).padStart(2, "0")}`;
        ctx.lineWidth = 0.6 + Math.min(2.4, Math.abs(l.beta) * 8);

        // Shorten line to stop at node edge
        const dx = tg.x - s.x;
        const dy = tg.y - s.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const tgRadius = (tg.radius ?? 8) + 2;
        const sRadius = (s.radius ?? 8);
        if (dist < tgRadius + sRadius) continue;
        const ux = dx / dist;
        const uy = dy / dist;
        const x1 = s.x + ux * sRadius;
        const y1 = s.y + uy * sRadius;
        const x2 = tg.x - ux * tgRadius;
        const y2 = tg.y - uy * tgRadius;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        // Arrowhead at target
        if (!dim) {
          const ah = 5;
          const angle = Math.atan2(uy, ux);
          ctx.beginPath();
          ctx.moveTo(x2, y2);
          ctx.lineTo(x2 - ah * Math.cos(angle - Math.PI / 6), y2 - ah * Math.sin(angle - Math.PI / 6));
          ctx.moveTo(x2, y2);
          ctx.lineTo(x2 - ah * Math.cos(angle + Math.PI / 6), y2 - ah * Math.sin(angle + Math.PI / 6));
          ctx.stroke();
        }
      }

      // Nodes
      for (const n of nodesRef.current) {
        if (typeof n.x !== "number" || typeof n.y !== "number") continue;
        const r = n.radius ?? 8;
        const dim = selected && n.id !== selected.id;
        ctx.globalAlpha = dim ? 0.25 : 1;
        ctx.fillStyle = NODE_COLOR;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fill();
        if (selected?.id === n.id || hover?.id === n.id) {
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        // Label always (only 14 sectors)
        ctx.font = "11px ui-sans-serif, system-ui";
        ctx.fillStyle = dim ? "rgba(15,23,42,0.4)" : "rgba(15,23,42,0.9)";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(n.id, n.x, n.y + r + 2);
      }
      ctx.restore();
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [dimensions, selected, hover]);

  // ── Hit testing ──
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
        const r = n.radius ?? 8;
        const dx = n.x - x;
        const dy = n.y - y;
        if (dx * dx + dy * dy <= (r + 3) * (r + 3)) { hit = n; break; }
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
        const r = n.radius ?? 8;
        const dx = n.x - x;
        const dy = n.y - y;
        if (dx * dx + dy * dy <= (r + 3) * (r + 3)) {
          setSelected((curr) => curr?.id === n.id ? null : n);
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

  // ── Selected sector spillovers ──
  const selectedSpillovers = useMemo(() => {
    if (!selected || !payload) return { incoming: [], outgoing: [] };
    const incoming = payload.edges
      .filter((e) => e.target === selected.id)
      .sort((a, b) => Math.abs(b.beta) - Math.abs(a.beta));
    const outgoing = payload.edges
      .filter((e) => e.source === selected.id)
      .sort((a, b) => Math.abs(b.beta) - Math.abs(a.beta));
    return { incoming, outgoing };
  }, [selected, payload]);

  if (loading) {
    return (
      <div className="border border-ki-border rounded p-6 bg-ki-surface-sunken">
        <span className="text-[12px] text-ki-on-surface-muted">
          Loading sector contagion graph<span className="cursor-blink">_</span>
        </span>
      </div>
    );
  }
  if (error || !payload) {
    return (
      <div className="border border-ki-border rounded p-6 bg-ki-surface-sunken">
        <span className="text-[12px] text-ki-on-surface-muted">
          Sector contagion graph unavailable: {error ?? "no data"}
        </span>
      </div>
    );
  }

  return (
    <section className="border border-ki-border rounded bg-ki-surface">
      <div className="border-b border-ki-border px-5 py-3 flex items-baseline justify-between">
        <div>
          <div className="eyebrow text-ki-primary mb-1">Sector Spillover · VAR(1)</div>
          <h3 className="text-[15px] font-medium text-ki-on-surface">
            {filtered.nodes.length} sectors · {filtered.edges.length} directed edges · {payload.start}→{payload.end.slice(0, 10)}
          </h3>
        </div>
        <div className="text-[11px] font-data text-ki-on-surface-muted">
          {payload.n_observations.toLocaleString()} trading days · |β|≥{payload.min_beta} & |t|≥{payload.min_t_stat}
        </div>
      </div>

      <div className="px-5 py-3 border-b border-ki-border flex items-center gap-3 text-[12px]">
        <label className="flex items-center gap-2">
          <span className="text-ki-on-surface-muted">|β| ≥</span>
          <input
            type="range"
            min={0.02}
            max={0.30}
            step={0.01}
            value={minBeta}
            onChange={(e) => setMinBeta(parseFloat(e.target.value))}
            className="w-32"
          />
          <span className="font-data tabular-nums w-10">{minBeta.toFixed(2)}</span>
        </label>
        <span className="text-ki-on-surface-muted text-[11px]">
          {filtered.nodes.length} nodes · {filtered.edges.length} edges
        </span>
        <div className="ml-auto flex items-center gap-3 text-[10px]">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5" style={{ background: POS_COLOR }} /> co-move</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5" style={{ background: NEG_COLOR }} /> anti-correlation</span>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row">
        <div ref={containerRef} className="flex-1 relative bg-ki-surface-sunken min-h-[480px]">
          <canvas ref={canvasRef} />
          {hover && !selected && (
            <div className="absolute top-3 left-3 bg-ki-surface border border-ki-border rounded px-3 py-2 text-[11px] shadow-sm pointer-events-none">
              <div className="font-medium font-data">{hover.id}</div>
              <div className="text-ki-on-surface-muted">{hover.n_tickers} tickers · out {hover.out_degree} · in {hover.in_degree}</div>
            </div>
          )}
        </div>

        <aside className="lg:w-80 border-t lg:border-t-0 lg:border-l border-ki-border p-4 bg-ki-surface">
          {selected ? (
            <>
              <div className="eyebrow text-ki-primary mb-1">Selected sector</div>
              <div className="font-data text-[14px] font-medium">{selected.id}</div>
              <div className="text-[11px] text-ki-on-surface-muted">
                {selected.n_tickers} tickers · out {selected.out_degree} · in {selected.in_degree}
              </div>

              <div className="mt-4 mb-1 eyebrow">Drives next-day (outgoing)</div>
              <ul className="space-y-1 mb-3">
                {selectedSpillovers.outgoing.slice(0, 8).map((e, i) => (
                  <li key={i} className="flex items-center justify-between text-[12px]">
                    <span className="font-data">→ {e.target}</span>
                    <span className={`font-data tabular-nums ${e.beta >= 0 ? "text-emerald-700" : "text-red-700"}`}>
                      β={e.beta >= 0 ? "+" : ""}{e.beta.toFixed(3)}
                    </span>
                  </li>
                ))}
                {selectedSpillovers.outgoing.length === 0 && (
                  <li className="text-[11px] text-ki-on-surface-muted">no significant outgoing spillovers</li>
                )}
              </ul>

              <div className="mt-4 mb-1 eyebrow">Driven by (incoming)</div>
              <ul className="space-y-1">
                {selectedSpillovers.incoming.slice(0, 8).map((e, i) => (
                  <li key={i} className="flex items-center justify-between text-[12px]">
                    <span className="font-data">← {e.source}</span>
                    <span className={`font-data tabular-nums ${e.beta >= 0 ? "text-emerald-700" : "text-red-700"}`}>
                      β={e.beta >= 0 ? "+" : ""}{e.beta.toFixed(3)}
                    </span>
                  </li>
                ))}
                {selectedSpillovers.incoming.length === 0 && (
                  <li className="text-[11px] text-ki-on-surface-muted">no significant incoming spillovers</li>
                )}
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
              Click any sector to see its directional spillovers: which sectors it
              drives the next day (outgoing), and which sectors drive it
              (incoming).
              <br /><br />
              Edges are univariate OLS lagged-β coefficients
              (r<sub>j</sub>(t) = α + β · r<sub>i</sub>(t-1)). Filtered by
              significance (|β|≥{payload.min_beta}, |t|≥{payload.min_t_stat}).
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

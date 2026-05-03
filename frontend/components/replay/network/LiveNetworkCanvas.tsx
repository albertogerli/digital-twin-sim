"use client";

import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import * as d3 from "d3";
import type { GraphSnapshot, PostImpact } from "@/lib/types";
import type { ActiveImpact } from "@/lib/replay/types";

/* ───────────────────────────────────────────────────────────
   LiveNetworkCanvas — D3 force layout rendered on Canvas.
   Drop-in replacement for the SVG-based LiveNetworkGraph.

   Why Canvas:
   - 2D rendering scales to >1000 nodes at 60fps where SVG starts
     bottlenecking on DOM mutation
   - Uses d3-quadtree for hit-testing on hover/click

   Same Props shape so CommandCenter doesn't change.
   ─────────────────────────────────────────────────────────── */

interface Props {
  snapshot: GraphSnapshot | null;
  activeAgentIds: string[];
  activeImpact: ActiveImpact | null;
  selectedPostId: string | null;
  selectedPostImpact: PostImpact | null;
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  type: string;
  position: number;
  delta: number;
  power_level: number;
  description?: string;
  category?: string;
  sentiment?: string;
  clusterSize?: number;
  radius?: number;
  fillColor?: string;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  weight: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  government: "#1e40af", opposition: "#dc2626", center: "#059669",
  institutions: "#7c3aed", judiciary: "#9333ea", legal: "#0891b2",
  media: "#d97706", academia: "#4f46e5", civil_society: "#0d9488",
  regions: "#ea580c", international: "#2563eb", public_opinion: "#f59e0b",
  governo: "#1e40af", opposizione: "#dc2626", centro: "#059669",
  istituzioni: "#7c3aed", magistratura: "#9333ea", avvocatura: "#0891b2",
  accademia: "#4f46e5", societa: "#0d9488", regioni: "#ea580c",
  ex_pm: "#6b7280", internazionale: "#2563eb", opinione_pubblica: "#f59e0b",
};

function nodeColor(n: SimNode): string {
  if (n.category && CATEGORY_COLORS[n.category]) return CATEGORY_COLORS[n.category];
  // Fallback: position color from red→gray→green
  const t = (n.position + 1) / 2;
  const r = Math.round(220 * (1 - t) + 22 * t);
  const g = Math.round(38 * (1 - t) + 163 * t);
  const b = Math.round(38 * (1 - t) + 74 * t);
  return `rgb(${r},${g},${b})`;
}

function nodeRadius(n: SimNode): number {
  // power_level (0-1) maps to 4-12px; cluster nodes a bit larger
  const base = 4 + (n.power_level || 0.3) * 8;
  return n.clusterSize && n.clusterSize > 1 ? base + 2 : base;
}

export default function LiveNetworkCanvas({
  snapshot,
  activeAgentIds,
  activeImpact,
  selectedPostId,
  selectedPostImpact,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const nodesRef = useRef<SimNode[]>([]);
  const linksRef = useRef<SimLink[]>([]);
  const transformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);
  const rafRef = useRef<number | null>(null);
  const hoverRef = useRef<SimNode | null>(null);
  const dprRef = useRef(typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);

  const [dimensions, setDimensions] = useState({ width: 600, height: 500 });
  const [hoverNode, setHoverNode] = useState<SimNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<SimNode | null>(null);

  const activeSet = useMemo(() => new Set(activeAgentIds), [activeAgentIds]);
  const influencedSet = useMemo(
    () => new Set(activeImpact?.influencedNodeIds ?? []),
    [activeImpact],
  );

  // ── Resize observer ────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── Build/update graph from snapshot ───────────────────────
  useEffect(() => {
    if (!snapshot) return;

    const { width, height } = dimensions;
    const incomingNodes: SimNode[] = snapshot.nodes.map((n) => ({
      id: n.id,
      name: n.name,
      type: n.type,
      position: n.position,
      delta: n.delta,
      power_level: n.power_level,
      description: n.description,
      category: n.category,
      sentiment: n.sentiment,
      clusterSize: n.clusterSize,
    }));

    const nodeIds = new Set(incomingNodes.map((n) => n.id));
    const incomingLinks: SimLink[] = snapshot.edges
      .filter((e) => nodeIds.has(e.source as string) && nodeIds.has(e.target as string))
      .map((e) => ({ source: e.source as string, target: e.target as string, weight: e.weight }));

    // Preserve existing positions on snapshot updates
    const existing = new Map<string, SimNode>(nodesRef.current.map((n) => [n.id, n]));
    const merged: SimNode[] = incomingNodes.map((n) => {
      const e = existing.get(n.id);
      if (e) {
        e.position = n.position;
        e.delta = n.delta;
        e.power_level = n.power_level;
        e.category = n.category;
        e.sentiment = n.sentiment;
        e.radius = nodeRadius(n);
        e.fillColor = nodeColor(n);
        return e;
      }
      return {
        ...n,
        x: width / 2 + (Math.random() - 0.5) * 100,
        y: height / 2 + (Math.random() - 0.5) * 100,
        radius: nodeRadius(n),
        fillColor: nodeColor(n),
      };
    });

    nodesRef.current = merged;
    linksRef.current = incomingLinks;

    if (!simRef.current) {
      simRef.current = d3
        .forceSimulation<SimNode, SimLink>(merged)
        .force("link", d3.forceLink<SimNode, SimLink>(incomingLinks).id((d) => d.id).distance(38).strength(0.25))
        .force("charge", d3.forceManyBody().strength(-65))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide<SimNode>().radius((d) => (d.radius ?? 6) + 2))
        .alphaDecay(0.025)
        .alpha(0.6);
    } else {
      const sim = simRef.current;
      sim.nodes(merged);
      (sim.force("link") as d3.ForceLink<SimNode, SimLink>).links(incomingLinks);
      sim.force("center", d3.forceCenter(width / 2, height / 2));
      sim.alpha(0.4).restart();
    }
  }, [snapshot, dimensions]);

  // ── Zoom + pan via d3-zoom on the canvas element ───────────
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

    let frame = 0;

    const draw = () => {
      frame++;
      const { width, height } = dimensions;
      const t = transformRef.current;

      ctx.save();
      ctx.scale(dpr, dpr);
      // Background — transparent over container's bg-ki-surface-sunken
      ctx.clearRect(0, 0, width, height);

      // Apply zoom transform
      ctx.translate(t.x, t.y);
      ctx.scale(t.k, t.k);

      // ── Draw edges ──────────────────────────────────────
      ctx.lineCap = "round";
      const edgeAlpha = activeImpact || selectedNode ? 0.10 : 0.18;
      ctx.strokeStyle = `rgba(120,120,140,${edgeAlpha})`;
      for (const l of linksRef.current) {
        const s = l.source as SimNode;
        const target = l.target as SimNode;
        if (typeof s.x !== "number" || typeof s.y !== "number" ||
            typeof target.x !== "number" || typeof target.y !== "number") continue;
        ctx.lineWidth = Math.min(1.5, 0.4 + l.weight * 1.2);
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();
      }

      // Highlight edges from active impact / selected node
      if (activeImpact?.affectedEdges?.length) {
        ctx.strokeStyle = "rgba(99,102,241,0.55)";
        ctx.lineWidth = 1.6;
        const byId = new Map(nodesRef.current.map((n) => [n.id, n]));
        for (const e of activeImpact.affectedEdges) {
          const s = byId.get(e.source);
          const tgt = byId.get(e.target);
          if (!s || !tgt ||
              typeof s.x !== "number" || typeof s.y !== "number" ||
              typeof tgt.x !== "number" || typeof tgt.y !== "number") continue;
          ctx.beginPath();
          ctx.moveTo(s.x, s.y);
          ctx.lineTo(tgt.x, tgt.y);
          ctx.stroke();
        }
      }

      // ── Draw nodes ──────────────────────────────────────
      const dim = activeImpact || selectedNode;
      for (const n of nodesRef.current) {
        if (typeof n.x !== "number" || typeof n.y !== "number") continue;
        const isHover = hoverRef.current?.id === n.id;
        const isSelected = selectedNode?.id === n.id;
        const isActive = activeSet.has(n.id);
        const isInfluenced = influencedSet.has(n.id);
        const isAuthor = activeImpact?.authorId === n.id;
        const isMuted = !!dim && !isActive && !isInfluenced && !isAuthor && !isHover && !isSelected;

        const r = (n.radius ?? 6) * (isHover ? 1.25 : isSelected ? 1.2 : 1);

        // Pulse ring on author of active impact
        if (isAuthor) {
          const pulse = 1 + 0.5 * Math.sin(frame * 0.15);
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 6 * pulse, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(99,102,241,0.18)";
          ctx.fill();
        }

        // Fill
        ctx.globalAlpha = isMuted ? 0.18 : 1;
        ctx.fillStyle = n.fillColor || "#888";
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fill();

        // Outline for active / hover / selected
        if (isActive || isHover || isSelected || isInfluenced) {
          ctx.lineWidth = isSelected || isHover ? 2 : 1.4;
          ctx.strokeStyle = isSelected
            ? "#0a0a0a"
            : isInfluenced
            ? "rgba(99,102,241,0.9)"
            : "rgba(220,38,38,0.8)";
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 1.2, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      // ── Labels for top-N hubs (power_level > 0.7) ────
      ctx.font = "11px Geist, Inter, system-ui, sans-serif";
      ctx.fillStyle = "rgba(15,17,21,0.85)";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      for (const n of nodesRef.current) {
        if ((n.power_level ?? 0) < 0.7) continue;
        if (typeof n.x !== "number" || typeof n.y !== "number") continue;
        const r = n.radius ?? 6;
        ctx.fillText(n.name.slice(0, 22), n.x + r + 4, n.y);
      }

      ctx.restore();

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [dimensions, activeImpact, selectedNode, activeSet, influencedSet]);

  // ── Hit testing on mouse move/click via quadtree ───────────
  const findNodeAt = useCallback((clientX: number, clientY: number): SimNode | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const t = transformRef.current;
    const x = (clientX - rect.left - t.x) / t.k;
    const y = (clientY - rect.top - t.y) / t.k;
    const qt = d3.quadtree<SimNode>()
      .x((d) => d.x ?? 0)
      .y((d) => d.y ?? 0)
      .addAll(nodesRef.current);
    const radius = 16 / t.k;
    return qt.find(x, y, radius) ?? null;
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const n = findNodeAt(e.clientX, e.clientY);
    if (hoverRef.current?.id !== n?.id) {
      hoverRef.current = n;
      setHoverNode(n);
    }
  }, [findNodeAt]);

  const onMouseLeave = useCallback(() => {
    hoverRef.current = null;
    setHoverNode(null);
  }, []);

  const onClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const n = findNodeAt(e.clientX, e.clientY);
    setSelectedNode((prev) => (prev?.id === n?.id ? null : n));
  }, [findNodeAt]);

  // Compute selected node screen position for the inspector card
  const inspectorPos = useMemo(() => {
    if (!selectedNode || typeof selectedNode.x !== "number" || typeof selectedNode.y !== "number") return null;
    const t = transformRef.current;
    return {
      x: Math.min(selectedNode.x * t.k + t.x + 14, dimensions.width - 240),
      y: Math.max(selectedNode.y * t.k + t.y - 80, 8),
    };
  }, [selectedNode, dimensions]);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-ki-surface-sunken overflow-hidden">
      <canvas
        ref={canvasRef}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
        onClick={onClick}
        className="absolute inset-0 cursor-crosshair"
        style={{ touchAction: "none" }}
      />

      {/* Top-left HUD */}
      <div className="absolute top-3 left-3 flex gap-2 pointer-events-none">
        <span className="inline-flex items-center gap-1.5 px-2 h-6 rounded-sm bg-ki-surface-raised border border-ki-border text-[11px] text-ki-on-surface-secondary">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="6" cy="6" r="3" /><circle cx="18" cy="18" r="3" /><circle cx="6" cy="18" r="3" />
            <line x1="6" y1="9" x2="6" y2="15" /><line x1="9" y1="6" x2="15" y2="18" />
          </svg>
          Influence network
        </span>
        <span className="inline-flex items-center px-2 h-6 rounded-sm bg-ki-surface-raised border border-ki-border font-data tabular text-[11px] text-ki-on-surface-secondary">
          {nodesRef.current.length} nodes · {linksRef.current.length} edges
        </span>
        <span className="inline-flex items-center px-2 h-6 rounded-sm bg-ki-surface-raised border border-ki-border font-data text-[11px] text-ki-on-surface-secondary">
          Canvas · d3-force
        </span>
      </div>

      {/* Hover tooltip */}
      {hoverNode && !selectedNode && (
        <div
          className="absolute pointer-events-none px-2 py-1.5 rounded-sm bg-ki-surface-raised border border-ki-border shadow-tint text-[11px]"
          style={{
            left: Math.min((hoverNode.x ?? 0) * transformRef.current.k + transformRef.current.x + 12, dimensions.width - 200),
            top: Math.max((hoverNode.y ?? 0) * transformRef.current.k + transformRef.current.y - 8, 4),
          }}
        >
          <div className="font-medium text-ki-on-surface">{hoverNode.name}</div>
          <div className="font-data tabular text-ki-on-surface-muted">
            {hoverNode.category || hoverNode.type} · pos {hoverNode.position.toFixed(2)}
          </div>
        </div>
      )}

      {/* Selected node inspector */}
      {selectedNode && inspectorPos && (
        <div
          className="absolute bg-ki-surface-raised border border-ki-border rounded shadow-tint p-3 w-[240px]"
          style={{ left: inspectorPos.x, top: inspectorPos.y }}
        >
          <div className="flex items-center justify-between">
            <span className="eyebrow">Selected</span>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-ki-on-surface-muted hover:text-ki-on-surface transition-colors"
              aria-label="Close"
            >
              <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 400" }}>close</span>
            </button>
          </div>
          <div className="text-[14px] font-medium text-ki-on-surface mt-1">{selectedNode.name}</div>
          <div className="text-[11px] text-ki-on-surface-muted">
            {selectedNode.category || selectedNode.type}
          </div>
          <div className="grid grid-cols-3 gap-3 mt-3">
            <div className="flex flex-col">
              <span className="eyebrow">Position</span>
              <span className="font-data tabular text-[14px] text-ki-on-surface">
                {selectedNode.position > 0 ? "+" : ""}{selectedNode.position.toFixed(2)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="eyebrow">Δ</span>
              <span className={`font-data tabular text-[14px] ${
                selectedNode.delta > 0 ? "text-ki-success" : selectedNode.delta < 0 ? "text-ki-error" : "text-ki-on-surface"
              }`}>
                {selectedNode.delta > 0 ? "+" : ""}{selectedNode.delta.toFixed(2)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="eyebrow">Power</span>
              <span className="font-data tabular text-[14px] text-ki-on-surface">
                {(selectedNode.power_level || 0).toFixed(2)}
              </span>
            </div>
          </div>
          {selectedNode.description && (
            <p className="text-[11px] text-ki-on-surface-secondary leading-relaxed mt-2 line-clamp-3">
              {selectedNode.description}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

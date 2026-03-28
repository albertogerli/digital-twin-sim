"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";
import type { GraphSnapshot, PostImpact } from "@/lib/types";
import type { ActiveImpact } from "@/lib/replay/types";

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
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  type: string;
  weight: number;
  is_new: boolean;
}

interface SelectedNode {
  node: SimNode;
  screenX: number;
  screenY: number;
}

// Generic category colors
const CATEGORY_COLORS: Record<string, string> = {
  government: "#1e40af",
  opposition: "#dc2626",
  center: "#059669",
  institutions: "#7c3aed",
  judiciary: "#9333ea",
  legal: "#0891b2",
  media: "#d97706",
  academia: "#4f46e5",
  civil_society: "#0d9488",
  regions: "#ea580c",
  international: "#2563eb",
  public_opinion: "#f59e0b",
  // Italian originals as fallback
  governo: "#1e40af",
  opposizione: "#dc2626",
  centro: "#059669",
  istituzioni: "#7c3aed",
  magistratura: "#9333ea",
  avvocatura: "#0891b2",
  accademia: "#4f46e5",
  societa: "#0d9488",
  regioni: "#ea580c",
  ex_pm: "#6b7280",
  internazionale: "#2563eb",
  opinione_pubblica: "#f59e0b",
};

function posColor(p: number): string {
  const t = (p + 1) / 2;
  const r = Math.round(220 * (1 - t) + 22 * t);
  const g = Math.round(38 * (1 - t) + 163 * t);
  const b = Math.round(38 * (1 - t) + 74 * t);
  return `rgb(${r},${g},${b})`;
}

export default function LiveNetworkGraph({
  snapshot,
  activeAgentIds,
  activeImpact,
  selectedPostId,
  selectedPostImpact,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 500 });
  const activeSet = useRef(new Set<string>());
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);

  activeSet.current = new Set(activeAgentIds);

  const handleNodeClick = useCallback((node: SimNode, svgEl: SVGSVGElement) => {
    const svgRect = svgEl.getBoundingClientRect();
    const transform = d3.zoomTransform(svgEl);
    const screenX = transform.applyX(node.x ?? 0);
    const screenY = transform.applyY(node.y ?? 0);
    setSelectedNode((prev) =>
      prev?.node.id === node.id
        ? null
        : { node, screenX: Math.min(screenX, svgRect.width - 180), screenY: Math.max(screenY - 80, 10) },
    );
  }, []);

  // ResizeObserver
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const parent = svg.parentElement;
    if (!parent) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);

  // Main D3 graph render
  useEffect(() => {
    if (!snapshot || !svgRef.current) return;

    const svgEl = svgRef.current;
    const svg = d3.select(svgEl);
    const { width, height } = dimensions;

    const nodeData: SimNode[] = snapshot.nodes.map((n) => ({
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

    const nodeIds = new Set(nodeData.map((n) => n.id));
    const linkData: SimLink[] = snapshot.edges
      .filter((e) => nodeIds.has(e.source as string) && nodeIds.has(e.target as string))
      .map((e) => ({
        source: e.source as string,
        target: e.target as string,
        type: e.type,
        weight: e.weight,
        is_new: false,
      }));

    if (!simRef.current) {
      simRef.current = d3
        .forceSimulation<SimNode, SimLink>()
        .force("link", d3.forceLink<SimNode, SimLink>().id((d) => d.id).distance(35).strength(0.3))
        .force("charge", d3.forceManyBody().strength(-60))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(10))
        .alphaDecay(0.02);
    }

    const sim = simRef.current;
    sim.force("center", d3.forceCenter(width / 2, height / 2));

    const existingNodes = new Map<string, SimNode>();
    (sim.nodes() || []).forEach((n) => existingNodes.set(n.id, n));

    const mergedNodes = nodeData.map((n) => {
      const existing = existingNodes.get(n.id);
      if (existing) {
        existing.position = n.position;
        existing.delta = n.delta;
        existing.power_level = n.power_level;
        existing.description = n.description;
        existing.category = n.category;
        existing.sentiment = n.sentiment;
        return existing;
      }
      return { ...n, x: width / 2 + (Math.random() - 0.5) * 100, y: height / 2 + (Math.random() - 0.5) * 100 };
    });

    sim.nodes(mergedNodes);
    (sim.force("link") as d3.ForceLink<SimNode, SimLink>).links(linkData);
    sim.alpha(0.4).restart();

    let defs = svg.select<SVGDefsElement>("defs");
    if (defs.empty()) {
      defs = svg.append("defs");
      const filter = defs.append("filter").attr("id", "live-glow");
      filter.append("feGaussianBlur").attr("stdDeviation", "2.5").attr("result", "coloredBlur");
      const merge = filter.append("feMerge");
      merge.append("feMergeNode").attr("in", "coloredBlur");
      merge.append("feMergeNode").attr("in", "SourceGraphic");

      const impactFilter = defs.append("filter").attr("id", "impact-glow");
      impactFilter.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "glow");
      const impactMerge = impactFilter.append("feMerge");
      impactMerge.append("feMergeNode").attr("in", "glow");
      impactMerge.append("feMergeNode").attr("in", "SourceGraphic");
    }

    let g = svg.select<SVGGElement>("g.graph-container");
    if (g.empty()) {
      g = svg.append("g").attr("class", "graph-container");
      g.append("g").attr("class", "links");
      g.append("g").attr("class", "nodes");
      g.append("g").attr("class", "impact-layer");
    }

    // Links
    const linkSel = g
      .select("g.links")
      .selectAll<SVGLineElement, SimLink>("line")
      .data(linkData, (d) => `${(d.source as SimNode).id || d.source}-${(d.target as SimNode).id || d.target}`);

    linkSel.exit().transition().duration(300).attr("stroke-opacity", 0).remove();

    const linkEnter = linkSel.enter().append("line").attr("stroke-opacity", 0);

    const linkMerged = linkEnter.merge(linkSel);
    linkMerged
      .transition()
      .duration(600)
      .attr("stroke", (d) => d.is_new ? "#3b82f6" : "#475569")
      .attr("stroke-opacity", (d) => Math.min(d.weight * 0.35, 0.45))
      .attr("stroke-width", (d) => Math.max(d.weight * 0.8, 0.5))
      .attr("stroke-dasharray", (d) => d.is_new ? "3,3" : "none");

    // Nodes
    const nodeSel = g
      .select("g.nodes")
      .selectAll<SVGGElement, SimNode>("g.node")
      .data(mergedNodes, (d) => d.id);

    nodeSel.exit().transition().duration(300).attr("opacity", 0).remove();

    const nodeEnter = nodeSel
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("opacity", 0)
      .style("cursor", "pointer")
      .call(
        d3.drag<SVGGElement, SimNode>()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }),
      );

    nodeEnter.on("click", (event, d) => {
      event.stopPropagation();
      handleNodeClick(d, svgEl);
    });

    // Cluster nodes get a rounded rect; others get a circle
    nodeEnter.each(function (d) {
      const el = d3.select(this);
      if (d.type === "cluster") {
        el.append("rect")
          .attr("class", "node-shape")
          .attr("rx", 4).attr("ry", 4);
      } else {
        el.append("circle").attr("class", "node-shape");
      }
    });

    // Name label
    nodeEnter.append("text")
      .attr("class", "node-label")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("fill", "white")
      .attr("font-size", "7px")
      .attr("font-weight", "bold")
      .attr("pointer-events", "none");

    const nodeMerged = nodeEnter.merge(nodeSel);

    nodeMerged.transition().duration(600).attr("opacity", 1);

    // Circles
    nodeMerged
      .filter((d) => d.type !== "cluster")
      .select("circle.node-shape")
      .transition()
      .duration(600)
      .attr("r", (d) => {
        if (d.type === "partito" || d.type === "party") return 11;
        return 5 + d.power_level * 5;
      })
      .attr("fill", (d) => posColor(d.position))
      .attr("stroke", (d) => {
        if (activeSet.current.has(d.id)) return "#3b82f6";
        return CATEGORY_COLORS[d.category ?? ""] ?? "#64748b";
      })
      .attr("stroke-width", (d) => activeSet.current.has(d.id) ? 2.5 : 1.5)
      .attr("filter", (d) => Math.abs(d.delta) > 0.1 ? "url(#live-glow)" : "none");

    // Rects (cluster nodes)
    nodeMerged
      .filter((d) => d.type === "cluster")
      .select("rect.node-shape")
      .transition()
      .duration(600)
      .attr("width", (d) => {
        const base = 20 + (d.clusterSize ?? 3000) / 300;
        return Math.min(base, 38);
      })
      .attr("height", (d) => {
        const base = 14 + (d.clusterSize ?? 3000) / 500;
        return Math.min(base, 24);
      })
      .attr("x", (d) => {
        const w = Math.min(20 + (d.clusterSize ?? 3000) / 300, 38);
        return -w / 2;
      })
      .attr("y", (d) => {
        const h = Math.min(14 + (d.clusterSize ?? 3000) / 500, 24);
        return -h / 2;
      })
      .attr("fill", (d) => posColor(d.position))
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "3,2")
      .attr("filter", (d) => Math.abs(d.delta) > 0.05 ? "url(#live-glow)" : "none");

    // Initials label
    nodeMerged
      .select("text.node-label")
      .text((d) => {
        if (d.type === "partito" || d.type === "party") return d.name.substring(0, 4);
        const parts = d.name.split(" ");
        if (parts.length >= 2) return parts[parts.length - 1].substring(0, 3);
        return d.name.substring(0, 3);
      });

    // Tooltips
    nodeMerged.select("title").remove();
    nodeMerged.append("title").text((d) =>
      `${d.name}\n${d.description ?? ""}\nPos: ${d.position.toFixed(2)} (D${d.delta > 0 ? "+" : ""}${d.delta.toFixed(2)})`
    );

    sim.on("tick", () => {
      linkMerged
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);

      nodeMerged.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on("zoom", (event) => {
        g.attr("transform", event.transform.toString());
      });

    svg.call(zoom);

    // Click on background to deselect
    svg.on("click", () => setSelectedNode(null));
  }, [snapshot, dimensions, handleNodeClick]);

  // Active agent highlighting
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg
      .selectAll<SVGGElement, SimNode>("g.node")
      .select(".node-shape")
      .attr("stroke", (d) => {
        if (activeSet.current.has(d.id)) return "#3b82f6";
        if (d.type === "cluster") return "#f59e0b";
        return CATEGORY_COLORS[d.category ?? ""] ?? "#64748b";
      })
      .attr("stroke-width", (d) => activeSet.current.has(d.id) ? 2.5 : (d.type === "cluster" ? 2 : 1.5));
  }, [activeAgentIds]);

  // Post impact animation
  useEffect(() => {
    if (!activeImpact || !svgRef.current) return;
    const svgEl = svgRef.current;
    const svg = d3.select(svgEl);
    const g = svg.select("g.graph-container");

    const authorNode = g.selectAll<SVGGElement, SimNode>("g.node")
      .filter((d) => d.id === activeImpact.authorId);

    const pulseRing = authorNode.append("circle")
      .attr("class", "pulse-ring")
      .attr("r", 8)
      .attr("fill", "none")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 2)
      .attr("opacity", 0.8);

    let pulseCount = 0;
    function animatePulse() {
      if (pulseCount >= 3) {
        pulseRing.remove();
        return;
      }
      pulseCount++;
      pulseRing
        .attr("r", 8).attr("opacity", 0.8)
        .transition().duration(700).ease(d3.easeQuadOut)
        .attr("r", 24).attr("opacity", 0)
        .on("end", animatePulse);
    }
    animatePulse();

    authorNode.select(".node-shape")
      .attr("filter", "url(#impact-glow)")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 3);

    const influenceTimer = setTimeout(() => {
      const influencedSet = new Set(activeImpact.influencedNodeIds);

      g.selectAll<SVGGElement, SimNode>("g.node")
        .filter((d) => influencedSet.has(d.id))
        .each(function (d) {
          const node = d3.select(this);
          const shift = activeImpact.nodeShifts.get(d.id) ?? 0;

          node.select(".node-shape")
            .transition().duration(200)
            .attr("stroke", shift > 0 ? "#22c55e" : "#ef4444")
            .attr("stroke-width", 2.5)
            .transition().duration(1800)
            .attr("stroke", CATEGORY_COLORS[d.category ?? ""] ?? "#64748b")
            .attr("stroke-width", 1.5);

          if (Math.abs(shift) > 0.005) {
            node.append("text")
              .attr("class", "shift-label")
              .attr("x", 14).attr("y", -6)
              .attr("font-size", "9px")
              .attr("font-family", "monospace")
              .attr("font-weight", "bold")
              .attr("fill", shift > 0 ? "#16a34a" : "#dc2626")
              .attr("opacity", 1)
              .text(shift > 0 ? `+${shift.toFixed(2)}` : shift.toFixed(2))
              .transition().delay(1800).duration(500)
              .attr("opacity", 0)
              .remove();
          }
        });

      const edgePairs = new Set(
        activeImpact.affectedEdges.map((e) => `${e.source}-${e.target}`),
      );
      g.select("g.links").selectAll<SVGLineElement, SimLink>("line")
        .filter((d) => {
          const sId = typeof d.source === "object" ? (d.source as SimNode).id : d.source;
          const tId = typeof d.target === "object" ? (d.target as SimNode).id : d.target;
          return edgePairs.has(`${sId}-${tId}`) || edgePairs.has(`${tId}-${sId}`);
        })
        .attr("stroke", "#3b82f6")
        .attr("stroke-opacity", 0.8)
        .attr("stroke-width", 2.5)
        .attr("filter", "url(#impact-glow)")
        .transition().delay(1200).duration(600)
        .attr("stroke", (d) => d.is_new ? "#3b82f6" : "#475569")
        .attr("stroke-opacity", (d) => Math.min(d.weight * 0.35, 0.45))
        .attr("stroke-width", (d) => Math.max(d.weight * 0.8, 0.5))
        .attr("filter", "none");
    }, 500);

    return () => {
      clearTimeout(influenceTimer);
      const s = d3.select(svgEl);
      s.selectAll(".pulse-ring").remove();
      s.selectAll(".shift-label").remove();
      authorNode.select(".node-shape")
        .attr("filter", "none")
        .attr("stroke", "transparent")
        .attr("stroke-width", 0);
    };
  }, [activeImpact]);

  // Post selection highlight
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const g = svg.select("g.graph-container");

    if (!selectedPostId || !selectedPostImpact) {
      g.selectAll<SVGGElement, SimNode>("g.node")
        .transition().duration(300)
        .attr("opacity", 1);
      g.select("g.links").selectAll<SVGLineElement, SimLink>("line")
        .transition().duration(300)
        .attr("stroke-opacity", (d) => Math.min(d.weight * 0.35, 0.45));
      return;
    }

    const highlightIds = new Set([
      selectedPostImpact.authorId,
      ...selectedPostImpact.influencedAgents.map((a) => a.agentId),
    ]);

    g.selectAll<SVGGElement, SimNode>("g.node")
      .transition().duration(400)
      .attr("opacity", (d) => highlightIds.has(d.id) ? 1 : 0.1);

    g.select("g.links").selectAll<SVGLineElement, SimLink>("line")
      .transition().duration(400)
      .attr("stroke-opacity", (d) => {
        const sId = typeof d.source === "object" ? (d.source as SimNode).id : d.source;
        const tId = typeof d.target === "object" ? (d.target as SimNode).id : d.target;
        return (highlightIds.has(sId as string) && highlightIds.has(tId as string)) ? 0.6 : 0.03;
      });

    g.selectAll<SVGGElement, SimNode>("g.node")
      .filter((d) => d.id === selectedPostImpact.authorId)
      .select(".node-shape")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 3.5)
      .attr("filter", "url(#impact-glow)");

    g.selectAll<SVGGElement, SimNode>("g.node")
      .filter((d) => selectedPostImpact.influencedAgents.some((a) => a.agentId === d.id))
      .select(".node-shape")
      .attr("stroke", "#60a5fa")
      .attr("stroke-width", 2);
  }, [selectedPostId, selectedPostImpact]);

  if (!snapshot) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2">
        <div className="w-12 h-12 rounded-full border border-gray-300 flex items-center justify-center">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-gray-500">
            <circle cx="12" cy="12" r="3" />
            <circle cx="5" cy="6" r="2" />
            <circle cx="19" cy="6" r="2" />
            <circle cx="5" cy="18" r="2" />
            <circle cx="19" cy="18" r="2" />
            <line x1="7" y1="7" x2="10" y2="10" />
            <line x1="17" y1="7" x2="14" y2="10" />
            <line x1="7" y1="17" x2="10" y2="14" />
            <line x1="17" y1="17" x2="14" y2="14" />
          </svg>
        </div>
        <span className="text-[10px] font-mono text-gray-400">Network waiting</span>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-1.5 border-b border-gray-200 flex items-center justify-between">
        <span className="font-mono text-[10px] text-gray-500">
          NETWORK — R{snapshot.round}
        </span>
        <span className="font-mono text-[9px] text-gray-400">
          {snapshot.nodes.filter((n) => n.type === "persona").length} agents -- {snapshot.edges.length} links
        </span>
      </div>
      <div className="flex-1 relative">
        <svg
          ref={svgRef}
          width={dimensions.width}
          height={dimensions.height}
          className="w-full h-full"
        />

        {/* Node detail panel */}
        {selectedNode && (
          <div
            className="absolute z-20 bg-white border border-gray-300 rounded-lg shadow-xl p-3 w-52 pointer-events-auto"
            style={{ left: selectedNode.screenX + 16, top: selectedNode.screenY }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 mb-2">
              <div
                className="w-6 h-6 rounded-full flex-shrink-0"
                style={{ backgroundColor: posColor(selectedNode.node.position) }}
              />
              <div className="min-w-0">
                <div className="text-xs font-semibold text-gray-900 truncate">
                  {selectedNode.node.name}
                </div>
                <div className="text-[9px] text-gray-500 truncate">
                  {selectedNode.node.description}
                </div>
              </div>
            </div>

            {selectedNode.node.category && (
              <div className="mb-2">
                <span
                  className="inline-block px-1.5 py-0.5 rounded text-[8px] font-mono font-medium text-white"
                  style={{ backgroundColor: CATEGORY_COLORS[selectedNode.node.category] ?? "#6b7280" }}
                >
                  {selectedNode.node.category}
                </span>
              </div>
            )}

            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono">
              <div className="text-gray-400">Position</div>
              <div className={`text-right font-bold ${selectedNode.node.position > 0 ? "text-green-600" : selectedNode.node.position < 0 ? "text-red-600" : "text-gray-500"}`}>
                {selectedNode.node.position > 0 ? "+" : ""}{selectedNode.node.position.toFixed(2)}
              </div>

              <div className="text-gray-400">Delta</div>
              <div className={`text-right font-bold ${selectedNode.node.delta > 0 ? "text-green-600" : selectedNode.node.delta < 0 ? "text-red-600" : "text-gray-400"}`}>
                {selectedNode.node.delta > 0 ? "+" : ""}{selectedNode.node.delta.toFixed(2)}
              </div>

              <div className="text-gray-400">{selectedNode.node.type === "cluster" ? "Engagement" : "Influence"}</div>
              <div className="text-right text-gray-700">
                {(selectedNode.node.power_level * 100).toFixed(0)}%
              </div>

              {selectedNode.node.clusterSize && (
                <>
                  <div className="text-gray-400">Population</div>
                  <div className="text-right text-gray-700">
                    {selectedNode.node.clusterSize.toLocaleString()}
                  </div>
                </>
              )}

              {selectedNode.node.sentiment && (
                <>
                  <div className="text-gray-400">Sentiment</div>
                  <div className="text-right text-gray-700 capitalize">
                    {selectedNode.node.sentiment}
                  </div>
                </>
              )}
            </div>

            <button
              onClick={() => setSelectedNode(null)}
              className="absolute top-1.5 right-1.5 w-4 h-4 flex items-center justify-center text-gray-400 hover:text-gray-700"
            >
              <svg width="8" height="8" viewBox="0 0 8 8">
                <path d="M1 1L7 7M7 1L1 7" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </button>
          </div>
        )}

        {/* Stats footer */}
        <div className="absolute bottom-2 left-2 right-2 flex justify-between font-mono text-[8px] text-gray-400">
          <span>Pol: {(snapshot.stats?.avg_position ?? 0).toFixed(1)}</span>
          <span>Avg: {(snapshot.stats?.avg_position ?? 0) > 0 ? "+" : ""}{(snapshot.stats?.avg_position ?? 0).toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}

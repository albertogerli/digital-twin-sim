"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface GraphNode {
  id: string;
  name: string;
  type: string;
  position: number;
  delta: number;
  power_level: number;
  sentiment: string;
  category: string;
  clusterSize?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  type: string;
}

interface GraphSnapshot {
  round: number;
  month: string;
  event_label: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    avg_position: number;
  };
}

interface Props {
  snapshots: GraphSnapshot[];
}

function positionColor(pos: number): string {
  if (pos > 0.3) return "#22c55e";
  if (pos > 0.1) return "#4ade80";
  if (pos > -0.1) return "#94a3b8";
  if (pos > -0.3) return "#f87171";
  return "#ef4444";
}

function sentimentBorder(sentiment: string): string {
  const map: Record<string, string> = {
    combative: "#dc2626",
    furious: "#dc2626",
    worried: "#f59e0b",
    cautious: "#f59e0b",
    satisfied: "#22c55e",
    triumphant: "#22c55e",
    optimistic: "#22c55e",
    neutral: "#6b7280",
    hopeful: "#3b82f6",
  };
  return map[sentiment?.toLowerCase()] || "#6b7280";
}

export default function NetworkGraph({ snapshots }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentRound, setCurrentRound] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 700, height: 500 });

  // Responsive — measure container and update on resize
  useEffect(() => {
    if (!containerRef.current) return;
    // Initial measurement
    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      setDimensions({ width: Math.max(400, rect.width), height: Math.max(350, rect.height) });
    }
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) {
        setDimensions({ width: Math.max(400, width), height: Math.max(350, height) });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Autoplay
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setCurrentRound((prev) => {
        if (prev >= snapshots.length - 1) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 2000);
    return () => clearInterval(timer);
  }, [playing, snapshots.length]);

  // D3 rendering
  useEffect(() => {
    if (!svgRef.current || !snapshots.length) return;
    const snapshot = snapshots[currentRound];
    if (!snapshot) return;

    const { width, height } = dimensions;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Defs for glow filter
    const defs = svg.append("defs");
    const filter = defs.append("filter").attr("id", "glow");
    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Build node map
    const nodeMap = new Map<string, GraphNode>();
    snapshot.nodes.forEach((n) => nodeMap.set(n.id, n));

    // Filter edges to valid nodes
    const validEdges = snapshot.edges.filter(
      (e) => nodeMap.has(e.source as string) && nodeMap.has(e.target as string)
    );

    // Create simulation nodes (copy to avoid mutation)
    const simNodes = snapshot.nodes.map((n) => ({
      ...n,
      x: width / 2 + (Math.random() - 0.5) * 200,
      y: height / 2 + (Math.random() - 0.5) * 200,
    }));

    const simEdges = validEdges.map((e) => ({
      ...e,
      source: e.source as string,
      target: e.target as string,
    }));

    // Force simulation
    const simulation = d3
      .forceSimulation(simNodes as any)
      .force(
        "link",
        d3
          .forceLink(simEdges as any)
          .id((d: any) => d.id)
          .distance(80)
          .strength(0.3)
      )
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d: any) =>
        d.type === "cluster" ? 8 + (d.clusterSize || 20) / 10 : 10
      ))
      .force("x", d3.forceX(width / 2).strength(0.04))
      .force("y", d3.forceY(height / 2).strength(0.04));

    // Container group with zoom
    const g = svg.append("g");
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 3])
        .on("zoom", (event) => g.attr("transform", event.transform)) as any
    );

    // Edges
    const edges = g
      .append("g")
      .selectAll("line")
      .data(simEdges)
      .join("line")
      .attr("stroke", (d: any) =>
        d.type === "cluster_influence" ? "#3b82f680" : "#64748b40"
      )
      .attr("stroke-width", (d: any) => Math.max(0.5, d.weight * 2));

    // Nodes
    const nodes = g
      .append("g")
      .selectAll("g")
      .data(simNodes)
      .join("g")
      .attr("cursor", "pointer")
      .call(
        d3
          .drag<SVGGElement, any>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }) as any
      );

    // Node circles
    nodes.each(function (d: any) {
      const el = d3.select(this);
      const isCluster = d.type === "cluster";
      const radius = isCluster ? 6 + (d.clusterSize || 20) / 8 : 5 + d.power_level * 8;
      const hasDelta = Math.abs(d.delta) > 0.05;

      if (isCluster) {
        // Rounded rect for clusters
        el.append("rect")
          .attr("x", -radius)
          .attr("y", -radius * 0.7)
          .attr("width", radius * 2)
          .attr("height", radius * 1.4)
          .attr("rx", 4)
          .attr("fill", positionColor(d.position))
          .attr("stroke", sentimentBorder(d.sentiment))
          .attr("stroke-width", 1.5)
          .attr("opacity", 0.85)
          .attr("filter", hasDelta ? "url(#glow)" : null);
      } else {
        // Circle for agents
        el.append("circle")
          .attr("r", radius)
          .attr("fill", positionColor(d.position))
          .attr("stroke", sentimentBorder(d.sentiment))
          .attr("stroke-width", 1.5)
          .attr("opacity", 0.9)
          .attr("filter", hasDelta ? "url(#glow)" : null);
      }

      // Label
      if (d.power_level > 0.4 || isCluster) {
        el.append("text")
          .attr("dy", isCluster ? radius + 12 : radius + 12)
          .attr("text-anchor", "middle")
          .attr("fill", "#4b5563")
          .attr("font-size", "9px")
          .attr("font-weight", d.power_level > 0.6 ? "600" : "400")
          .text(d.name.length > 18 ? d.name.slice(0, 16) + "..." : d.name);
      }

      // Delta indicator
      if (hasDelta) {
        const arrow = d.delta > 0 ? "▲" : "▼";
        const color = d.delta > 0 ? "#22c55e" : "#ef4444";
        el.append("text")
          .attr("dx", radius + 3)
          .attr("dy", 3)
          .attr("fill", color)
          .attr("font-size", "8px")
          .attr("font-weight", "bold")
          .text(`${arrow}${Math.abs(d.delta).toFixed(2)}`);
      }
    });

    // Tooltip
    nodes
      .on("mouseenter", function (event, d: any) {
        d3.select(this).select("circle,rect").attr("stroke-width", 3);
        // Show tooltip
        const tooltip = g
          .append("g")
          .attr("class", "tooltip")
          .attr("transform", `translate(${d.x + 15}, ${d.y - 10})`);
        const bg = tooltip
          .append("rect")
          .attr("fill", "#ffffff")
          .attr("stroke", "#e5e7eb")
          .attr("rx", 6)
          .attr("opacity", 0.95);
        const text = tooltip.append("text").attr("fill", "#1f2937").attr("font-size", "11px");
        text.append("tspan").attr("x", 8).attr("dy", 16).attr("font-weight", "bold").text(d.name);
        text.append("tspan").attr("x", 8).attr("dy", 14).text(`Position: ${d.position.toFixed(2)}`);
        text.append("tspan").attr("x", 8).attr("dy", 14).text(`Delta: ${d.delta > 0 ? "+" : ""}${d.delta.toFixed(3)}`);
        text.append("tspan").attr("x", 8).attr("dy", 14).text(`Sentiment: ${d.sentiment}`);
        if (d.clusterSize) {
          text.append("tspan").attr("x", 8).attr("dy", 14).text(`Population: ${d.clusterSize}`);
        }
        const bbox = (text.node() as SVGTextElement).getBBox();
        bg.attr("width", bbox.width + 16).attr("height", bbox.height + 12);
      })
      .on("mouseleave", function () {
        d3.select(this).select("circle,rect").attr("stroke-width", 1.5);
        g.selectAll(".tooltip").remove();
      });

    // Tick
    simulation.on("tick", () => {
      edges
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);
      nodes.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [currentRound, snapshots, dimensions]);

  if (!snapshots.length) return null;

  const snap = snapshots[currentRound];

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      {/* Header with controls */}
      <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-gray-900">Network Graph</h2>
          {snap && (
            <span className="text-xs text-gray-400">
              {snap.month} — {snap.stats.total_nodes} nodes, {snap.stats.total_edges} edges
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Play/Pause */}
          <button
            onClick={() => setPlaying(!playing)}
            className="p-1.5 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-900 transition-colors"
          >
            {playing ? (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Round selector */}
      <div className="px-4 py-2 border-b border-gray-200 flex gap-1 overflow-x-auto">
        {snapshots.map((s, i) => (
          <button
            key={i}
            onClick={() => { setCurrentRound(i); setPlaying(false); }}
            className={`px-3 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors ${
              i === currentRound
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-500 hover:bg-gray-200"
            }`}
          >
            R{s.round}
          </button>
        ))}
      </div>

      {/* SVG */}
      <div ref={containerRef} className="h-[500px] w-full relative bg-gray-50">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
          preserveAspectRatio="xMidYMid meet"
          className="w-full h-full"
        />
      </div>

      {/* Legend */}
      <div className="px-4 py-2 border-t border-gray-200 flex flex-wrap gap-4 text-xs text-gray-500">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>Pro (+)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-gray-400" />
          <span>Neutral</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>Contro (-)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-2 rounded bg-blue-500/50" />
          <span>Cluster</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-green-600 font-bold">▲</span>
          <span>Shift positivo</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-red-600 font-bold">▼</span>
          <span>Shift negativo</span>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useRef, useMemo } from "react";
import * as d3 from "d3";
import { type WgAgent, type WgRoundState } from "@/lib/wargame-types";

interface Node extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  tier: number;
  position: number;
  sentiment: string;
  cluster: string;
  radius: number;
}

interface Link extends d3.SimulationLinkDatum<Node> {
  value: number;
}

export function ContagionGraph({ agents, state }: { agents: WgAgent[]; state: WgRoundState }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<Node, Link> | null>(null);

  // Sample nodes: all elites + institutions + sample of citizens
  const { nodes, links } = useMemo(() => {
    const elites = agents.filter((a) => a.tier <= 2);
    // Sample 80 citizens for visual density
    const citizens = agents.filter((a) => a.tier === 3);
    const sampledCitizens = citizens.filter((_, i) => i % 9 === 0).slice(0, 80);
    const allAgents = [...elites, ...sampledCitizens];

    const nodesMap = new Map<string, Node>();
    const nodes: Node[] = allAgents.map((a) => {
      const n: Node = {
        id: a.id,
        name: a.name,
        tier: a.tier,
        position: a.position,
        sentiment: a.sentiment,
        cluster: a.cluster,
        radius: a.tier === 1 ? 6 : a.tier === 2 ? 4 : 2,
      };
      nodesMap.set(a.id, n);
      return n;
    });

    // Generate edges: cluster-based + random cross-cluster
    const links: Link[] = [];
    const clusters = new Map<string, Node[]>();
    for (const n of nodes) {
      if (!clusters.has(n.cluster)) clusters.set(n.cluster, []);
      clusters.get(n.cluster)!.push(n);
    }

    // Intra-cluster links
    for (const [, members] of clusters) {
      for (let i = 0; i < members.length; i++) {
        for (let j = i + 1; j < Math.min(i + 4, members.length); j++) {
          links.push({ source: members[i], target: members[j], value: 0.5 + Math.random() * 0.5 });
        }
      }
    }

    // Cross-cluster links (elite to elite, elite to institutions)
    const eliteNodes = nodes.filter((n) => n.tier === 1);
    for (let i = 0; i < eliteNodes.length; i++) {
      for (let j = i + 1; j < eliteNodes.length; j++) {
        if (Math.random() < 0.4) {
          links.push({ source: eliteNodes[i], target: eliteNodes[j], value: 0.3 });
        }
      }
    }

    return { nodes, links };
  }, [agents]);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const rect = svgRef.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    svg.selectAll("*").remove();

    const g = svg.append("g");

    // Colors: position-based (green = positive, red = negative)
    const posColor = (pos: number, sent: string) => {
      if (sent === "negative") return "#ff3b3b";
      if (sent === "positive") return "#00d26a";
      return "#6b7280";
    };

    // Edge opacity based on contagion risk
    const edgeOpacity = 0.05 + state.contagionRisk * 0.15;

    // Links
    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (d: Link) => {
        const s = d.source as Node;
        const t = d.target as Node;
        // If both negative → red edge (contagion spreading)
        if (s.sentiment === "negative" && t.sentiment === "negative") return "#ff3b3b";
        if (s.sentiment === "positive" && t.sentiment === "positive") return "#00d26a";
        return "#2a2d35";
      })
      .attr("stroke-opacity", edgeOpacity)
      .attr("stroke-width", 0.5);

    // Nodes
    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d: Node) => d.radius)
      .attr("fill", (d: Node) => posColor(d.position, d.sentiment))
      .attr("fill-opacity", (d: Node) => d.tier === 1 ? 0.9 : d.tier === 2 ? 0.7 : 0.4)
      .attr("stroke", (d: Node) => d.tier === 1 ? "#c8ccd4" : "none")
      .attr("stroke-width", (d: Node) => d.tier === 1 ? 0.5 : 0)
      .attr("cursor", (d: Node) => d.tier <= 2 ? "pointer" : "default");

    // Labels for elite agents only
    const labels = g.append("g")
      .selectAll("text")
      .data(nodes.filter((n: Node) => n.tier === 1))
      .join("text")
      .text((d: Node) => d.name.split(" ").pop() || d.name)
      .attr("font-family", "Geist Mono, monospace")
      .attr("font-size", "8px")
      .attr("fill", "#6b7280")
      .attr("text-anchor", "middle")
      .attr("dy", (d: Node) => -(d.radius + 4));

    // Tooltip
    node.on("mouseover", function (event: MouseEvent, d: Node) {
      if (d.tier > 2) return;
      d3.select(this).attr("r", d.radius * 1.5);
    }).on("mouseout", function (event: MouseEvent, d: Node) {
      d3.select(this).attr("r", d.radius);
    });

    // Force simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d: any) => d.id).distance(30).strength(0.3))
      .force("charge", d3.forceManyBody().strength((d: any) => d.tier === 1 ? -80 : d.tier === 2 ? -40 : -8))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d: any) => d.radius + 2))
      .force("x", d3.forceX(width / 2).strength(0.05))
      .force("y", d3.forceY(height / 2).strength(0.05));

    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);

      labels
        .attr("x", (d: any) => d.x)
        .attr("y", (d: any) => d.y);
    });

    simRef.current = simulation;

    // Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    return () => { simulation.stop(); };
  }, [nodes, links, state.contagionRisk]);

  return (
    <div className="flex-1 min-h-0 relative bg-ki-surface">
      <svg ref={svgRef} className="w-full h-full" />
      {/* Coalition overlay */}
      <div className="absolute bottom-0 left-0 right-0 h-8 flex items-center px-2 bg-ki-surface/80 border-t border-ki-border-strong gap-3">
        {state.coalitions.map((c) => (
          <div key={c.label} className="flex items-center gap-1">
            <span className="font-data text-[8px] text-ki-on-surface-muted">{c.label}</span>
            <span className="font-data text-[9px]" style={{
              color: c.position > 0.3 ? "#00d26a" : c.position < -0.3 ? "#ff3b3b" : "#6b7280"
            }}>{c.size}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

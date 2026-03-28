"use client";

import { useRef, useEffect } from "react";
import * as d3 from "d3";

interface Props {
  positive: number;
  neutral: number;
  negative: number;
}

export default function SentimentDonut({ positive, neutral, negative }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const width = 100;
    const height = 100;
    const radius = 40;
    const innerRadius = 28;

    const svg = d3.select(svgRef.current);
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    let g = svg.select<SVGGElement>("g.donut");
    if (g.empty()) {
      g = svg.append("g").attr("class", "donut").attr("transform", `translate(${width / 2},${height / 2})`);
    }

    const total = positive + neutral + negative;
    if (total === 0) return;

    const data = [
      { label: "Pos", value: positive / total, color: "#22c55e" },
      { label: "Neu", value: neutral / total, color: "#64748b" },
      { label: "Neg", value: negative / total, color: "#ef4444" },
    ];

    const pie = d3
      .pie<(typeof data)[0]>()
      .value((d) => d.value)
      .sort(null)
      .padAngle(0.03);

    const arc = d3
      .arc<d3.PieArcDatum<(typeof data)[0]>>()
      .innerRadius(innerRadius)
      .outerRadius(radius)
      .cornerRadius(2);

    const arcs = g.selectAll<SVGPathElement, d3.PieArcDatum<(typeof data)[0]>>("path").data(pie(data));

    arcs
      .enter()
      .append("path")
      .attr("d", arc)
      .attr("fill", (d) => d.data.color)
      .attr("opacity", 0.8)
      .merge(arcs)
      .transition()
      .duration(800)
      .attrTween("d", function (d) {
        const interpolate = d3.interpolate(
          (this as SVGPathElement & { _current?: d3.PieArcDatum<(typeof data)[0]> })._current || d,
          d,
        );
        (this as SVGPathElement & { _current?: d3.PieArcDatum<(typeof data)[0]> })._current = d;
        return (t: number) => arc(interpolate(t)) || "";
      })
      .attr("fill", (d) => d.data.color);

    // Center percentage
    let centerText = g.select<SVGTextElement>("text.center");
    if (centerText.empty()) {
      centerText = g.append("text").attr("class", "center").attr("text-anchor", "middle").attr("dy", "0.1em");
      centerText.append("tspan").attr("class", "pct").attr("fill", "#e2e8f0").attr("font-size", "12").attr("font-weight", "bold").attr("font-family", "monospace");
      centerText.append("tspan").attr("class", "lbl").attr("fill", "#94a3b8").attr("font-size", "6").attr("font-family", "monospace").attr("x", "0").attr("dy", "11");
    }
    centerText.select("tspan.pct").text(`${Math.round((positive / total) * 100)}%`);
    centerText.select("tspan.lbl").text("positive");

  }, [positive, neutral, negative]);

  return (
    <div className="flex flex-col items-center gap-1">
      <svg ref={svgRef} className="w-full max-w-[100px]" />
      <div className="flex gap-3 text-[8px] font-mono">
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
          <span className="text-gray-500">Positive</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-gray-400" />
          <span className="text-gray-500">Neutral</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
          <span className="text-gray-500">Negative</span>
        </span>
      </div>
    </div>
  );
}

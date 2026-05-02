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

    const cs = getComputedStyle(document.documentElement);
    const cPos = cs.getPropertyValue("--pos").trim() || "oklch(0.58 0.13 150)";
    const cNeg = cs.getPropertyValue("--neg").trim() || "oklch(0.55 0.18 25)";
    const cNeu = cs.getPropertyValue("--ink-3").trim() || "oklch(0.6 0.01 260)";

    const data = [
      { label: "Pos", value: positive / total, color: cPos },
      { label: "Neu", value: neutral / total, color: cNeu },
      { label: "Neg", value: negative / total, color: cNeg },
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
    const cInk  = cs.getPropertyValue("--ink").trim()  || "oklch(0.2 0.012 260)";
    const cInk3 = cs.getPropertyValue("--ink-3").trim() || "oklch(0.6 0.01 260)";
    let centerText = g.select<SVGTextElement>("text.center");
    if (centerText.empty()) {
      centerText = g.append("text").attr("class", "center").attr("text-anchor", "middle").attr("dy", "0.1em");
      centerText.append("tspan").attr("class", "pct").attr("font-size", "13").attr("font-weight", "500").attr("font-family", "var(--font-mono)");
      centerText.append("tspan").attr("class", "lbl").attr("font-size", "7").attr("font-family", "var(--font-mono)").attr("x", "0").attr("dy", "11");
    }
    centerText.select("tspan.pct").attr("fill", cInk).text(`${Math.round((positive / total) * 100)}%`);
    centerText.select("tspan.lbl").attr("fill", cInk3).text("positive");

  }, [positive, neutral, negative]);

  return (
    <div className="flex flex-col">
      <div className="eyebrow mb-2">Sentiment</div>
      <div className="flex items-center gap-3">
        <svg ref={svgRef} className="w-[88px] h-[88px] flex-shrink-0" />
        <div className="flex flex-col gap-1.5 font-data tabular text-[11px] flex-1">
          <span className="flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-ki-success" />
              <span className="text-ki-on-surface-secondary">Positive</span>
            </span>
            <span className="text-ki-on-surface">{Math.round(positive * 100)}%</span>
          </span>
          <span className="flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-ki-on-surface-muted" />
              <span className="text-ki-on-surface-secondary">Neutral</span>
            </span>
            <span className="text-ki-on-surface">{Math.round(neutral * 100)}%</span>
          </span>
          <span className="flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-ki-error" />
              <span className="text-ki-on-surface-secondary">Negative</span>
            </span>
            <span className="text-ki-on-surface">{Math.round(negative * 100)}%</span>
          </span>
        </div>
      </div>
    </div>
  );
}

"use client";

import dynamic from "next/dynamic";

const ResponsiveContainer = dynamic(
  () => import("recharts").then((m) => m.ResponsiveContainer),
  { ssr: false }
);
const AreaChart = dynamic(
  () => import("recharts").then((m) => m.AreaChart),
  { ssr: false }
);
const Area = dynamic(
  () => import("recharts").then((m) => m.Area),
  { ssr: false }
);
const XAxis = dynamic(
  () => import("recharts").then((m) => m.XAxis),
  { ssr: false }
);
const YAxis = dynamic(
  () => import("recharts").then((m) => m.YAxis),
  { ssr: false }
);
const CartesianGrid = dynamic(
  () => import("recharts").then((m) => m.CartesianGrid),
  { ssr: false }
);
const Tooltip = dynamic(
  () => import("recharts").then((m) => m.Tooltip),
  { ssr: false }
);

interface PolarizationPoint {
  round: number;
  polarization: number;
  avg_position: number;
  num_agents?: number;
}

interface Props {
  data: PolarizationPoint[];
}

export default function PolarizationChart({ data }: Props) {
  const chartData = data.map((d) => ({
    ...d,
    label: `R${d.round}`,
  }));

  return (
    <div className="h-72 bg-white rounded-xl border border-gray-200 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="polGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="posGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: "#6b7280" }}
            axisLine={{ stroke: "#9ca3af" }}
            tickLine={{ stroke: "#9ca3af" }}
          />
          <YAxis
            yAxisId="left"
            domain={[0, "auto"]}
            tick={{ fontSize: 11, fill: "#6b7280" }}
            axisLine={{ stroke: "#9ca3af" }}
            tickLine={{ stroke: "#9ca3af" }}
            label={{
              value: "Polarization",
              angle: -90,
              position: "insideLeft",
              fill: "#6b7280",
              fontSize: 10,
            }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            domain={[-1, 1]}
            tick={{ fontSize: 11, fill: "#6b7280" }}
            axisLine={{ stroke: "#9ca3af" }}
            tickLine={{ stroke: "#9ca3af" }}
            label={{
              value: "Avg Position",
              angle: 90,
              position: "insideRight",
              fill: "#6b7280",
              fontSize: 10,
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#ffffff",
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              fontSize: 12,
              color: "#1f2937",
              fontFamily: "monospace",
            }}
          />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="polarization"
            stroke="#3b82f6"
            fill="url(#polGrad)"
            strokeWidth={2}
            name="Polarization"
          />
          <Area
            yAxisId="right"
            type="monotone"
            dataKey="avg_position"
            stroke="#f59e0b"
            fill="url(#posGrad)"
            strokeWidth={2}
            name="Avg Position"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

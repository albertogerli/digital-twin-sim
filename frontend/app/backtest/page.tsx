import dynamic from "next/dynamic";

const BacktestDashboard = dynamic(
  () => import("@/components/backtest/BacktestDashboard"),
  { ssr: false },
);

export default function BacktestPage() {
  return <BacktestDashboard />;
}

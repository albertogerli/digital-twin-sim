import dynamic from "next/dynamic";

const WargameTerminal = dynamic(
  () => import("@/components/wargame/WargameTerminal"),
  { ssr: false },
);

export default function WargamePage() {
  return <WargameTerminal />;
}

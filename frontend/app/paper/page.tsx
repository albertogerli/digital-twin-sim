"use client";

import Link from "next/link";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell,
  AreaChart, Area, ReferenceLine, ResponsiveContainer,
} from "recharts";

/* ── Data ─────────────────────────────────────────── */

const SOBOL_DATA = [
  { param: "α_herd", s1: 0.364, st: 0.555, type: "Calibrable" },
  { param: "α_anchor", s1: 0.207, st: 0.452, type: "Calibrable" },
  { param: "α_social", s1: 0.086, st: 0.213, type: "Calibrable" },
  { param: "α_event", s1: 0.026, st: 0.115, type: "Calibrable" },
  { param: "λ_citizen", s1: 0.002, st: 0.121, type: "Frozen" },
  { param: "λ_elite", s1: 0.007, st: 0.016, type: "Frozen" },
  { param: "θ_herd", s1: 0.003, st: 0.024, type: "Frozen" },
  { param: "δ_drift", s1: 0.003, st: 0.013, type: "Frozen" },
];

const ENKF_ROUNDS = [
  { round: "0 (prior)", obs: "—", mean: 50.3, ci: "[50.2, 50.4]", width: 0.2 },
  { round: "1", obs: "41.0%", mean: 50.7, ci: "[50.5, 51.3]", width: 0.8 },
  { round: "2", obs: "42.0%", mean: 51.4, ci: "[50.9, 52.0]", width: 1.0 },
  { round: "3", obs: "43.0%", mean: 50.3, ci: "[50.1, 50.7]", width: 0.6 },
  { round: "4", obs: "43.0%", mean: 50.0, ci: "[50.0, 50.1]", width: 0.1 },
  { round: "5", obs: "44.0%", mean: 50.0, ci: "[50.0, 50.1]", width: 0.1 },
  { round: "6 (final)", obs: "44.0%", mean: 50.1, ci: "[50.0, 50.4]", width: 0.4 },
];

const ENKF_BASELINES = [
  { method: "Last available poll", pred: 44.0, err: 7.9, dynamics: false, params: false },
  { method: "Running poll average", pred: 42.8, err: 9.1, dynamics: false, params: false },
  { method: "EnKF (state only)", pred: 50.1, err: 1.8, dynamics: true, params: false },
  { method: "EnKF (state + params)", pred: 50.1, err: 1.8, dynamics: true, params: true },
];

const VERSION_COMPARISON = [
  { version: "v2", scenarios: "42 (34/8)", maeTest: 19.2, maeTrain: 14.3, rmseTest: 26.6, cov90train: 79.4, cov90test: 75.0, strategy: "Discrepancy model", pollFree: true },
  { version: "v2.1", scenarios: "42 (34/8)", maeTest: 18.9, maeTrain: 15.1, rmseTest: 21.6, cov90train: 2.9, cov90test: null, strategy: "Selective grounding", pollFree: true },
  { version: "v2.2", scenarios: "20 (15/5)", maeTest: 11.4, maeTrain: 15.3, rmseTest: 16.8, cov90train: 73.3, cov90test: 80.0, strategy: "Hybrid grounding", pollFree: true },
  { version: "v2.3", scenarios: "20 (15/5)", maeTest: 13.6, maeTrain: 13.9, rmseTest: 20.0, cov90train: 86.7, cov90test: 80.0, strategy: "Hybrid + PubOp\u2020", pollFree: false },
  { version: "v2.8", scenarios: "42 (34/8)", maeTest: 17.6, maeTrain: 14.0, rmseTest: 24.7, cov90train: 82.4, cov90test: 87.5, strategy: "Discrepancy + Sprint 1-13 sim hardening", pollFree: true },
];

const TEST_RESULTS = [
  { scenario: "Archegos Capital*", domain: "Financial", gt: 35.0, pred: ">99.9", err: 65.0, ci: "[99.2, 100.0]", covered: false },
  { scenario: "Greek Bailout", domain: "Political", gt: 38.7, pred: "65.3 ± 11.7", err: 26.6, ci: "[42.7, 84.5]", covered: false },
  { scenario: "Net Neutrality", domain: "Technology", gt: 83.0, pred: "66.3 ± 12.6", err: 16.7, ci: "[47.3, 82.8]", covered: true },
  { scenario: "French Election", domain: "Political", gt: 66.1, pred: "51.6 ± 13.7", err: 14.5, ci: "[29.7, 72.8]", covered: true },
  { scenario: "COVID Vax (IT)", domain: "Pub. Health", gt: 80.0, pred: "70.8 ± 11.2", err: 9.2, ci: "[52.1, 88.7]", covered: true },
  { scenario: "Tesla Cybertruck", domain: "Commercial", gt: 62.0, pred: "54.1 ± 14.7", err: 7.9, ci: "[26.1, 72.6]", covered: true },
  { scenario: "Amazon HQ2", domain: "Corporate", gt: 56.0, pred: "63.3 ± 12.9", err: 7.3, ci: "[48.7, 80.6]", covered: true },
  { scenario: "Turkish Ref.", domain: "Political", gt: 51.4, pred: "57.5 ± 13.0", err: 6.1, ci: "[37.4, 75.7]", covered: true },
];

const DOMAIN_DISCREPANCY = [
  { domain: "Financial", n: 6, bd: -0.054, bs: 0.744, bias: "Over-predicts" },
  { domain: "Energy\u2021", n: 1, bd: -0.026, bs: 0.709, bias: "Over-predicts" },
  { domain: "Public Health", n: 4, bd: -0.032, bs: 0.534, bias: "Over-predicts" },
  { domain: "Corporate", n: 6, bd: -0.040, bs: 0.432, bias: "Over-predicts" },
  { domain: "Labor\u2021", n: 1, bd: -0.011, bs: 0.301, bias: "Insufficient data" },
  { domain: "Commercial\u2021", n: 2, bd: +0.008, bs: 0.245, bias: "Insufficient data" },
  { domain: "Political", n: 11, bd: +0.019, bs: 0.198, bias: "No systematic" },
  { domain: "Environmental\u2021", n: 1, bd: -0.015, bs: 0.178, bias: "Insufficient data" },
  { domain: "Social\u2021", n: 1, bd: +0.023, bs: 0.112, bias: "Insufficient data" },
  { domain: "Technology\u2021", n: 2, bd: -0.021, bs: 0.066, bias: "No systematic" },
];

/* ── v2.5: Null-Baseline Benchmark ─────────────── */

const BASELINE_SUPPORT = [
  { baseline: "Naive persistence", meanRmse: 0.038, medianRmse: 0.021, terminal: 0.026, beats: "—" },
  { baseline: "Running mean", meanRmse: 0.063, medianRmse: 0.034, terminal: 0.078, beats: "0 / 43" },
  { baseline: "OLS linear trend", meanRmse: 0.042, medianRmse: 0.015, terminal: 0.043, beats: "4 / 43" },
  { baseline: "AR(1)", meanRmse: 0.055, medianRmse: 0.034, terminal: 0.057, beats: "0 / 43" },
];

const BASELINE_SIGNED = [
  { baseline: "Naive persistence", meanRmse: 0.072, medianRmse: 0.040, terminal: 0.097, beats: "—" },
  { baseline: "Running mean", meanRmse: 0.117, medianRmse: 0.077, terminal: 0.148, beats: "0 / 43" },
  { baseline: "OLS linear trend", meanRmse: 0.082, medianRmse: 0.035, terminal: 0.121, beats: "6 / 43" },
  { baseline: "AR(1)", meanRmse: 0.102, medianRmse: 0.058, terminal: 0.118, beats: "0 / 43" },
];

const PERSISTENCE_BY_DOMAIN = [
  { domain: "Political", n: 15, persistence: 0.012, ols: 0.009, ar1: 0.020 },
  { domain: "Labor", n: 1, persistence: 0.009, ols: 0.008, ar1: 0.016 },
  { domain: "Environmental", n: 2, persistence: 0.021, ols: 0.013, ar1: 0.033 },
  { domain: "Corporate", n: 8, persistence: 0.048, ols: 0.062, ar1: 0.065 },
  { domain: "Public health", n: 5, persistence: 0.052, ols: 0.054, ar1: 0.081 },
  { domain: "Commercial", n: 5, persistence: 0.059, ols: 0.075, ar1: 0.079 },
  { domain: "Financial", n: 7, persistence: 0.063, ols: 0.073, ar1: 0.094 },
];

const SCENARIO_MATRIX = [
  { axis: "Domain", value: "political", count: 15 },
  { axis: "Domain", value: "corporate", count: 8 },
  { axis: "Domain", value: "financial", count: 7 },
  { axis: "Domain", value: "commercial", count: 5 },
  { axis: "Domain", value: "public_health", count: 5 },
  { axis: "Domain", value: "environmental", count: 2 },
  { axis: "Domain", value: "labor", count: 1 },
  { axis: "Region", value: "US", count: 22 },
  { axis: "Region", value: "EU", count: 16 },
  { axis: "Region", value: "APAC", count: 2 },
  { axis: "Region", value: "LATAM", count: 2 },
  { axis: "Region", value: "GLOBAL", count: 1 },
  { axis: "Tension", value: "low", count: 26 },
  { axis: "Tension", value: "moderate", count: 7 },
  { axis: "Tension", value: "critical", count: 6 },
  { axis: "Tension", value: "high", count: 4 },
];

/* ── Components ───────────────────────────────────── */

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mb-10">
      <h2 className="text-sm font-extrabold uppercase tracking-[0.06em] text-ki-on-surface mb-4 pb-1.5 border-b border-ki-border">
        {title}
      </h2>
      {children}
    </section>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-ki-on-surface-secondary leading-relaxed mb-3">{children}</p>;
}

function Eq({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-4 p-3 bg-ki-surface-sunken border border-ki-border rounded-sm font-data text-xs text-ki-on-surface overflow-x-auto">
      {children}
    </div>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return (
    <th className={`text-xs font-data text-ki-on-surface-muted uppercase tracking-wider px-3 py-1.5 ${right ? "text-right" : "text-left"}`}>
      {children}
    </th>
  );
}

function Td({ children, mono, right, bold, color }: { children: React.ReactNode; mono?: boolean; right?: boolean; bold?: boolean; color?: string }) {
  return (
    <td
      className={`px-3 py-1.5 text-xs ${mono ? "font-data" : ""} ${right ? "text-right" : ""} ${bold ? "font-semibold" : ""} ${color || "text-ki-on-surface-secondary"}`}
    >
      {children}
    </td>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-3 text-center">
      <div className="text-xs font-data text-ki-on-surface-muted uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-bold text-ki-on-surface">{value}</div>
      {sub && <div className="text-xs text-ki-on-surface-muted mt-0.5">{sub}</div>}
    </div>
  );
}

function BarInline({ value, max, color }: { value: number; max: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-ki-surface-sunken h-2 overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${(value / max) * 100}%` }} />
      </div>
      <span className="text-xs font-data text-ki-on-surface-muted w-10 text-right">{value.toFixed(3)}</span>
    </div>
  );
}

/* ── Page ──────────────────────────────────────────── */

export default function PaperPage() {
  return (
    <main>
      {/* Title block */}
      <div className="max-w-4xl mx-auto px-5 pt-6 pb-6">
        <Link href="/" className="inline-flex items-center gap-1 text-[12px] text-ki-on-surface-muted hover:text-ki-on-surface mb-3 group">
          <span className="material-symbols-outlined text-[14px] group-hover:-translate-x-0.5 transition-transform">arrow_back</span>
          Dashboard
        </Link>
        <div className="eyebrow text-ki-primary mb-2">Working paper</div>
        <h1 className="text-[28px] sm:text-[34px] font-medium tracking-tight2 text-ki-on-surface leading-[1.15] mb-4">
          DigitalTwinSim: Bayesian Calibration, Null-Baseline Benchmarking, and Online Data Assimilation for LLM-Agent Opinion Dynamics
        </h1>
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-[12px] text-ki-on-surface-secondary mb-1">
          <span>Alberto Giovanni Gerli</span>
          <span className="font-data">May 2026</span>
        </div>
        <div className="text-[12px] text-ki-on-surface-muted mb-6">
          Tourbillon Tech Srl · Università degli Studi di Milano
        </div>

        {/* Abstract */}
        <div className="bg-ki-surface-sunken border border-ki-border rounded p-5">
          <div className="eyebrow mb-2">Abstract</div>
          <p className="text-xs text-ki-on-surface-secondary leading-relaxed">
            LLM-agent simulations can generate rich opinion-dynamics narratives, but their outputs are
            uncalibrated. We address this by embedding a differentiable, force-based opinion simulator
            inside a hierarchical Bayesian calibration framework. Five competing mechanisms are combined
            through a gauge-fixed softmax mixture; a three-level model (global, domain, scenario) with
            explicit readout discrepancy is fitted via SVI on <strong>42 empirical scenarios</strong>
            spanning 10 domains. On <strong>8 held-out scenarios</strong> the calibrated model achieves
            <strong>17.6 pp</strong> MAE with <strong>87.5%</strong> coverage of 90% credible intervals.
            Simulation-based calibration confirms well-specified posteriors under NUTS; however, the
            production SVI approximation underestimates uncertainty on weaker parameters
            (5&ndash;13&times; narrower than NUTS)&mdash;a central finding, not a side caveat, since it
            bounds the reliability of all reported intervals.
            An Ensemble Kalman Filter demonstrates the feasibility of online assimilation: on an
            in-sample case study the point prediction improves substantially (<strong>1.8 pp</strong>),
            but the ensemble remains under-dispersed and out-of-sample validation is needed.
            <br /><br />
            <strong>Version 2.5 adds a null-baseline benchmarking layer.</strong> Four standard forecasters
            (naive persistence, running mean, OLS linear trend, AR(1)) are evaluated against the same 43 empirical
            trajectories using Diebold&ndash;Mariano with the Harvey&ndash;Leybourne&ndash;Newbold small-sample correction.
            Naive persistence is a surprisingly strong baseline (mean RMSE = <strong>0.038</strong>, i.e. 3.8 pp on support),
            and OLS linear trend beats persistence significantly on only 4/43 (support) and 6/43 (signed) scenarios.
            Domain skill is heterogeneous: political persistence RMSE = 0.012, financial = 0.063 &mdash; a
            <strong>5&times; spread</strong>. The corpus covers 25/140 cells of a 7&times;5&times;4 diversity matrix,
            establishing a reproducible skill floor any future calibrated sim must clear.
          </p>
        </div>

        {/* Key stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
          <StatCard label="Test MAE (canonical)" value="17.6 pp" sub="N=8 held-out" />
          <StatCard label="Persistence RMSE" value="0.038" sub="null baseline, support" />
          <StatCard label="Scenarios" value="43" sub="25/140 matrix cells" />
          <StatCard label="Benchmark tests" value="103" sub="2.94 s wall time" />
        </div>
      </div>

      {/* Table of Contents */}
      <article className="max-w-4xl mx-auto p-3 pb-16">
        <nav className="mb-10 p-4 bg-ki-surface-sunken rounded-sm border border-ki-border">
          <h3 className="text-xs font-data uppercase text-ki-on-surface-muted tracking-wider mb-2">Contents</h3>
          <ol className="space-y-1.5 text-sm">
            {[
              ["five-forces", "1. Five-Force Opinion Dynamics"],
              ["llm-engine", "2. LLM Agent Engine"],
              ["hierarchy", "3. Hierarchical Bayesian Calibration"],
              ["results", "4. Calibration Results (v2 Baseline)"],
              ["discrepancy", "5. Readout Discrepancy"],
              ["sensitivity", "6. Sensitivity & Validation"],
              ["enkf", "7. Online Assimilation (EnKF Feasibility)"],
              ["evolution", "8. Calibration Evolution (v2 → v2.3)"],
              ["pubop", "9. Public Opinion Agent (v2.3, poll-informed)"],
              ["limitations", "10. Limitations & Future Work"],
              ["null-baseline", "11. Null-Baseline Benchmark (v2.5 \u2605)"],
            ].map(([id, label]) => (
              <li key={id}>
                <a href={`#${id}`} className="text-ki-primary hover:text-ki-primary transition-colors">
                  {label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        {/* ── Evaluation Protocols ────────────────── */}
        <div className="mb-10 p-4 bg-ki-warning-soft border border-ki-warning/30 rounded">
          <div className="eyebrow text-ki-warning mb-2">Evaluation protocols</div>
          <p className="text-[12px] text-ki-on-surface-secondary mb-3 leading-relaxed">
            This paper uses multiple evaluation protocols across versions. To avoid confusion, we summarize
            them here. <strong className="text-ki-on-surface">The canonical test metric is MAE on the full held-out test set.</strong> Coverage
            is reported on both training and test sets where available (v2, v2.2, v2.3). We caution against
            direct cross-version coverage comparison since the denominator changes (8 test for v2 vs 5 test for v2.2/v2.3).
          </p>
          <div className="overflow-x-auto rounded border border-ki-warning/30">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="bg-ki-warning-soft border-b border-ki-warning/30">
                  <th className="text-left px-3 py-1.5 eyebrow text-ki-warning">Protocol</th>
                  <th className="text-left px-3 py-1.5 eyebrow text-ki-warning">Used in</th>
                  <th className="text-left px-3 py-1.5 eyebrow text-ki-warning">Train / Test</th>
                  <th className="text-left px-3 py-1.5 eyebrow text-ki-warning">Note</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-warning/15 text-ki-on-surface-secondary">
                <tr>
                  <td className="px-3 py-1.5 font-semibold">Full test (N=8)</td>
                  <td className="px-3 py-1.5">v2 &sect;3</td>
                  <td className="px-3 py-1.5 font-data">34 / 8</td>
                  <td className="px-3 py-1.5">Canonical. Includes Archegos (65 pp outlier)</td>
                </tr>
                <tr>
                  <td className="px-3 py-1.5 font-semibold">Verified test (N=7)</td>
                  <td className="px-3 py-1.5">v2 &sect;3</td>
                  <td className="px-3 py-1.5 font-data">34 / 7</td>
                  <td className="px-3 py-1.5">Excludes Archegos (NEEDS_VERIFICATION flag)</td>
                </tr>
                <tr>
                  <td className="px-3 py-1.5 font-semibold">Hybrid subset (N=5)</td>
                  <td className="px-3 py-1.5">v2.2, v2.3 &sect;7</td>
                  <td className="px-3 py-1.5 font-data">15 / 5</td>
                  <td className="px-3 py-1.5">Curated high-quality scenarios only</td>
                </tr>
                <tr>
                  <td className="px-3 py-1.5 font-semibold">EnKF case study</td>
                  <td className="px-3 py-1.5">&sect;6</td>
                  <td className="px-3 py-1.5 font-data">in-sample</td>
                  <td className="px-3 py-1.5">Brexit is in the training set; not OOS validation</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* ── 1. Five Forces ───────────────────────── */}
        <Section id="five-forces" title="1. Five-Force Opinion Dynamics">
          <P>
            Each agent maintains a scalar position p<sub>i</sub> &isin; [-1, +1].
            At each round, five independent forces act on every agent, capturing distinct social mechanisms.
            Forces are standardized via exponential moving average (EMA) across ALL agents per round,
            ensuring permutation invariance, then combined through a gauge-fixed softmax mixture.
          </P>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 my-6">
            {[
              { name: "Direct LLM", alpha: "0 (gauge)", pi: "20.1%", desc: "LLM-generated per-agent opinion shifts, attenuated by rigidity" },
              { name: "Herd Behavior", alpha: "-0.176", pi: "16.9%", desc: "Consensus pull when deviation exceeds learned threshold" },
              { name: "Anchor Rigidity", alpha: "+0.297", pi: "27.1%", desc: "Restorative force toward original position (dominant)" },
              { name: "Social Influence", alpha: "-0.105", pi: "18.1%", desc: "Bounded-confidence averaging with tolerance-gated interaction" },
              { name: "Event Shock", alpha: "-0.130", pi: "17.7%", desc: "Uniform directional push from exogenous events" },
            ].map((f) => (
              <div key={f.name} className="bg-ki-surface-sunken border border-ki-border rounded-sm p-3">
                <div className="flex items-baseline justify-between mb-1">
                  <span className="text-xs font-semibold text-ki-on-surface">{f.name}</span>
                  <span className="text-xs font-data text-ki-primary">&pi; = {f.pi}</span>
                </div>
                <div className="text-xs font-data text-ki-on-surface-muted mb-1">&alpha; = {f.alpha}</div>
                <div className="text-xs text-ki-on-surface-secondary leading-relaxed">{f.desc}</div>
              </div>
            ))}
          </div>
          {/* Radar chart: softmax weights */}
          <div className="my-8 flex justify-center">
            <div className="w-full max-w-md">
              <h4 className="text-xs font-data text-ki-on-surface-muted uppercase tracking-wider mb-3 text-center">
                Softmax Force Weights (&pi;)
              </h4>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={[
                  { force: "Direct", weight: 20.1 },
                  { force: "Herd", weight: 16.9 },
                  { force: "Anchor", weight: 27.1 },
                  { force: "Social", weight: 18.1 },
                  { force: "Event", weight: 17.7 },
                ]}>
                  <PolarGrid stroke="#d4d4d4" />
                  <PolarAngleAxis dataKey="force" tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                  <PolarRadiusAxis angle={90} domain={[0, 30]} tick={{ fontSize: 10, fill: "#8a8a8a" }} />
                  <Radar dataKey="weight" stroke="#1a6dff" fill="#1a6dff" fillOpacity={0.2} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <P>
            The direct force weight &alpha;<sub>direct</sub> &equiv; 0 serves as the gauge reference, making the
            four remaining weights interpretable as log-odds relative to direct LLM influence.
            Anchor rigidity receives the largest share (~27%), consistent with well-documented status quo bias
            in opinion dynamics.
          </P>
          <Eq>
            &Delta;p<sub>i</sub> = &lambda;<sub>&tau;</sub> &middot; &Sigma;<sub>k</sub> &pi;<sub>k</sub> &middot; f&#771;<sub>k,i</sub>
            &nbsp;&nbsp; where &pi; = softmax([0, &alpha;<sub>h</sub>, &alpha;<sub>a</sub>, &alpha;<sub>s</sub>, &alpha;<sub>e</sub>])
          </Eq>
        </Section>

        {/* ── 2. LLM Agent Engine ─────────────────── */}
        <Section id="llm-engine" title="2. LLM Agent Engine">
          <P>
            The LLM provides two inputs to the force system: per-agent opinion shifts &Delta;<sup>LLM</sup><sub>i</sub>(t)
            (Force 1) and event narratives with magnitude m<sub>t</sub> and direction d<sub>t</sub> (Force 5).
            Both are generated by Google Gemini (<code>gemini-2.0-flash-lite</code>), selected for cost efficiency
            across 42 multi-round scenarios.
          </P>

          <div className="my-6 space-y-4">
            {[
              { label: "Prompting", text: "Structured prompts with scenario context, agent positions, social graph, and recent events. Zero-shot instruction following requesting JSON with Δ ∈ [-1, +1] and reasoning chain. Full templates in the code repository." },
              { label: "Temperature", text: "T = 0.7, top-p = 0.95. Each scenario uses a single LLM rollout; Δ^LLM values are cached and reused across all SVI optimization steps for deterministic gradients." },
              { label: "Inter-run variance", text: "Not measured directly (single seed per scenario). Discrepancy terms b_d and b_s absorb both structural model error and LLM stochasticity without separating the two. Multiple rollouts per scenario would enable this decomposition." },
              { label: "Reproducibility", text: "LLM outputs are not bitwise reproducible across API versions. Cached outputs (January 2026) are included in the repository for exact numerical reproduction." },
            ].map((item) => (
              <div key={item.label} className="bg-ki-surface-sunken border border-ki-border rounded-sm p-3">
                <span className="text-xs font-semibold text-ki-on-surface-secondary">{item.label}.</span>{" "}
                <span className="text-xs text-ki-on-surface-secondary">{item.text}</span>
              </div>
            ))}
          </div>
        </Section>

        {/* ── 3. Hierarchical Bayesian ─────────────── */}
        <Section id="hierarchy" title="3. Hierarchical Bayesian Calibration">
          <P>
            A single set of global parameters cannot describe opinion dynamics across diverse domains.
            We use a three-level hierarchy enabling partial pooling: scenarios share information through
            domain and global priors while retaining scenario-specific flexibility through discrepancy terms.
          </P>

          {/* Visual hierarchy */}
          <div className="my-8 space-y-3">
            {[
              { level: "Level 1 — Global", eq: "μ_global ~ N(0, I₄), σ_global ~ HalfNormal(0.3)", desc: "Shared prior on mixing weights" },
              { level: "Level 2 — Domain", eq: "μ_d ~ N(μ_global, diag(σ²_global))", desc: "Domain-level means (political, financial, ...)" },
              { level: "Level 3 — Scenario", eq: "θ_s ~ N(μ_d + B·x_s, diag(σ²_d))", desc: "Scenario-specific params with covariate regression" },
            ].map((l, i) => (
              <div key={l.level} className="flex gap-4 items-start">
                <div className="flex flex-col items-center shrink-0">
                  <div className={`w-7 h-7 rounded-sm flex items-center justify-center text-white text-xs font-bold ${
                    i === 0 ? "bg-ki-primary" : i === 1 ? "bg-ki-primary/80" : "bg-ki-primary/60"
                  }`}>{i + 1}</div>
                  {i < 2 && <div className="w-px h-3 bg-ki-border" />}
                </div>
                <div className="flex-1 bg-ki-surface-sunken border border-ki-border rounded-sm p-3">
                  <div className="text-xs font-semibold text-ki-on-surface mb-1">{l.level}</div>
                  <div className="font-data text-xs text-ki-on-surface-secondary mb-1">{l.eq}</div>
                  <div className="text-xs text-ki-on-surface-muted">{l.desc}</div>
                </div>
              </div>
            ))}
          </div>

          <p className="text-xs text-[#b45309] bg-[#fff8e1] border border-[#ffe082] rounded-sm p-3 mb-3">
            <strong>Domain-level identifiability.</strong> Of 10 domains, only 3 have N &ge; 4 training scenarios
            (political: 11, corporate: 6, financial: 6). Five domains have N &le; 2 (energy, labor, commercial,
            environmental, social). For these, domain-level parameters &mu;<sub>d</sub> are effectively determined by
            the global prior, and domain-level discrepancy b<sub>d</sub> is unidentifiable from scenario-level b<sub>s</sub>.
            We retain the 10-domain hierarchy (domains do differ mechanistically) but mark under-represented domains
            with &Dagger; throughout.
          </p>
          <P>
            <strong>Readout discrepancy.</strong> Rather than absorbing structural misspecification into parameters,
            we model it explicitly: b<sub>d</sub> (domain-level bias) and b<sub>s</sub> (scenario-specific), operating
            in logit space on the readout. This follows Kennedy &amp; O&apos;Hagan (2001): calibrated parameters &theta;
            represent genuine opinion dynamics mechanisms, while discrepancy terms absorb systematic simulator errors.
          </P>
          <P>
            <strong>Inference.</strong> SVI with AutoLowRankMultivariateNormal guide in NumPyro, 3000 steps,
            learning rate 0.002 with cosine annealing. Gradients computed via JAX autodiff through the full
            simulate &rarr; readout &rarr; likelihood chain.
          </P>
          <div className="my-3 p-3 bg-[#fff8e1] border border-[#ffe082] rounded-sm">
            <p className="text-xs text-[#b45309] leading-relaxed">
              <strong>Caveat on uncertainty estimates.</strong> Comparison with gold-standard NUTS (&sect;5)
              reveals that SVI agrees on dominant parameters (&alpha;<sub>herd</sub>, &alpha;<sub>anchor</sub>)
              but substantially underestimates uncertainty on weaker ones (&alpha;<sub>event</sub>:
              |&Delta;&mu;|/&sigma;<sub>NUTS</sub> = 1.71, SVI standard deviations 5&ndash;13&times; narrower).
              All credible intervals in this paper are likely too narrow for &alpha;<sub>social</sub> and
              &alpha;<sub>event</sub>. The effect on coverage is ambiguous: wider NUTS intervals could raise or
              lower it depending on posterior mean shifts. Without full NUTS calibration on all 42 scenarios,
              the reported 75.0% test coverage is not directly comparable to what exact inference would yield.
            </p>
          </div>
        </Section>

        {/* ── 3. Calibration Results ───────────────── */}
        <Section id="results" title="4. Calibration Results (v2 Baseline)">
          <P>
            The model is calibrated on 42 empirical scenarios (34 train / 8 test) across 10 domains,
            each with a ground-truth final outcome and, where available, intermediate polling data.
          </P>

          {/* Global posterior */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-4">Calibrated Global Posterior</h3>
          <div className="my-6 overflow-hidden rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Parameter</Th>
                  <Th right>Mean</Th>
                  <Th right>95% CI</Th>
                  <Th>Interpretation</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {[
                  { p: "α_herd", m: "-0.176", ci: "[-0.265, -0.079]", i: "Below direct (consensus pull)" },
                  { p: "α_anchor", m: "+0.297", ci: "[+0.199, +0.401]", i: "Strongest mechanism (status quo bias)" },
                  { p: "α_social", m: "-0.105", ci: "[-0.202, -0.005]", i: "Below direct (peer influence)" },
                  { p: "α_event", m: "-0.130", ci: "[-0.227, -0.033]", i: "Below direct (exogenous shocks)" },
                ].map((r) => (
                  <tr key={r.p}>
                    <Td mono bold>{r.p}</Td>
                    <Td mono right>{r.m}</Td>
                    <Td mono right>{r.ci}</Td>
                    <Td>{r.i}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Test set results */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-4">Test Set Performance</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            <StatCard label="MAE (full test, N=8)" value="17.6 pp" sub="canonical" />
            <StatCard label="MAE (verified, N=7)" value="12.6 pp" sub="excl. Archegos" />
            <StatCard label="Cov 90% (test, N=8)" value="87.5%" />
            <StatCard label="Cov 90% (train, N=34)" value="79.4%" />
          </div>

          <div className="my-6 overflow-x-auto rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Scenario</Th>
                  <Th>Domain</Th>
                  <Th right>GT %</Th>
                  <Th right>Predicted</Th>
                  <Th right>Error</Th>
                  <Th right>90% CI</Th>
                  <Th right>Cov</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {TEST_RESULTS.map((r) => (
                  <tr key={r.scenario} className={r.scenario.includes("*") ? "bg-[#fff8e1]/50" : ""}>
                    <Td bold>{r.scenario}</Td>
                    <Td>{r.domain}</Td>
                    <Td mono right>{r.gt}</Td>
                    <Td mono right>{r.pred}</Td>
                    <Td mono right bold color={r.err > 20 ? "text-red-600" : r.err < 10 ? "text-green-600" : "text-ki-on-surface-secondary"}>
                      {r.err.toFixed(1)}
                    </Td>
                    <Td mono right>{r.ci}</Td>
                    <Td right>{r.covered ? "\u2713" : "\u2717"}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-ki-on-surface-muted mt-1 mb-4">
            *NEEDS_VERIFICATION &mdash; excluded from primary metrics. CI computed in logit space and back-transformed.
          </p>

          {/* Test set error chart */}
          <div className="my-8">
            <h4 className="text-xs font-data text-ki-on-surface-muted uppercase tracking-wider mb-3 text-center">
              Test Set: Prediction Error by Scenario (pp)
            </h4>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={TEST_RESULTS.map(r => ({ name: r.scenario.replace("*", ""), err: r.err, covered: r.covered }))}
                layout="vertical"
                margin={{ left: 100, right: 30, top: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#d4d4d4" />
                <XAxis type="number" domain={[0, 70]} tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "#8a8a8a" }} width={100} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #d4d4d4" }} formatter={(v: unknown) => `${Number(v).toFixed(1)} pp`} />
                <Bar dataKey="err" name="Error (pp)" radius={[0, 4, 4, 0]}>
                  {TEST_RESULTS.map((r, i) => (
                    <Cell key={i} fill={r.err > 20 ? "#f87171" : r.err < 10 ? "#4ade80" : "#fbbf24"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-6 mt-2 text-xs text-ki-on-surface-muted">
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-green-400 inline-block" /> &lt; 10 pp</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-yellow-400 inline-block" /> 10&ndash;20 pp</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-red-400 inline-block" /> &gt; 20 pp</span>
            </div>
          </div>
        </Section>

        {/* ── 4. Discrepancy ───────────────────────── */}
        <Section id="discrepancy" title="5. Readout Discrepancy">
          <P>
            The within-scenario discrepancy (&sigma;<sub>b,within</sub> = 0.558 logit) is ~5&times; the
            between-domain discrepancy (&sigma;<sub>b,between</sub> = 0.115 logit), indicating that
            scenario-specific factors dominate over systematic domain-level biases. In practical terms,
            0.558 logit translates to ~12&ndash;14 pp of irreducible prediction error.
          </P>

          <div className="my-6 overflow-hidden rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Domain</Th>
                  <Th right>N</Th>
                  <Th right>b<sub>d</sub></Th>
                  <Th right>Mean |b<sub>s</sub>|</Th>
                  <Th>Bias</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {DOMAIN_DISCREPANCY.map((d) => (
                  <tr key={d.domain}>
                    <Td bold>{d.domain}</Td>
                    <Td mono right>{d.n}</Td>
                    <Td mono right>{d.bd > 0 ? "+" : ""}{d.bd.toFixed(3)}</Td>
                    <Td mono right bold color={d.bs > 0.5 ? "text-red-600" : "text-ki-on-surface-secondary"}>{d.bs.toFixed(3)}</Td>
                    <Td>{d.bias}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Discrepancy chart */}
          <div className="my-8">
            <h4 className="text-xs font-data text-ki-on-surface-muted uppercase tracking-wider mb-3 text-center">
              Mean Scenario Discrepancy |b<sub>s</sub>| by Domain (logit space)
            </h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={DOMAIN_DISCREPANCY} margin={{ left: 20, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d4d4d4" vertical={false} />
                <XAxis dataKey="domain" tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <YAxis domain={[0, 0.8]} tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #d4d4d4" }} formatter={(v: unknown) => Number(v).toFixed(3)} />
                <Bar dataKey="bs" name="|b_s| mean" radius={[4, 4, 0, 0]}>
                  {DOMAIN_DISCREPANCY.map((d, i) => (
                    <Cell key={i} fill={d.bs > 0.5 ? "#f87171" : d.bs > 0.3 ? "#fbbf24" : "#4ade80"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <P>
            Financial scenarios exhibit the largest discrepancy (|b<sub>s</sub>| = 0.744), driven by trust
            cascades and contagion dynamics the base model cannot capture. Political scenarios, with the most
            training data (N=11), show the lowest systematic bias. Domains marked &Dagger; have N&le;2 scenarios;
            their discrepancy estimates are unreliable point estimates, not robust domain-level conclusions.
            In particular, energy&rsquo;s |b<sub>s</sub>| = 0.709 is based on a single scenario.
          </P>
        </Section>

        {/* ── 5. Sensitivity ──────────────────────── */}
        <Section id="sensitivity" title="6. Sensitivity & Validation">
          <P>
            Variance-based global sensitivity analysis (Sobol indices, N=1024 Saltelli samples, 18,432 simulator
            evaluations) decomposes output variance into contributions from individual parameters and their
            interactions. This motivates (though does not conclusively justify) the partition of 8 parameters
            into 4 calibrable and 4 frozen &mdash; notably &lambda;<sub>citizen</sub> (S<sub>T</sub> = 0.121)
            shows up to 7.9 pp MAE sensitivity under perturbation and should be promoted to calibrable.
          </P>

          <div className="my-6 overflow-hidden rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Parameter</Th>
                  <Th>S<sub>1</sub> (Main)</Th>
                  <Th>S<sub>T</sub> (Total)</Th>
                  <Th right>Type</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {SOBOL_DATA.map((s) => (
                  <tr key={s.param} className={s.type === "Frozen" ? "bg-ki-surface-sunken/50" : ""}>
                    <Td mono bold>{s.param}</Td>
                    <td className="px-3 py-1.5">
                      <BarInline value={s.s1} max={0.4} color="bg-ki-primary/50" />
                    </td>
                    <td className="px-3 py-1.5">
                      <BarInline value={s.st} max={0.6} color="bg-ki-primary" />
                    </td>
                    <Td right>
                      <span className={`text-xs font-data px-2 py-0.5 rounded-sm ${
                        s.type === "Calibrable" ? "bg-ki-primary/10 text-ki-primary" : "bg-ki-surface-sunken text-ki-on-surface-muted"
                      }`}>{s.type}</span>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Sobol bar chart */}
          <div className="my-8">
            <h4 className="text-xs font-data text-ki-on-surface-muted uppercase tracking-wider mb-3 text-center">
              Sobol Sensitivity Indices
            </h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={SOBOL_DATA} layout="vertical" margin={{ left: 60, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d4d4d4" />
                <XAxis type="number" domain={[0, 0.6]} tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <YAxis type="category" dataKey="param" tick={{ fontSize: 11, fill: "#8a8a8a", fontFamily: "monospace" }} width={60} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #d4d4d4" }}
                  formatter={(v: unknown) => Number(v).toFixed(3)}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="s1" name="S₁ (Main)" fill="#1a6dff80" radius={[0, 3, 3, 0]} />
                <Bar dataKey="st" name="Sₜ (Total)" fill="#1a6dff" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <P>
            <strong>&alpha;<sub>herd</sub></strong> (S<sub>T</sub> = 0.555) and <strong>&alpha;<sub>anchor</sub></strong> (S<sub>T</sub> = 0.452)
            together dominate output variance. Their pairwise interaction S<sub>2</sub> = 0.094 is the largest,
            indicating strong nonlinear coupling between herd behavior and anchor rigidity.
          </P>
          <P>
            Note: &lambda;<sub>citizen</sub> has S<sub>T</sub> = 0.121 despite S<sub>1</sub> &asymp; 0 &mdash; its
            influence arises entirely from interactions. Freezing it is a pragmatic choice for computational
            stability, not definitively justified by the data. Perturbation analysis shows up to 7.9 pp MAE
            variation under &plusmn;40% perturbation.
          </P>

          {/* SBC */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-4">Simulation-Based Calibration</h3>
          <P>
            SBC (Talts et al., 2018) confirms the generative model is well-specified under NUTS: all 6 parameters
            pass KS uniformity (p &gt; 0.20). However, the SVI vs NUTS comparison reveals a central finding:
            while point estimates agree on dominant parameters (&alpha;<sub>herd</sub>: |&Delta;&mu;|/&sigma; = 0.15;
            &alpha;<sub>anchor</sub>: 0.35), SVI standard deviations are 5&ndash;13&times; narrower than NUTS on
            weaker parameters (&alpha;<sub>social</sub>: 0.85; &alpha;<sub>event</sub>: 1.71). This means the
            model is calibrated in principle (under NUTS), but the production SVI uncertainty is approximate:
            point predictions are trustworthy, while credible interval widths should be interpreted conservatively.
          </P>
        </Section>

        {/* ── 6. EnKF ─────────────────────────────── */}
        <Section id="enkf" title="7. Online Assimilation (EnKF Feasibility Study)">
          <P>
            The Ensemble Kalman Filter bridges offline calibration to real-time operation. It maintains an
            augmented state vector [&theta;, z] combining the 4 model parameters with n agent positions,
            jointly updated as streaming observations arrive.
          </P>

          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-4">Brexit Case Study (GT = 51.89%)</h3>
          <p className="text-xs text-[#b45309] bg-[#fff8e1] border border-[#ffe082] rounded-sm p-3 mb-3">
            Brexit is part of the calibration training set. This case study demonstrates EnKF operational
            mechanics, not an independent out-of-sample validation.
          </p>

          <div className="my-6 overflow-x-auto rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Round</Th>
                  <Th right>Observation</Th>
                  <Th right>EnKF Mean</Th>
                  <Th right>90% CI</Th>
                  <Th right>Width (pp)</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {ENKF_ROUNDS.map((r) => (
                  <tr key={r.round}>
                    <Td mono bold>{r.round}</Td>
                    <Td mono right>{r.obs}</Td>
                    <Td mono right>{r.mean.toFixed(1)}%</Td>
                    <Td mono right>{r.ci}</Td>
                    <Td mono right bold color={r.width < 0.3 ? "text-green-600" : "text-ki-on-surface-secondary"}>
                      {r.width.toFixed(1)}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* EnKF trajectory chart */}
          <div className="my-8">
            <h4 className="text-xs font-data text-ki-on-surface-muted uppercase tracking-wider mb-3 text-center">
              EnKF Prediction vs Polls vs Ground Truth
            </h4>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart
                data={ENKF_ROUNDS.map((r, i) => ({
                  round: i,
                  label: `R${i}`,
                  mean: r.mean,
                  ciLo: parseFloat(r.ci.replace("[", "").split(",")[0]),
                  ciHi: parseFloat(r.ci.split(",")[1].replace("]", "")),
                  poll: r.obs === "—" ? null : parseFloat(r.obs),
                }))}
                margin={{ left: 10, right: 20, top: 10, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#d4d4d4" />
                <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <YAxis domain={[38, 54]} tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #d4d4d4" }} />
                <ReferenceLine y={51.89} stroke="#ef4444" strokeDasharray="6 3" strokeWidth={2} label={{ value: "GT 51.89%", position: "right", fontSize: 10, fill: "#ef4444" }} />
                <Area type="monotone" dataKey="ciHi" stackId="ci" stroke="none" fill="#1a6dff20" fillOpacity={0} />
                <Area type="monotone" dataKey="ciLo" stackId="ci" stroke="none" fill="#1a6dff20" fillOpacity={0.6} />
                <Area type="monotone" dataKey="mean" stroke="#1a6dff" strokeWidth={2.5} fill="none" dot={{ r: 4, fill: "#1a6dff" }} name="EnKF Mean" />
                <Area type="monotone" dataKey="poll" stroke="#22c55e" strokeWidth={0} fill="none" dot={{ r: 5, fill: "#22c55e", stroke: "#16a34a", strokeWidth: 2 }} name="Poll" connectNulls={false} />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-6 mt-2 text-xs text-ki-on-surface-muted">
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-ki-primary inline-block" /> EnKF mean</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-green-500 inline-block" /> Polls</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-red-500 inline-block border-dashed" /> Ground truth</span>
            </div>
          </div>

          <P>
            Despite polls systematically underestimating Leave by ~8 pp, the dynamics model produces a final
            prediction of 50.1% (error 1.8 pp). However, the final CI [50.0, 50.4] &mdash; a width of
            only 0.4 pp &mdash; does not cover the ground truth (51.89%). The ensemble is severely
            under-dispersed: while the point prediction is accurate, the uncertainty estimates are unreliable.
            This likely stems from insufficient process noise inflation (Q<sub>z</sub>, Q<sub>&theta;</sub>)
            and ensemble collapse after several assimilation steps. Adaptive inflation or wider initial
            ensemble spread are needed before the EnKF uncertainty can be trusted.
          </P>

          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-4">Baseline Comparison</h3>
          <div className="my-6 overflow-hidden rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Method</Th>
                  <Th right>Prediction</Th>
                  <Th right>Error (pp)</Th>
                  <Th right>Dynamics</Th>
                  <Th right>Params</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {ENKF_BASELINES.map((b) => (
                  <tr key={b.method}>
                    <Td bold>{b.method}</Td>
                    <Td mono right>{b.pred.toFixed(1)}%</Td>
                    <Td mono right bold color={b.err < 3 ? "text-green-600" : b.err > 7 ? "text-red-600" : "text-ki-on-surface-secondary"}>
                      {b.err.toFixed(1)}
                    </Td>
                    <Td right>{b.dynamics ? "\u2713" : "\u2717"}</Td>
                    <Td right>{b.params ? "\u2713" : "\u2717"}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Baseline comparison bar chart */}
          <div className="my-8">
            <h4 className="text-xs font-data text-ki-on-surface-muted uppercase tracking-wider mb-3 text-center">
              Prediction Error by Method (pp)
            </h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={ENKF_BASELINES} margin={{ left: 20, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d4d4d4" vertical={false} />
                <XAxis dataKey="method" tick={{ fontSize: 10, fill: "#8a8a8a" }} interval={0} />
                <YAxis domain={[0, 10]} tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #d4d4d4" }} formatter={(v: unknown) => `${Number(v).toFixed(1)} pp`} />
                <Bar dataKey="err" name="Error (pp)" radius={[4, 4, 0, 0]}>
                  {ENKF_BASELINES.map((b, i) => (
                    <Cell key={i} fill={b.dynamics ? "#22c55e" : "#f87171"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-6 mt-2 text-xs text-ki-on-surface-muted">
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-red-400 inline-block" /> No dynamics</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-green-500 inline-block" /> With dynamics model</span>
            </div>
          </div>

          <P>
            The EnKF reduces error by <strong>77%</strong> vs last-poll baseline (1.8 vs 7.9 pp)
            and <strong>80%</strong> vs running average (1.8 vs 9.1 pp). Note that state-only and
            state+params variants converge to the same 1.8 pp error with 6 observations, meaning the
            current evidence demonstrates the value of state updating but does not yet show a benefit
            from joint parameter learning online.
          </P>
        </Section>

        {/* ── 7. Calibration Evolution ─────────────── */}
        <Section id="evolution" title="8. Calibration Evolution (v2 → v2.3)">
          <P>
            After establishing the v2 baseline (sections 3&ndash;4), we explored three alternative modelling
            strategies targeting the two key failure modes: high domain discrepancy in financial/energy/corporate scenarios,
            and insufficient credible interval coverage.
          </P>
          <p className="text-xs text-[#b45309] bg-[#fff8e1] border border-[#ffe082] rounded-sm p-3 mb-3">
            <strong>Cross-version comparability warning.</strong> Each version uses a different dataset (42 vs 20 scenarios),
            different train/test splits, and&mdash;for v2.3&mdash;additional information (polling data). Direct numerical comparison
            across versions is invalid. We present these as independent experiments illustrating trade-offs, not a monotonic improvement narrative.
          </p>
          {/* Version timeline */}
          <div className="my-8 space-y-4">
            {[
              {
                v: "v2", date: "Mar 30", color: "bg-gray-500",
                title: "Baseline: Hierarchical Bayesian + Discrepancy",
                desc: "42 scenarios (34 train / 8 test), 10 domains. Explicit readout discrepancy b_d + b_s in logit space. SVI with AutoLowRankMVN guide.",
                result: "MAE 19.2 pp test, Coverage 79.4%",
              },
              {
                v: "v2.1", date: "Apr 1", color: "bg-blue-500",
                title: "Selective Grounding of High-Discrepancy Domains",
                desc: "Grounded only domains with |δ_s| > 0.4 (financial, corporate, energy, public_health) using Google Search verified events. Kept low-discrepancy domains unchanged. 20 grounded + 22 kept = 42 total.",
                result: "MAE 18.9 pp test (−0.3), but Coverage collapsed to 2.9%",
              },
              {
                v: "v2.2", date: "Apr 2", color: "bg-emerald-500",
                title: "Hybrid Grounding + Discrepancy Model",
                desc: "Combined original LLM agents with grounded verified events from Google Search. Reduced to 20 curated scenarios (15 train / 5 test) with higher data quality. Retained hierarchical discrepancy model.",
                result: "MAE 11.4 pp test (best), RMSE 16.8 pp, Test Cov 80.0% (4/5), Train Cov 73.3%",
              },
              {
                v: "v2.3", date: "Apr 3", color: "bg-purple-500",
                title: "Hybrid + Public Opinion Anchor Agent",
                desc: "Added an implicit Public Opinion agent derived from first available polling data, providing an empirical anchor for citizen opinion dynamics. Same 20 hybrid scenarios.",
                result: "MAE 13.6 pp test, Test Cov 80.0% (4/5), Train Cov 86.7% (best), MAE train 13.9 pp (best)",
              },
            ].map((item, i) => (
              <div key={item.v} className="flex gap-4 items-start">
                <div className="flex flex-col items-center shrink-0">
                  <div className={`w-8 h-8 rounded-sm flex items-center justify-center text-white text-xs font-bold ${item.color}`}>
                    {item.v}
                  </div>
                  {i < 3 && <div className="w-px h-3 bg-ki-border" />}
                </div>
                <div className="flex-1 bg-ki-surface-sunken border border-ki-border rounded-sm p-3">
                  <div className="flex items-baseline justify-between mb-1">
                    <span className="text-xs font-semibold text-ki-on-surface">{item.title}</span>
                    <span className="text-xs font-data text-ki-on-surface-muted">{item.date}</span>
                  </div>
                  <p className="text-xs text-ki-on-surface-secondary leading-relaxed mb-1">{item.desc}</p>
                  <div className="text-xs font-data text-ki-primary">{item.result}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Cross-version comparison table */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-4">Cross-Version Comparison</h3>
          <div className="my-6 overflow-x-auto rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Version</Th>
                  <Th>Strategy</Th>
                  <Th right>Scenarios</Th>
                  <Th right>MAE test</Th>
                  <Th right>MAE train</Th>
                  <Th right>RMSE test</Th>
                  <Th right>Cov 90% (train)</Th>
                  <Th right>Cov 90% (test)</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {VERSION_COMPARISON.map((r) => (
                  <tr key={r.version} className={!r.pollFree ? "bg-ki-primary/5" : ""}>
                    <Td mono bold>{r.version}</Td>
                    <Td>{r.strategy}</Td>
                    <Td mono right>{r.scenarios}</Td>
                    <Td mono right bold color={r.maeTest <= 11.4 ? "text-green-600" : r.maeTest >= 19 ? "text-red-600" : "text-ki-on-surface-secondary"}>
                      {r.maeTest.toFixed(1)} pp
                    </Td>
                    <Td mono right>{r.maeTrain.toFixed(1)} pp</Td>
                    <Td mono right>{r.rmseTest.toFixed(1)} pp</Td>
                    <Td mono right>{r.cov90train.toFixed(1)}%</Td>
                    <Td mono right>{r.cov90test !== null ? `${r.cov90test.toFixed(1)}%` : "n/r"}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-ki-on-surface-muted mt-1 mb-3">
            &dagger; v2.3 is poll-informed: the PubOp agent requires an initial polling observation,
            making it not directly comparable to poll-free versions (v2&ndash;v2.2).
            &ldquo;n/r&rdquo; = not reported in the calibration output.
            Coverage denominators differ across versions (34 train for v2/v2.1 vs 15 for v2.2/v2.3),
            limiting direct comparison.
          </p>

          {/* Version comparison bar chart — MAE only (coverage chart removed: mixing train/test was misleading) */}
          <div className="my-8">
            <h4 className="text-xs font-data text-ki-on-surface-muted uppercase tracking-wider mb-3 text-center">
              Test MAE Across Versions (pp)
            </h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={VERSION_COMPARISON} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d4d4d4" vertical={false} />
                <XAxis dataKey="version" tick={{ fontSize: 12, fill: "#8a8a8a", fontFamily: "monospace" }} />
                <YAxis domain={[0, 25]} tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #d4d4d4" }} formatter={(v: unknown) => `${Number(v).toFixed(1)} pp`} />
                <Bar dataKey="maeTest" name="MAE test" radius={[4, 4, 0, 0]}>
                  {VERSION_COMPARISON.map((r, i) => (
                    <Cell key={i} fill={r.maeTest <= 12 ? "#22c55e" : r.maeTest >= 19 ? "#f87171" : "#fbbf24"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Key insights */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-4">Key Insights</h3>
          <div className="space-y-3">
            {[
              {
                title: "v2.1: Grounding alone is insufficient",
                text: "Replacing LLM-generated events with Google Search verified events reduced RMSE by 5 pp but collapsed training coverage from 79.4% to 2.9%. The discrepancy model lost its ability to absorb scenario-specific variance when events became too precise.",
              },
              {
                title: "v2.2: Hybrid approach achieves best test MAE",
                text: "Combining original agents with verified events achieved the best test MAE (11.4 pp). Training coverage recovered to 73.3%. The discrepancy σ_b,between rose from 0.115 to 0.358 logit, indicating the model learned to separate domain bias from scenario noise.",
              },
              {
                title: "v2.3: Poll-informed PubOp trades accuracy for training calibration",
                text: "Adding a poll-informed Public Opinion agent increased test MAE by 2.2 pp (13.6 vs 11.4) but improved training coverage to 86.7%. Test-set coverage matches v2.2 at 80.0% (4/5). Note: v2.3 uses additional information (initial polling), so it is not an apples-to-apples comparison with poll-free versions. Training MAE of 13.9 pp was the lowest.",
              },
              {
                title: "Stable test coverage across versions",
                text: "Test-set coverage is remarkably stable: 75.0% (v2, N=8), 80.0% (v2.2, N=5), 80.0% (v2.3, N=5). The AMC Short Squeeze is the consistent test-set outlier across v2.2/v2.3 (35-41 pp error). The stability of test coverage despite different training strategies suggests the hierarchical discrepancy model provides robust uncertainty quantification, though small test sets (N=5-8) limit the precision of these estimates.",
              },
            ].map((item) => (
              <div key={item.title} className="bg-ki-surface-sunken border border-ki-border rounded-sm p-3">
                <div className="text-xs font-semibold text-ki-on-surface mb-1">{item.title}</div>
                <div className="text-xs text-ki-on-surface-secondary leading-relaxed">{item.text}</div>
              </div>
            ))}
          </div>

          {/* Financial outlier analysis */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-4">Financial Outlier Analysis</h3>
          <P>
            Across all versions, financial scenarios consistently exhibit the largest prediction errors.
            Three scenarios &mdash; Archegos Capital (65 pp error), AMC/GameStop, and SVB &mdash; share
            common failure patterns: rapid trust cascades, binary &ldquo;collapse vs survive&rdquo; dynamics,
            and contagion effects that the five-force model cannot express.
          </P>
          <div className="my-3 p-3 bg-[#fff8e1] border border-[#ffe082] rounded-sm">
            <p className="text-xs text-[#b45309] leading-relaxed">
              <strong>Structural limitation:</strong> Financial crisis dynamics are fundamentally different
              from opinion formation. The base model predicts gradual opinion shifts, but financial trust
              cascades are threshold-triggered and nonlinear. Excluding Archegos alone reduces v2 test MAE
              from 19.2 to 12.6 pp, illustrating the outsized influence of a single flagged scenario.
              However, this does not mean all non-financial predictions are accurate: Greek Bailout
              (political) also shows 26.6 pp error. Future work should consider domain-specific
              force functions for financial scenarios.
            </p>
          </div>

          {/* v1 → v2 comparison (kept for context) */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-4">From v1 to v2</h3>
          <div className="my-6 overflow-hidden rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Aspect</Th>
                  <Th>v1 (2025)</Th>
                  <Th>v2 (2026)</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {[
                  ["Calibration", "Grid search (972 combos)", "Hierarchical Bayesian SVI"],
                  ["Training data", "1,000 LLM-generated scenarios", "42 empirical scenarios"],
                  ["Parameters", "5 (all ad hoc)", "4 calibrable + 4 frozen (Sobol-justified)"],
                  ["Uncertainty", "None", "90% CI, test coverage 75.0% (N=8)"],
                  ["Model discrepancy", "None", "Explicit b_d + b_s"],
                  ["Validation", "Same data as training", "Held-out test set + SBC"],
                  ["Online assimilation", "None", "EnKF (1.8 pp, in-sample Brexit)"],
                  ["MAE (test)", "14.8 pp (synthetic)*", "19.2 pp full / 11.4 pp best (v2.2)"],
                ].map(([aspect, v1, v2]) => (
                  <tr key={aspect}>
                    <Td bold>{aspect}</Td>
                    <Td>{v1}</Td>
                    <Td>{v2}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-ki-on-surface-muted mt-1 mb-4">
            *Direct MAE comparison is not straightforward: v1 synthetic ground truth inherits LLM biases
            (epistemic circularity), likely understating true error.
          </p>
        </Section>

        {/* ── 8. Public Opinion Agent ───────────────── */}
        <Section id="pubop" title="9. Public Opinion Agent (v2.3, poll-informed)">
          <P>
            The key innovation in v2.3 is the <strong>Public Opinion (PubOp) agent</strong> &mdash; an implicit
            agent injected at simulation start that represents the aggregate polling signal. Rather than asking
            the LLM to guess the initial public opinion distribution, we anchor it to observed data.
          </P>

          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-4">Agent Design</h3>
          <div className="my-6 overflow-hidden rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Property</Th>
                  <Th>Value</Th>
                  <Th>Rationale</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {[
                  ["Type", "citizen", "Participates in citizen swarm dynamics, not elite/institutional"],
                  ["Initial position", "2 × (pro_pct / 100) − 1", "First available polling mapped to [−1, +1]"],
                  ["Rigidity", "0.1 (very low)", "Highly responsive to events — mirrors actual public opinion volatility"],
                  ["Tolerance", "0.9 (very high)", "Interacts with agents across the spectrum — broad social exposure"],
                  ["Influence", "0.7 (high)", "Exerts substantial pull on other citizens as an aggregate signal"],
                  ["Cluster size", "Proportional to poll N", "Weighted by sample size reliability"],
                ].map(([prop, val, reason]) => (
                  <tr key={prop}>
                    <Td bold>{prop}</Td>
                    <Td mono>{val}</Td>
                    <Td>{reason}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Eq>
            p<sub>pubop</sub> = 2 &middot; (pro_pct / 100) &minus; 1
            &nbsp;&nbsp;&nbsp; e.g. 41% &rarr; &minus;0.18 (lean negative)
          </Eq>

          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-4">Why It Works</h3>
          <P>
            The PubOp agent addresses a fundamental limitation: the simulation starts with LLM-generated
            agent positions that may be systematically biased. By injecting a high-influence, low-rigidity
            citizen that starts at the empirically observed position, the model gets an initial anchor that:
          </P>
          <ul className="space-y-1.5 text-xs text-ki-on-surface-secondary mb-3 ml-3">
            <li className="flex gap-2">
              <span className="text-ki-on-surface-muted shrink-0">&bull;</span>
              <span>Corrects systematic LLM bias toward centrist positions</span>
            </li>
            <li className="flex gap-2">
              <span className="text-ki-on-surface-muted shrink-0">&bull;</span>
              <span>Provides a &ldquo;gravity well&rdquo; that pulls citizen swarm toward observed reality</span>
            </li>
            <li className="flex gap-2">
              <span className="text-ki-on-surface-muted shrink-0">&bull;</span>
              <span>Low rigidity allows the anchor to drift with events (not frozen at initial poll)</span>
            </li>
            <li className="flex gap-2">
              <span className="text-ki-on-surface-muted shrink-0">&bull;</span>
              <span>Improves training-set coverage (73.3% &rarr; 86.7%); test coverage stable at 80.0% (4/5) for both v2.2 and v2.3</span>
            </li>
          </ul>

          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-4">Trade-off Analysis</h3>
          <P>
            The PubOp agent introduces a 2.2 pp increase in test MAE (13.6 vs 11.4 in v2.2) but achieves
            the highest training-set coverage (86.7%) and lowest training MAE (13.9 pp).
            Test-set coverage is identical for v2.2 and v2.3 at 80.0% (4/5), with AMC Short Squeeze as the consistent outlier.
            Note that v2.3 uses strictly more information than v2.2 (an initial polling observation),
            so the comparison is not apples-to-apples.
          </P>
          <div className="my-4 grid grid-cols-3 gap-3">
            <StatCard label="Test MAE cost" value="+2.2 pp" sub="vs v2.2" />
            <StatCard label="Train cov. gain" value="+13.4 pp" sub="73.3% → 86.7% (train)" />
            <StatCard label="Train MAE" value="13.9 pp" sub="best across versions" />
          </div>
        </Section>

        {/* ── 9. Limitations ──────────────────────── */}
        <Section id="limitations" title="10. Limitations & Future Work">
          <ul className="space-y-3 text-xs text-ki-on-surface-secondary">
            {[
              {
                title: "Financial domain performance & domain generalization",
                text: "Leave-one-domain-out CV (20 HQ scenarios, 4 domains) yields overall MAE 15.9 pp. Financial is the clear outlier (32.3 pp), consistent with known trust cascade limitations. Excluding financial: LODO MAE drops to 10.2 pp. Corporate (4.9 pp) and public health (9.8 pp) generalize well; political (12.6 pp) is moderate.",
              },
              {
                title: "Accuracy–coverage trade-off",
                text: "Test-set coverage is stable across versions: 75.0% (v2, N=8), 80.0% (v2.2, N=5), 80.0% (v2.3, N=5). AMC Short Squeeze is the consistent outlier. The stability suggests robust uncertainty quantification, but small test sets limit the precision of coverage estimates.",
              },
              {
                title: "Variational approximation",
                text: "SVI agrees with NUTS on dominant parameters but concentrates on weaker ones (α_social, α_event). Uncertainty may be underestimated — evidenced by v2.1's coverage collapse when the discrepancy model lost flexibility.",
              },
              {
                title: "Frozen parameter approximation & combined uncertainty",
                text: "A preliminary 5-parameter SVI confirms λ_citizen is identifiable: posterior mean λ=0.85, far from frozen default λ=0.25. Error bars are underestimated for two independent reasons: (i) SVI posterior is 5–13× narrower than NUTS on weak parameters, and (ii) freezing λ_citizen omits additional variance. These compound. A full production 5-parameter calibration with Phase A pre-training is planned.",
              },
              {
                title: "PubOp agent dependency on polling data",
                text: "The v2.3 PubOp agent requires at least one polling observation to set initial position. For scenarios without polling data, the agent cannot be constructed, limiting applicability.",
              },
              {
                title: "LLM stochasticity",
                text: "LLM outputs are treated as fixed inputs. A perturbation experiment (10 seeds, σ=0.1 noise on events) yields σ_LLM ≈ 3.2 pp on final predictions (range 9.6 pp), accounting for ~3-8% of total prediction variance. Full LLM rollout experiments with different API seeds are planned.",
              },
              {
                title: "Sample size",
                text: "42 scenarios (v2) reduced to 20 high-quality scenarios (v2.2/v2.3). Several domains have 1–2 scenarios, making domain-level inference unreliable.",
              },
              {
                title: "EnKF validation",
                text: "The Brexit case study uses a training-set scenario. Out-of-sample EnKF validation with held-out sequential polling is a priority for future work.",
              },
              {
                title: "Grounding scalability",
                text: "Google Search-based event grounding (v2.1/v2.2) requires manual curation. Automating event verification and integrating real-time news feeds is planned for v3.",
              },
            ].map((l) => (
              <li key={l.title} className="flex gap-3">
                <span className="text-ki-on-surface-muted mt-0.5 shrink-0">&bull;</span>
                <div>
                  <strong>{l.title}.</strong> {l.text}
                </div>
              </li>
            ))}
          </ul>
        </Section>

        {/* ── 11. Null-Baseline Benchmark (v2.5) ────── */}
        <Section id="null-baseline" title="11. Null-Baseline Benchmark (v2.5)">
          <div className="mb-4 p-3 bg-ki-primary/10 border border-ki-primary/30 rounded-sm">
            <p className="text-xs text-ki-on-surface leading-relaxed">
              <strong className="text-ki-primary">Added in v2.5.</strong>{" "}
              Null-baseline forecasters (persistence, running mean, OLS trend, AR(1)) evaluated against the same
              43 empirical trajectories, with Diebold&ndash;Mariano tests (Harvey&ndash;Leybourne&ndash;Newbold small-sample correction).
              The goal: establish a reproducible <strong>skill floor</strong> any calibrated sim must clear.
            </p>
          </div>

          <P>
            A calibrated simulator is only worth its complexity if it forecasts better than trivial statistical
            baselines. We evaluate four standard one-step-ahead forecasters on the 43 empirical scenarios used
            for calibration, testing pairwise predictive accuracy with the Diebold&ndash;Mariano statistic. At
            our sample sizes (n = 4&ndash;9 forecast pairs per scenario), the Harvey&ndash;Leybourne&ndash;Newbold
            correction is non-optional: uncorrected DM over-rejects on small samples.
          </P>

          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-3">
            Table 15a &mdash; Skill on normalized support (pro_pct / 100)
          </h3>
          <div className="my-4 overflow-hidden rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Baseline</Th>
                  <Th right>Mean RMSE</Th>
                  <Th right>Median RMSE</Th>
                  <Th right>Terminal err.</Th>
                  <Th right>Beats pers. (sig.)</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {BASELINE_SUPPORT.map((row) => (
                  <tr key={row.baseline} className={row.baseline === "Naive persistence" ? "bg-ki-primary/5" : ""}>
                    <Td bold>{row.baseline}</Td>
                    <Td right mono>{row.meanRmse.toFixed(3)}</Td>
                    <Td right mono>{row.medianRmse.toFixed(3)}</Td>
                    <Td right mono>{row.terminal.toFixed(3)}</Td>
                    <Td right mono>{row.beats}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-3">
            Table 15b &mdash; Skill on signed position ((pro &minus; against) / 100)
          </h3>
          <div className="my-4 overflow-hidden rounded-sm border border-ki-border">
            <table className="w-full">
              <thead>
                <tr className="bg-ki-surface-sunken border-b border-ki-border">
                  <Th>Baseline</Th>
                  <Th right>Mean RMSE</Th>
                  <Th right>Median RMSE</Th>
                  <Th right>Terminal err.</Th>
                  <Th right>Beats pers. (sig.)</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ki-border/50">
                {BASELINE_SIGNED.map((row) => (
                  <tr key={row.baseline} className={row.baseline === "Naive persistence" ? "bg-ki-primary/5" : ""}>
                    <Td bold>{row.baseline}</Td>
                    <Td right mono>{row.meanRmse.toFixed(3)}</Td>
                    <Td right mono>{row.medianRmse.toFixed(3)}</Td>
                    <Td right mono>{row.terminal.toFixed(3)}</Td>
                    <Td right mono>{row.beats}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <P>
            <strong>Three findings.</strong> (i)&nbsp;Naive persistence is a surprisingly strong baseline
            (mean RMSE 0.038 on support &mdash; a 3.8 pp one-step-ahead error). (ii)&nbsp;No baseline reliably
            dominates persistence: OLS linear trend wins at p &lt; 0.05 on only 4/43 (support) and 6/43 (signed)
            scenarios; running mean and AR(1) never beat persistence significantly. (iii)&nbsp;Skill is
            domain-heterogeneous by a factor of five.
          </P>

          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-8 mb-3">
            Table 16 &mdash; Persistence RMSE by domain (support)
          </h3>
          <div className="my-4">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={PERSISTENCE_BY_DOMAIN} layout="vertical" margin={{ left: 90, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d4d4d4" />
                <XAxis type="number" domain={[0, 0.1]} tick={{ fontSize: 11, fill: "#8a8a8a" }} />
                <YAxis type="category" dataKey="domain" tick={{ fontSize: 11, fill: "#8a8a8a" }} width={90} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #d4d4d4" }}
                  formatter={(v: unknown) => Number(v).toFixed(3)}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="persistence" name="Persistence" fill="#1a6dff" radius={[0, 3, 3, 0]} />
                <Bar dataKey="ols" name="OLS linear trend" fill="#1a6dff80" radius={[0, 3, 3, 0]} />
                <Bar dataKey="ar1" name="AR(1)" fill="#d4d4d4" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <P>
            Political polling is the easiest to forecast (persistence RMSE 0.012), financial the hardest (0.063)
            &mdash; a <strong>5&times; spread</strong> that mirrors the |b<sub>s</sub>|<sub>financial</sub> = 0.74
            logit-space bias of &sect;5. Both point to the same conclusion: financial scenarios exhibit
            regime-switching dynamics that neither simple persistence nor a stationary Gaussian discrepancy captures well.
          </P>

          {/* Residual Bootstrap Coverage */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-3">
            Empirical coverage via residual bootstrap (&sect;6.7)
          </h3>
          <P>
            A model-free complement to the logit-space credible intervals of &sect;5. For each scenario, we pool
            in-sample residuals and resample B = 500 to build predictive intervals at each round. Pooled across
            271 round-level observations:
          </P>
          <div className="my-4 grid grid-cols-2 gap-3">
            <StatCard label="Persistence coverage" value="88.4%" sub="CI [85.6%, 91.1%]" />
            <StatCard label="OLS trend coverage" value="89.2%" sub="CI [86.4%, 91.9%]" />
          </div>
          <P>
            Both nominal-90% intervals land inside the bootstrap band &mdash; the model-free check agrees with
            the parametric logit-space coverage of &sect;5 at the corpus aggregate level. Agreement across two
            independent calculation paths increases confidence that reported coverage is not an artifact of the
            logit parameterization.
          </P>

          {/* Scenario Matrix */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-3">
            Table 17 &mdash; Scenario-diversity matrix (7 &times; 5 &times; 4 = 140 cells)
          </h3>
          <P>
            We define a three-axis matrix on <strong>domain</strong> (7), <strong>region</strong> (inferred from
            ISO country code, 5), and <strong>tension</strong> (inferred from signed-position volatility, 4).
            The 43-scenario corpus occupies <strong>25/140 (18%) of cells</strong> &mdash; no axis value is
            empty, but the distribution is uneven: US + EU account for 88% (38/43).
          </P>
          <div className="my-4 grid grid-cols-1 sm:grid-cols-3 gap-4">
            {["Domain", "Region", "Tension"].map((axis) => (
              <div key={axis} className="border border-ki-border rounded-sm overflow-hidden">
                <div className="bg-ki-surface-sunken px-3 py-1.5 text-xs font-data uppercase tracking-wider text-ki-on-surface-muted border-b border-ki-border">
                  {axis}
                </div>
                <table className="w-full">
                  <tbody className="divide-y divide-ki-border/50">
                    {SCENARIO_MATRIX.filter((r) => r.axis === axis).map((r) => (
                      <tr key={`${axis}-${r.value}`}>
                        <Td mono>{r.value}</Td>
                        <td className="px-3 py-1.5">
                          <BarInline value={r.count} max={26} color="bg-ki-primary" />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
          <P>
            <strong>Highest-marginal-value additions:</strong> (i) APAC / LATAM scenarios across any domain,
            (ii) financial scenarios with <code>critical</code> tension (currently 6 total), representing the
            most diagnostic regime for model discrimination.
          </P>

          {/* Reproducibility */}
          <h3 className="text-xs font-bold uppercase tracking-[0.04em] text-ki-on-surface mt-10 mb-3">
            Reproducibility (Appendix D)
          </h3>
          <P>
            All numbers above are produced by <code>benchmarks/</code>, a self-contained Python module with
            <strong> 103 automated tests</strong> (runtime 2.94 s). Two one-line commands regenerate the
            full reference distribution:
          </P>
          <Eq>
            $ python -m benchmarks.historical_runner --out outputs/historical_benchmark.json{"\n"}
            $ python -m benchmarks --out outputs/benchmark_report.json
          </Eq>
          <P>
            Any modification to the simulator, calibration pipeline, or EnKF can be A/B-tested against this
            v2.5 reference by running the commands before and after. Disagreement on any DM cell at
            p &lt; 0.05 constitutes a reproducibility-relevant regression.
          </P>
        </Section>

        {/* Footer */}
        <div className="mt-12 pt-4 border-t border-ki-border">
          <div className="flex items-center justify-between text-xs text-ki-on-surface-muted">
            <span>DigitalTwinSim &mdash; Technical Paper v2.8</span>
            <span>April 2026</span>
          </div>
        </div>
      </article>
    </main>
  );
}

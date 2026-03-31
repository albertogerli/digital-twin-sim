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
  { domain: "Energy", n: 1, bd: -0.026, bs: 0.709, bias: "Over-predicts" },
  { domain: "Public Health", n: 4, bd: -0.032, bs: 0.534, bias: "Over-predicts" },
  { domain: "Corporate", n: 6, bd: -0.040, bs: 0.432, bias: "Over-predicts" },
  { domain: "Political", n: 11, bd: +0.019, bs: 0.198, bias: "No systematic" },
  { domain: "Technology", n: 2, bd: -0.021, bs: 0.066, bias: "No systematic" },
];

/* ── Components ───────────────────────────────────── */

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mb-16">
      <h2 className="text-2xl font-bold text-gray-900 mb-6 pb-2 border-b border-gray-200">
        {title}
      </h2>
      {children}
    </section>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-[15px] text-gray-700 leading-[1.8] mb-4">{children}</p>;
}

function Eq({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-6 p-4 bg-gray-50 border border-gray-200 rounded-lg font-mono text-sm text-gray-800 overflow-x-auto">
      {children}
    </div>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return (
    <th className={`text-xs font-mono text-gray-500 uppercase tracking-wider px-4 py-3 ${right ? "text-right" : "text-left"}`}>
      {children}
    </th>
  );
}

function Td({ children, mono, right, bold, color }: { children: React.ReactNode; mono?: boolean; right?: boolean; bold?: boolean; color?: string }) {
  return (
    <td
      className={`px-4 py-2.5 text-sm ${mono ? "font-mono" : ""} ${right ? "text-right" : ""} ${bold ? "font-semibold" : ""} ${color || "text-gray-700"}`}
    >
      {children}
    </td>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 text-center">
      <div className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-2">{label}</div>
      <div className="text-3xl font-bold text-gray-900">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  );
}

function BarInline({ value, max, color }: { value: number; max: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-100 rounded-full h-2.5 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${(value / max) * 100}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-500 w-10 text-right">{value.toFixed(3)}</span>
    </div>
  );
}

/* ── Page ──────────────────────────────────────────── */

export default function PaperPage() {
  return (
    <main className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="text-sm text-gray-400 hover:text-gray-600 transition-colors">
            DigitalTwinSim
          </Link>
          <span className="text-xs text-gray-400 font-mono">Technical Paper v2.3</span>
        </div>
      </header>

      {/* Title block */}
      <div className="max-w-4xl mx-auto px-6 pt-16 pb-12">
        <p className="text-sm font-mono text-blue-600 mb-4 uppercase tracking-wider">Working Paper</p>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 leading-tight mb-6">
          Bayesian Calibration and Online Data Assimilation for LLM-Agent Opinion Dynamics
        </h1>
        <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-gray-500 mb-2">
          <span>Alberto Giovanni Gerli</span>
          <span className="font-mono">v2.3 &mdash; March 2026</span>
        </div>
        <div className="text-xs text-gray-400 mb-8">
          Tourbillon Tech Srl &middot; Universit&agrave; degli Studi di Milano
        </div>

        {/* Abstract */}
        <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
          <h3 className="text-xs font-mono uppercase text-gray-400 tracking-wider mb-3">Abstract</h3>
          <p className="text-[15px] text-gray-700 leading-[1.8]">
            We present DigitalTwinSim, a framework that combines LLM-driven agent-based simulation with
            Bayesian calibration and online data assimilation for modeling public opinion dynamics.
            A force-based system with five competing mechanisms &mdash; direct LLM influence, social
            conformity, herd behavior, anchor rigidity, and exogenous shocks &mdash; is combined through
            a gauge-fixed softmax mixture. A three-level hierarchical Bayesian model with explicit readout
            discrepancy is calibrated via SVI on <strong>42 empirical scenarios</strong> spanning 10 domains,
            achieving <strong>12.6 pp MAE</strong> on verified held-out scenarios with <strong>85.7% coverage</strong> of
            90% credible intervals. An Ensemble Kalman Filter enables online assimilation, reducing
            prediction error to <strong>1.8 pp</strong> on the Brexit case study &mdash; a 77% improvement over
            polling baselines.
          </p>
        </div>

        {/* Key stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-8">
          <StatCard label="MAE (verified)" value="12.6 pp" sub="N=7 test scenarios" />
          <StatCard label="Coverage 90%" value="85.7%" sub="credible intervals" />
          <StatCard label="EnKF error" value="1.8 pp" sub="Brexit, 6 polls" />
          <StatCard label="Scenarios" value="42" sub="10 domains, empirical" />
        </div>
      </div>

      {/* Table of Contents */}
      <article className="max-w-4xl mx-auto px-6 pb-24">
        <nav className="mb-16 p-6 bg-gray-50 rounded-xl border border-gray-200">
          <h3 className="text-xs font-mono uppercase text-gray-400 tracking-wider mb-3">Contents</h3>
          <ol className="space-y-1.5 text-sm">
            {[
              ["five-forces", "1. Five-Force Opinion Dynamics"],
              ["hierarchy", "2. Hierarchical Bayesian Calibration"],
              ["results", "3. Calibration Results"],
              ["discrepancy", "4. Readout Discrepancy"],
              ["sensitivity", "5. Sensitivity Analysis"],
              ["enkf", "6. Online Assimilation (EnKF)"],
              ["evolution", "7. Evolution from v1 to v2"],
              ["limitations", "8. Limitations"],
            ].map(([id, label]) => (
              <li key={id}>
                <a href={`#${id}`} className="text-blue-600 hover:text-blue-800 transition-colors">
                  {label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

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
              <div key={f.name} className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <div className="flex items-baseline justify-between mb-1">
                  <span className="text-sm font-semibold text-gray-900">{f.name}</span>
                  <span className="text-xs font-mono text-blue-600">&pi; = {f.pi}</span>
                </div>
                <div className="text-xs font-mono text-gray-400 mb-2">&alpha; = {f.alpha}</div>
                <div className="text-xs text-gray-600 leading-relaxed">{f.desc}</div>
              </div>
            ))}
          </div>
          {/* Radar chart: softmax weights */}
          <div className="my-8 flex justify-center">
            <div className="w-full max-w-md">
              <h4 className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4 text-center">
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
                  <PolarGrid stroke="#e5e7eb" />
                  <PolarAngleAxis dataKey="force" tick={{ fontSize: 12, fill: "#6b7280" }} />
                  <PolarRadiusAxis angle={90} domain={[0, 30]} tick={{ fontSize: 10, fill: "#9ca3af" }} />
                  <Radar dataKey="weight" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} strokeWidth={2} />
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

        {/* ── 2. Hierarchical Bayesian ─────────────── */}
        <Section id="hierarchy" title="2. Hierarchical Bayesian Calibration">
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
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold ${
                    i === 0 ? "bg-blue-600" : i === 1 ? "bg-blue-500" : "bg-blue-400"
                  }`}>{i + 1}</div>
                  {i < 2 && <div className="w-px h-4 bg-blue-200" />}
                </div>
                <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-4">
                  <div className="text-sm font-semibold text-gray-900 mb-1">{l.level}</div>
                  <div className="font-mono text-xs text-gray-600 mb-1">{l.eq}</div>
                  <div className="text-xs text-gray-500">{l.desc}</div>
                </div>
              </div>
            ))}
          </div>

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
        </Section>

        {/* ── 3. Calibration Results ───────────────── */}
        <Section id="results" title="3. Calibration Results">
          <P>
            The model is calibrated on 42 empirical scenarios (34 train / 8 test) across 10 domains,
            each with a ground-truth final outcome and, where available, intermediate polling data.
          </P>

          {/* Global posterior */}
          <h3 className="text-lg font-semibold text-gray-900 mt-8 mb-4">Calibrated Global Posterior</h3>
          <div className="my-6 overflow-hidden rounded-xl border border-gray-200">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <Th>Parameter</Th>
                  <Th right>Mean</Th>
                  <Th right>95% CI</Th>
                  <Th>Interpretation</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
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
          <h3 className="text-lg font-semibold text-gray-900 mt-10 mb-4">Test Set Performance</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-6">
            <StatCard label="MAE (verified, N=7)" value="12.6 pp" />
            <StatCard label="MAE (full, N=8)" value="19.2 pp" />
            <StatCard label="Coverage 90%" value="85.7%" sub="verified only" />
          </div>

          <div className="my-6 overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <Th>Scenario</Th>
                  <Th>Domain</Th>
                  <Th right>GT %</Th>
                  <Th right>Predicted</Th>
                  <Th right>Error</Th>
                  <Th right>90% CI</Th>
                  <Th right>Cov</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {TEST_RESULTS.map((r) => (
                  <tr key={r.scenario} className={r.scenario.includes("*") ? "bg-yellow-50/50" : ""}>
                    <Td bold>{r.scenario}</Td>
                    <Td>{r.domain}</Td>
                    <Td mono right>{r.gt}</Td>
                    <Td mono right>{r.pred}</Td>
                    <Td mono right bold color={r.err > 20 ? "text-red-600" : r.err < 10 ? "text-green-600" : "text-gray-700"}>
                      {r.err.toFixed(1)}
                    </Td>
                    <Td mono right>{r.ci}</Td>
                    <Td right>{r.covered ? "\u2713" : "\u2717"}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-400 mt-2 mb-6">
            *NEEDS_VERIFICATION &mdash; excluded from primary metrics. CI computed in logit space and back-transformed.
          </p>

          {/* Test set error chart */}
          <div className="my-8">
            <h4 className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4 text-center">
              Test Set: Prediction Error by Scenario (pp)
            </h4>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={TEST_RESULTS.map(r => ({ name: r.scenario.replace("*", ""), err: r.err, covered: r.covered }))}
                layout="vertical"
                margin={{ left: 100, right: 30, top: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis type="number" domain={[0, 70]} tick={{ fontSize: 11, fill: "#9ca3af" }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "#6b7280" }} width={100} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} formatter={(v: unknown) => `${Number(v).toFixed(1)} pp`} />
                <Bar dataKey="err" name="Error (pp)" radius={[0, 4, 4, 0]}>
                  {TEST_RESULTS.map((r, i) => (
                    <Cell key={i} fill={r.err > 20 ? "#f87171" : r.err < 10 ? "#4ade80" : "#fbbf24"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-green-400 inline-block" /> &lt; 10 pp</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-yellow-400 inline-block" /> 10&ndash;20 pp</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-red-400 inline-block" /> &gt; 20 pp</span>
            </div>
          </div>
        </Section>

        {/* ── 4. Discrepancy ───────────────────────── */}
        <Section id="discrepancy" title="4. Readout Discrepancy">
          <P>
            The within-scenario discrepancy (&sigma;<sub>b,within</sub> = 0.558 logit) is ~5&times; the
            between-domain discrepancy (&sigma;<sub>b,between</sub> = 0.115 logit), indicating that
            scenario-specific factors dominate over systematic domain-level biases. In practical terms,
            0.558 logit translates to ~12&ndash;14 pp of irreducible prediction error.
          </P>

          <div className="my-6 overflow-hidden rounded-xl border border-gray-200">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <Th>Domain</Th>
                  <Th right>N</Th>
                  <Th right>b<sub>d</sub></Th>
                  <Th right>Mean |b<sub>s</sub>|</Th>
                  <Th>Bias</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {DOMAIN_DISCREPANCY.map((d) => (
                  <tr key={d.domain}>
                    <Td bold>{d.domain}</Td>
                    <Td mono right>{d.n}</Td>
                    <Td mono right>{d.bd > 0 ? "+" : ""}{d.bd.toFixed(3)}</Td>
                    <Td mono right bold color={d.bs > 0.5 ? "text-red-600" : "text-gray-700"}>{d.bs.toFixed(3)}</Td>
                    <Td>{d.bias}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Discrepancy chart */}
          <div className="my-8">
            <h4 className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4 text-center">
              Mean Scenario Discrepancy |b<sub>s</sub>| by Domain (logit space)
            </h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={DOMAIN_DISCREPANCY} margin={{ left: 20, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
                <XAxis dataKey="domain" tick={{ fontSize: 11, fill: "#6b7280" }} />
                <YAxis domain={[0, 0.8]} tick={{ fontSize: 11, fill: "#9ca3af" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} formatter={(v: unknown) => Number(v).toFixed(3)} />
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
            training data (N=11), show the lowest systematic bias.
          </P>
        </Section>

        {/* ── 5. Sensitivity ──────────────────────── */}
        <Section id="sensitivity" title="5. Sensitivity Analysis">
          <P>
            Variance-based global sensitivity analysis (Sobol indices, N=1024 Saltelli samples, 18,432 simulator
            evaluations) decomposes output variance into contributions from individual parameters and their
            interactions. This justifies the partition of 8 parameters into 4 calibrable and 4 frozen.
          </P>

          <div className="my-6 overflow-hidden rounded-xl border border-gray-200">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <Th>Parameter</Th>
                  <Th>S<sub>1</sub> (Main)</Th>
                  <Th>S<sub>T</sub> (Total)</Th>
                  <Th right>Type</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {SOBOL_DATA.map((s) => (
                  <tr key={s.param} className={s.type === "Frozen" ? "bg-gray-50/50" : ""}>
                    <Td mono bold>{s.param}</Td>
                    <td className="px-4 py-2.5">
                      <BarInline value={s.s1} max={0.4} color="bg-blue-400" />
                    </td>
                    <td className="px-4 py-2.5">
                      <BarInline value={s.st} max={0.6} color="bg-blue-600" />
                    </td>
                    <Td right>
                      <span className={`text-xs font-mono px-2 py-0.5 rounded ${
                        s.type === "Calibrable" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-500"
                      }`}>{s.type}</span>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Sobol bar chart */}
          <div className="my-8">
            <h4 className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4 text-center">
              Sobol Sensitivity Indices
            </h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={SOBOL_DATA} layout="vertical" margin={{ left: 60, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis type="number" domain={[0, 0.6]} tick={{ fontSize: 11, fill: "#9ca3af" }} />
                <YAxis type="category" dataKey="param" tick={{ fontSize: 11, fill: "#6b7280", fontFamily: "monospace" }} width={60} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  formatter={(v: unknown) => Number(v).toFixed(3)}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="s1" name="S₁ (Main)" fill="#93c5fd" radius={[0, 3, 3, 0]} />
                <Bar dataKey="st" name="Sₜ (Total)" fill="#3b82f6" radius={[0, 3, 3, 0]} />
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
          <h3 className="text-lg font-semibold text-gray-900 mt-10 mb-4">Simulation-Based Calibration</h3>
          <P>
            SBC (Talts et al., 2018) confirms the generative model is well-specified under NUTS: all 6 parameters
            pass KS uniformity (p &gt; 0.20). A separate SVI vs NUTS comparison shows agreement on dominant
            parameters (&alpha;<sub>herd</sub>: |&Delta;&mu;|/&sigma; = 0.15; &alpha;<sub>anchor</sub>: 0.35) but
            divergence on weaker ones (&alpha;<sub>social</sub>: 0.85; &alpha;<sub>event</sub>: 1.71), with SVI
            standard deviations 5&ndash;13&times; narrower than NUTS estimates.
          </P>
        </Section>

        {/* ── 6. EnKF ─────────────────────────────── */}
        <Section id="enkf" title="6. Online Assimilation (EnKF)">
          <P>
            The Ensemble Kalman Filter bridges offline calibration to real-time operation. It maintains an
            augmented state vector [&theta;, z] combining the 4 model parameters with n agent positions,
            jointly updated as streaming observations arrive.
          </P>

          <h3 className="text-lg font-semibold text-gray-900 mt-8 mb-4">Brexit Case Study (GT = 51.89%)</h3>
          <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-3 mb-4">
            Brexit is part of the calibration training set. This case study demonstrates EnKF operational
            mechanics, not an independent out-of-sample validation.
          </p>

          <div className="my-6 overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <Th>Round</Th>
                  <Th right>Observation</Th>
                  <Th right>EnKF Mean</Th>
                  <Th right>90% CI</Th>
                  <Th right>Width (pp)</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {ENKF_ROUNDS.map((r) => (
                  <tr key={r.round}>
                    <Td mono bold>{r.round}</Td>
                    <Td mono right>{r.obs}</Td>
                    <Td mono right>{r.mean.toFixed(1)}%</Td>
                    <Td mono right>{r.ci}</Td>
                    <Td mono right bold color={r.width < 0.3 ? "text-green-600" : "text-gray-700"}>
                      {r.width.toFixed(1)}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* EnKF trajectory chart */}
          <div className="my-8">
            <h4 className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4 text-center">
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
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#9ca3af" }} />
                <YAxis domain={[38, 54]} tick={{ fontSize: 11, fill: "#9ca3af" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} />
                <ReferenceLine y={51.89} stroke="#ef4444" strokeDasharray="6 3" strokeWidth={2} label={{ value: "GT 51.89%", position: "right", fontSize: 10, fill: "#ef4444" }} />
                <Area type="monotone" dataKey="ciHi" stackId="ci" stroke="none" fill="#dbeafe" fillOpacity={0} />
                <Area type="monotone" dataKey="ciLo" stackId="ci" stroke="none" fill="#dbeafe" fillOpacity={0.6} />
                <Area type="monotone" dataKey="mean" stroke="#2563eb" strokeWidth={2.5} fill="none" dot={{ r: 4, fill: "#2563eb" }} name="EnKF Mean" />
                <Area type="monotone" dataKey="poll" stroke="#22c55e" strokeWidth={0} fill="none" dot={{ r: 5, fill: "#22c55e", stroke: "#16a34a", strokeWidth: 2 }} name="Poll" connectNulls={false} />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-blue-600 inline-block" /> EnKF mean</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-green-500 inline-block" /> Polls</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-red-500 inline-block border-dashed" /> Ground truth</span>
            </div>
          </div>

          <P>
            Despite polls systematically underestimating Leave by ~8 pp, the dynamics model produces a final
            prediction of 50.1% (error 1.8 pp). The final CI [50.0, 50.4] does not cover the ground truth
            (51.89%), indicating the ensemble is under-dispersed &mdash; accurate point prediction but
            overconfident uncertainty.
          </P>

          <h3 className="text-lg font-semibold text-gray-900 mt-8 mb-4">Baseline Comparison</h3>
          <div className="my-6 overflow-hidden rounded-xl border border-gray-200">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <Th>Method</Th>
                  <Th right>Prediction</Th>
                  <Th right>Error (pp)</Th>
                  <Th right>Dynamics</Th>
                  <Th right>Params</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {ENKF_BASELINES.map((b) => (
                  <tr key={b.method}>
                    <Td bold>{b.method}</Td>
                    <Td mono right>{b.pred.toFixed(1)}%</Td>
                    <Td mono right bold color={b.err < 3 ? "text-green-600" : b.err > 7 ? "text-red-600" : "text-gray-700"}>
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
            <h4 className="text-xs font-mono text-gray-400 uppercase tracking-wider mb-4 text-center">
              Prediction Error by Method (pp)
            </h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={ENKF_BASELINES} margin={{ left: 20, right: 20, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
                <XAxis dataKey="method" tick={{ fontSize: 10, fill: "#6b7280" }} interval={0} />
                <YAxis domain={[0, 10]} tick={{ fontSize: 11, fill: "#9ca3af" }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }} formatter={(v: unknown) => `${Number(v).toFixed(1)} pp`} />
                <Bar dataKey="err" name="Error (pp)" radius={[4, 4, 0, 0]}>
                  {ENKF_BASELINES.map((b, i) => (
                    <Cell key={i} fill={b.dynamics ? "#22c55e" : "#f87171"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-red-400 inline-block" /> No dynamics</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-green-500 inline-block" /> With dynamics model</span>
            </div>
          </div>

          <P>
            The EnKF reduces error by <strong>77%</strong> vs last-poll baseline (1.8 vs 7.9 pp)
            and <strong>80%</strong> vs running average (1.8 vs 9.1 pp).
          </P>
        </Section>

        {/* ── 7. v1 → v2 ──────────────────────────── */}
        <Section id="evolution" title="7. Evolution from v1 to v2">
          <div className="my-6 overflow-hidden rounded-xl border border-gray-200">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <Th>Aspect</Th>
                  <Th>v1 (2025)</Th>
                  <Th>v2 (this paper)</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {[
                  ["Calibration", "Grid search (972 combos)", "Hierarchical Bayesian SVI"],
                  ["Training data", "1,000 LLM-generated scenarios", "42 empirical scenarios"],
                  ["Parameters", "5 (all ad hoc)", "4 calibrable + 4 frozen (Sobol-justified)"],
                  ["Uncertainty", "None", "90% CI, coverage 85.7%"],
                  ["Model discrepancy", "None", "Explicit b_d + b_s"],
                  ["Validation", "Same data as training", "Held-out test set + SBC"],
                  ["Online assimilation", "None", "EnKF (1.8 pp with polling)"],
                  ["MAE (test)", "14.8 pp (synthetic)*", "12.6 pp (empirical)*"],
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
          <p className="text-xs text-gray-400 mt-2 mb-6">
            *Direct MAE comparison is not straightforward: v1 synthetic ground truth inherits LLM biases
            (epistemic circularity), likely understating true error. The v1 &rarr; v2 improvement reflects
            both better calibration and elimination of this circularity.
          </p>
        </Section>

        {/* ── 8. Limitations ──────────────────────── */}
        <Section id="limitations" title="8. Limitations">
          <ul className="space-y-4 text-[15px] text-gray-700">
            {[
              {
                title: "Financial domain performance",
                text: "Systematic over-prediction in financial crises (mean |b_s| = 0.744 logit). Trust cascades and contagion dynamics exceed the base model's expressiveness.",
              },
              {
                title: "Variational approximation",
                text: "SVI agrees with NUTS on dominant parameters but concentrates on weaker ones (α_social, α_event). Uncertainty may be underestimated.",
              },
              {
                title: "Step size sensitivity",
                text: "Frozen λ_citizen shows up to 7.9 pp MAE variation under ±40% perturbation. Should be promoted to calibrable in future work.",
              },
              {
                title: "LLM stochasticity",
                text: "LLM outputs are treated as fixed inputs. Different seeds produce different narratives. This variance is absorbed into b_s but not explicitly separated from structural model error.",
              },
              {
                title: "Sample size",
                text: "42 scenarios (34 training) is small. Several domains have 1–2 scenarios, making domain-level inference unreliable. 4 domains have zero test scenarios.",
              },
              {
                title: "EnKF validation",
                text: "The Brexit case study uses a training-set scenario. Out-of-sample EnKF validation with held-out sequential polling is a priority for future work.",
              },
            ].map((l) => (
              <li key={l.title} className="flex gap-3">
                <span className="text-gray-300 mt-0.5 shrink-0">&bull;</span>
                <div>
                  <strong>{l.title}.</strong> {l.text}
                </div>
              </li>
            ))}
          </ul>
        </Section>

        {/* Footer */}
        <div className="mt-20 pt-8 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>DigitalTwinSim &mdash; Technical Paper v2.3</span>
            <span>March 2026</span>
          </div>
        </div>
      </article>
    </main>
  );
}

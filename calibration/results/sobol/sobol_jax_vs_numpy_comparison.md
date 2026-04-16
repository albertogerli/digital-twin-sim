# Sobol Sensitivity: JAX vs NumPy Model Comparison

## Context

Phase 0 (Sobol analysis, gauge fixing, freeze decisions) was performed on the
NumPy DynamicsV2 model. The JAX port introduces structural changes:

| Feature | NumPy | JAX (realigned) |
|---|---|---|
| Force standardization | Rolling buffer (deque maxlen=8) | EMA (tail-8, decay=0.3) |
| Herd activation | Hard step: `\|gap\| > threshold` | Smooth sigmoid: `σ((\|gap\| - θ)/0.02)` |
| Bounded confidence | Hard: `distance < tolerance` | Smooth sigmoid |
| Delta clamp | `min(max(...), cap)` | `cap * tanh(δ/cap)` |
| Social/herd data | Feed-based (random 5 posts) | Sparse K=5 random neighbors |

**Question**: Do Phase 0 conclusions (parameter hierarchy, freeze decisions) hold?

## Design

- SALib Sobol, N=1024, D=8, total=18432 evaluations
- 30 agents, 7 rounds
- JAX evaluation time: 2.76s (vs ~120s for NumPy sequential)
- Identical parameter bounds and Sobol design matrix

## Results: S1 and ST Comparison

| Parameter | S1 (NumPy) | S1 (JAX) | ST (NumPy) | ST (JAX) | Status |
|---|---|---|---|---|---|
| alpha_anchor | 0.2068 | 0.3793 | 0.4523 | 0.5509 | CONFIRMED_ACTIVE |
| alpha_herd | 0.3639 | 0.2587 | 0.5546 | 0.4454 | CONFIRMED_ACTIVE |
| alpha_social | 0.0857 | 0.0699 | 0.2125 | 0.1641 | CONFIRMED_ACTIVE |
| alpha_event | 0.0256 | 0.0051 | 0.1152 | 0.0643 | FROZEN_IN_JAX |
| lambda_citizen | 0.0021 | 0.0219 | 0.1206 | 0.0641 | UNFROZEN_IN_JAX |
| lambda_elite | 0.0070 | 0.0092 | 0.0158 | 0.0260 | CONFIRMED |
| herd_threshold | 0.0032 | 0.0060 | 0.0243 | 0.0188 | CONFIRMED |
| anchor_drift_rate | 0.0029 | -0.0045 | 0.0127 | 0.0080 | CONFIRMED |

Sum(S1): NumPy=0.6972, JAX=0.7456

Sum(ST): NumPy=1.5080, JAX=1.3416

## Verification Checks

### 1. α_herd still dominant?

ST ranking (JAX): alpha_anchor, alpha_herd, alpha_social, alpha_event
ST ranking (NumPy): alpha_herd, alpha_anchor, alpha_social, lambda_citizen
**Result: NO — STRUCTURE CHANGED** — α_herd is NOT #1 by ST

### 2. Frozen parameters still S1 < 0.01?

- lambda_elite: S1=0.0092 < 0.01 OK
- lambda_citizen: S1=0.0219 **>= 0.01 — REVIEW NEEDED**
- herd_threshold: S1=0.0060 < 0.01 OK
- anchor_drift_rate: S1=-0.0045 < 0.01 OK

**Result: FREEZE DECISION NEEDS REVIEW**

### 3. α_herd × α_anchor dominant interaction?

Top S2 (NumPy): alpha_herd:alpha_anchor = 0.0940
Top S2 (JAX): alpha_herd:alpha_anchor = 0.0669
**Result: CONFIRMED**

### 4. Top-4 ST ranking preserved?

NumPy top-4: ['alpha_herd', 'alpha_anchor', 'alpha_social', 'lambda_citizen']
JAX top-4: ['alpha_anchor', 'alpha_herd', 'alpha_social', 'alpha_event']
**Result: CHANGED**

## Verdict

**1/4 CHECKS PASSED — REVIEW REQUIRED**
Significant structural differences detected. Freeze decisions should
be re-evaluated before proceeding with calibration.

## Output Statistics Comparison

| Stat | NumPy | JAX |
|---|---|---|
| mean | 26.35 | 19.69 |
| std | 21.17 | 19.67 |
| min | 0.00 | 0.00 |
| max | 69.23 | 58.65 |

#!/usr/bin/env python3
"""Full calibration run with production parameters.

Runs in foreground with unbuffered output.
"""
import sys
import time
import json
import jax.numpy as jnp

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from src.inference.calibration_pipeline import (
    run_phase_a, run_phase_b, run_phase_c, save_results, RESULTS_DIR
)

t_total = time.time()

# ── PHASE A: 50 synthetic scenarios, 2000 steps ──
# 50 scenarios: JIT ~30s, per-step ~0.36s → total ~13 min
pa = run_phase_a(
    n_steps=2000, base_lr=0.005,
    n_agents=10, max_scenarios=50, seed=42, log_every=200,
)

# Save Phase A checkpoint
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
prior = {
    'mu_global': pa['mu_global_mean'].tolist(),
    'sigma_global': pa['sigma_global_mean'].tolist(),
    'n_synthetic_scenarios': pa['n_scenarios'],
    'n_domains': pa['n_domains'],
    'domain_names': pa['domain_names'],
    'elapsed_s': pa['elapsed_s'],
    'final_loss': float(pa['losses'][-1]),
    'losses_every_10': pa['losses'][::10].tolist(),
}
with open(RESULTS_DIR / 'synthetic_prior.json', 'w') as f:
    json.dump(prior, f, indent=2)
print(f'Saved synthetic_prior.json')
sys.stdout.flush()

# ── PHASE B: empirical fine-tuning, 3000 steps ──
pb = run_phase_b(pa, n_steps=3000, lr=0.002, seed=42, log_every=300)

# ── PHASE C: validation ──
pc = run_phase_c(pb, phase_a_result=pa, n_posterior_samples=200, seed=42)

# ── Save everything ──
save_results(pa, pb, pc)

t_end = time.time()
print(f'\n{"="*60}')
print(f'TOTAL TIME: {(t_end - t_total)/60:.1f} min')
print(f'  Phase A: {pa["elapsed_s"]/60:.1f} min')
print(f'  Phase B: {pb["elapsed_s"]/60:.1f} min')
print(f'  Phase C: {pc["elapsed_s"]/60:.1f} min')
print(f'{"="*60}')

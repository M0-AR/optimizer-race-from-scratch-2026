# PAPER DRAFT (IMRaD) — From Valley to Pilot: Optimizer Dynamics From Scratch, Two Inventions, Live-Market Transfer

## Abstract
We reproduce seven optimizers in invention order (NumPy, 6,634-param MLP, 1297/500 digits split) and verify
a narrated benchmark's claims within small margins. Auditing our own port surfaces two real bugs (dropped
bias momenta; inverted Nesterov blend), fixed and re-verified. The market-transfer experiment (live BTC/USD)
shows vision rankings compress near coin-flip and motivates two original contributions: (i) **SpectraMix**,
stable-rank-gated Muon↔Adam blending with exact endpoints (digits 400-step 0.058/95.2%); (ii) **Referee**,
zero-scout per-step empirical update selection, whose loss votes rediscover "adaptive early, spectral late"
(Hedge 200-step 0.053/95.6%, greedy 400-step 0.052/95.6%); (iii) **Pilot**, step-size-free direction+scale
election with DoG base and z-guard — 400-step 0.019/96.0% with zero learning rates. Every number is executed
output; failures and autopsies are reported alongside wins.

## 1. Introduction
Optimizers are update rules; Adam trained most large models of the last decade not because its ceiling is much
higher but because its basin is much wider (robust LRs/101: SGD 11, Momentum 37, Adam 52 — measured). At LLM
scale one run costs millions and nobody gets a 17-try sweep: width beats peak, and the logical endpoint is
abolishing the sweep entirely (§5.8).

## 2. Related work
Cauchy 1847 → Robbins-Monro 1951 → Polyak 1964 → RMSprop lecture 2012 → Adam 2014 → AdamW 2017 → Muon 2024 →
MuonClip/Kimi K2 2025 (1.04T MoE, 15.5T tokens, zero spikes) → ICLR 2026 Muon+Newton-Schulz convergence →
2026 spectral family (Muon^p fractional powers, DynMuon stage scheduling, Musec clipping, matrix-factorization
reassessments) → epoch-level selection (ROR tournaments, AOS switching, OptiRoulette, RL policies) →
parameter-free line (DoG/DoWG/D-Adaptation/Prodigy/DAoG 2025 decay) → instability monitoring (ZClip adaptive
clipping, Spectral-Alignment early warning, R-metric, TIOI scaling law). Gaps we occupy: LR-free × spectral
(empty), per-step zero-scout selection (ROR's cost curve pushed to s→0), guards inside (not beside) optimizers.

## 3. Method
### 3.1 Valley L=½x²+50y² (κ=100, reverse-engineered from start/loss/gradients; speed limit lr<0.02)
### 3.2 MLP 64-32×5-10 (6634 params), He init, ReLU, softmax-CE, batch 32, sklearn digits 1297/500
### 3.3 Protocols: quick (5 LRs × 3 seeds × 200 steps) and full (17 × 5 × 400), median reported, loss<0.1
works; valley robustness over 101 LRs; gradient-spread diagnostic (measured 46,052×)
### 3.4 Market transfer: live CoinGecko BTC/USD (+Frankfurter EUR/USD context), 21/30-day log-return
windows, direction labels, 80/20 time split; fair grid (5 LRs × 5 seeds) for contender comparisons.
Sources recorded per run; synthetic AR(2) fallback explicitly flagged, never silent.
### 3.5 Contenders: SpectraMix (stable-rank α=σ(20(conc−c₀)), NS-5, Nesterov-before-ortho, c₀ stage drift),
Referee (shared-params experts, greedy/Hedge-η8/AdaHedge, holdout election), Pilot (unit-RMS direction
election × DoG base × {0.0625,0.25,1,4} scale election, grad-norm z-guard, holdout=128).

## 4. Results
### 4.1 Valley: 0.019→1e-15, 0.021→1.2e43 (diverged); robustness SGD 11 / Momentum 37 / Adam 52 per 101
### 4.2 Digits quick: Adam 0.114/93.2% edges faithful-Muon 0.121/93.0% at 200 steps (Nesterov-correct Muon
starts slower); 400-step: SpectraMix 0.058/95.2%, Muon 0.076/94.2%, Adam 0.084/94.4%
### 4.3 Unit demos: Adam bias-correction magnitudes exact; AdamW 3-weight 0.889/0.377/0.023 → uniform 0.889;
NS-5 band-mapping verified; SpectraMix endpoints exact to 3e-13 over 20-step trajectories
### 4.4 Market (live): vision gaps compress to 50–54% (efficient-market coin-flip); fair-grid bests cluster
within seed noise (n=69) — the boundary condition for orthogonalization (§5.4)
### 4.5 Referee: Hedge-η8 0.053/95.6% (200-step), greedy 0.052/95.6% (400-step ×5); selection log shows
Adam 78% in Q1 → Muon-family ~100% after; momentum re-hired late under greedy (24% Q4); dropping it
hurts (0.065→0.087); AdaHedge parameter-free 0.072; cost ~1.3×/step
### 4.6 Pilot (0 learning rates): 200-step 0.046/94.2%, 400-step ×5 0.019/96.0% (seeds 0.005–0.039) —
best of everything on vision; markets neutral (49.3%, coin-flip). Three documented autopsies: DoG
cold-start ratchet (→24.5), EMA-distance freeze (stall 2.30), dimensional error (45× pinball); cures:
cold-start floor, scale election, ||g||/√N-correct DoG norm, 128-sample holdout (0.20→0.046).

## 5. Discussion
Mechanism → measurement → prediction per pattern (README §5). Strongest predictions: (a) optimizer rank
as a function of spectrum concentration × noise tail-weight (phase diagram, 3 domains); (b) election
schedules transfer across architectures (Adam-first appears in two independent mechanisms: Hedge weights
and DynMuon theory — coincidence or structure?); (c) TIOI-style stability targeting subsumes warmup.

## 6. Threats to validity
CPU NumPy MLP ≠ GPU transformer; k/schedule/holdout tuned on one vision task; market n=69 (noise ±3pp);
holdout streams from train pool (test untouched); no LR schedules/warmup ablations for baselines; single
market day per run (sources recorded in results).

## 7. Future work (PhD roadmap)
1. Phase diagram (Gini vs rank) across vision/language/finance. 2. Pilot at transformer scale (forwards get
relatively cheaper — economics improve with size). 3. Head-to-head vs ROR one-epoch cost and DynMuon/Muon^p.
4. Hedge regret accounting on the actual selection trace (close the theory loop). 5. MuonClip/QK-Clip
ablation below 1B params; cubic-vs-quintic NS frontier on MLP.

## 8. References — README §7 + §5.6–5.8 citations (Muon^p, DynMuon, Musec, ROR, AOS, OptiRoulette, AdaHedge,
FlipFlop, DoG/Prodigy/DAoG, ZClip, SA, R-metric, TIOI).

## 9. Artifact checklist — code+docker+results.json+figures, `experiments/run_all.py --full`,
`experiments/try_{spectramix,referee,pilot}.py`; Pages site with quiz (`docs/`, Settings → Pages → main → /docs).

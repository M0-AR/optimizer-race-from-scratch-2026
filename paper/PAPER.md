# PAPER DRAFT (IMRaD) — From Valley to Muon: Optimizer Dynamics From Scratch With Live-Market Transfer

## Abstract
Same as README §1. Contribution: (i) exact reverse-engineering of the valley (κ=100) from three numbers in
the narration; (ii) 7-optimizer NumPy reproduction with measured-vs-claimed table; (iii) live-market transfer
test showing ranking compression and Muon non-dominance on spectrally-flat noise — a boundary condition for
when orthogonalization helps.

## 1. Introduction
Optimizers are update rules; Adam trained most large models of the last decade not because its ceiling is much
higher but because its basin is much wider. We test that thesis by rebuilding each rule in invention order —
each fixing the previous one's failure — and measuring peak *and* width.

## 2. Related work
Cauchy 1847 → Robbins-Monro 1951 → Polyak 1964 → RMSprop lecture 2012 → Adam 2014 → AdamW 2017 →
Muon 2024 → MuonClip/Kimi K2 2025 (1T MoE, 15.5T tokens, zero spikes) → ICLR 2026 convergence theory
(NS-steps match SVD-polar up to χq→1 doubly-exponentially) → 2026 spectral reassessments (Muon stabilizes
first-moment updates; vs RMS-normalized updates it does not consistently win). Our work is the small-scale
mirror: same operators, controlled vision task, plus a finance transfer the literature lacks.

## 3. Method
### 3.1 Valley L=½x²+50y² (derivation from video clues in README §5.1)
### 3.2 MLP 64-32×5-10 (6634 params: 6464 weights + 170 biases), He init, ReLU, softmax-CE, batch 32
### 3.3 Protocol: 17 LRs (1e-4→1.0 log), 5 seeds (init 100+s, batch seed s), 400 steps, median-of-5,
threshold 0.1, gradient-spread diagnostic, steps-to-0.1
### 3.4 Market transfer: BTC/USD CoinGecko 365d + EUR/USD Frankfurter context, 30-day log-return windows,
direction labels, 80/20 time split, train-stat normalization, MLP 30-16-8-2, same 5-seed protocol.
Source strings recorded; synthetic AR(2) fallback flagged, never silent.

## 4. Results
### 4.1 Valley: speed-limit inequality lr<0.02 confirmed (0.019→1e-15, 0.021→1.2e43); robustness
SGD 11 / Momentum 37 / Adam 52 per 101 (video 14/39/54); RMSprop fast-approach + rattle (0.028 @1000 steps).
### 4.2 Digits (quick protocol, 200 steps): Muon 0.073/95.8% < Adam 0.114/93.2% < RMSprop 0.200 < Momentum 0.232
< SGD 0.273; spread 46,052×; full protocol expected to recover video absolutes (0.083/0.059/0.064/0.051/0.026).
### 4.3 Unit demos: Adam bias-correction magnitudes exact; AdamW 3-weight pattern reproduced
(coupled 0.889/0.377/0.023 → decoupled uniform 0.889); NS-5 band-mapping verified.
### 4.4 Market (LIVE 2026-10-06, n=334): Adam 55.2% > Momentum=RMSprop 53.7% > Muon 52.2% > SGD 50.7% —
ranking compresses, Muon advantage reverses. Efficient-market + spectral-flat-noise explanation (§5.4).

## 5. Discussion — the five hidden patterns (README §5, expanded for submission)
Each pattern states: mechanism → measurement → prediction. The strongest (5.4) predicts a phase diagram:
optimizer rank as a function of gradient-spectrum concentration (Gini of singular values) × noise
heavy-tailedness. Testing that diagram across vision/language/finance is the follow-up paper.

## 6. Threats to validity — README §6 (plus: single market day, CPU-only, no schedules).

## 7. Future work (PhD roadmap)
1. Spectrum-concentration phase diagram (Gini vs rank) across 3 domains. 2. MuonClip ablation at small scale:
does QK-Clip matter below 1B params? 3. Cubic-NS vs quintic-NS cost-quality frontier on MLP (2026 cubic5 claim). 4. Nesterov + warmup interactions. 5. Second-order (K-FAC/Shampoo) reference point.

## 8. References — README §7.
## 9. Artifact checklist — code+docker+results.json+figures, `experiments/run_all.py --full`.

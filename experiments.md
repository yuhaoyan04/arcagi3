# SWM (Spherical World Model) Experiment Log

## Summary

Goal: spherical representations (S^{d-1}) + non-SIGReg anti-collapse for world models (plan.md / plan_v2.md V0).

Base config: SphericalJEPA, ViT-Tiny, embed_dim=64, batch_size=128, T=4, Two-Room, lr=5e-5, spread weight=0.1.

### Uniformity_loss scale reference (t=2, N=512)

| State | uniformity_loss value | Meaning |
|-------|----------------------|---------|
| Full collapse (all z_i identical) | **6.24** = log(511) | All sq_dist=0, exp(0)=1 |
| Random 64D unit vectors | **≈2.2** = log(511·e⁻⁴) | avg sq_dist≈2 |
| Well-spread | **< 2.2** | avg sq_dist > 2 |

### Mean cosine spread_loss scale reference

| State | spread_loss value |
|-------|------------------|
| Full collapse | **1.0** (all cosine sim = 1) |
| Random 64D unit vectors | **≈0.0** |

### Sliced spread_loss scale reference (`D=64`)

| State | sliced_spread_loss value | Meaning |
|-------|--------------------------|---------|
| Full collapse | **≈0.03121** (`≈ 2 / D`) | Sorted projections all equal to one constant |
| Random 64D unit vectors | **≈8e-5** | Projection order statistics match target quantiles |
| Well-spread | **→ 0** | 1D marginals close to uniform-on-sphere target |

### Results overview

| # | Approach | Loss type | Init spread | E0 fit spread | E0 val spread | Collapse broken? | Failure mode |
|---|----------|-----------|-------------|--------------|---------------|-----------------|--------------|
| 1 | Linear + mean cosine | cosine | 1.0 | 1.0 | 1.0 | No | Gradient dead zone |
| 2 | + detach target | cosine | 1.0 | 1.0 | 1.0 | No | Gradient dead zone |
| 3 | InfoNCE (τ=0.1) | InfoNCE | 16.24 | 16.24→7.55(E8) | 7.57 | E1 violent break | Destroys temporal structure |
| 4 | MLP+BN projector | cosine | 1.0 | **0.006** | **1.0** | Train only | Batch masking |
| 5 | + LayerNorm pre-L2 | cosine | 1.0 | 1.0 | 1.0 | No | Gradient dead zone |
| 6 | + noise (σ=1e-2) | cosine | 1.0 | 1.0 | 1.0 | No | Gradient dead zone |
| 7 | variance_loss on emb_raw | variance | 0.996 | 1.0 | 1.0 | No | Gradient dead zone (0/0) |
| 8 | DINO centering + uniformity | uniformity | 6.24 | **3.94** | **6.24** | Train only | Batch masking |
| 9 | uniformity_loss only (no centering) | uniformity | 6.24 | 6.24 | 6.23 | No | Gradient dead zone |
| 10 | sliced Wasserstein on sphere | sliced | 0.03296 | — | 0.03126 (E99) | No | Weak signal after L2 normalisation |
| 11 | MLP+BN + uniformity | uniformity | 4.28 | 2.83 | 4.28 | Yes | Slow BN / running-stats alignment |

### Three failure modes

| Mode | Experiments | Mechanism |
|------|------------|-----------|
| **Gradient dead zone** | 1, 2, 5, 6, 7, 9 | At collapse all z_i identical → (z_i−z_j)=0, std=0/0 → zero gradient |
| **Batch-dependent masking** | 4, 8 | BN / centering creates diversity during training only → loss gets no corrective signal → eval reveals true collapse |
| **Temporal destruction** | 3 | InfoNCE pushes ALL pairs apart (including adjacent frames) → conflicts with pred_loss |
| **Weak post-norm signal** | 10 | Sorting gives gradient, but L2 normalisation already erased most cross-sample variation |
| **Slow but real escape** | 11 | BN creates useful perturbation; eval lags early, then running stats catch up and collapse is broken |

### Root cause

Random ViT CLS tokens = shared component (v, magnitude ~10) + input-dependent variation (ε_i, magnitude ~0.01). L2 normalisation maps v+ε_i to v/||v|| ≈ same unit vector for all i, killing the ε_i signal. This is unique to spherical architectures — LeWM in R^d preserves ε_i and SIGReg's sorting mechanism detects it.

### What works (SIGReg) and why

SIGReg succeeds because of **sorting + quantile matching** inside Epps-Pulley: project embeddings to random 1D direction → sort → compare sorted values against Gaussian characteristic function. Sorting assigns different ranks to identical values, providing non-zero gradient at exact collapse. No batch-dependence, no pairwise differences.

---

## Experiment Details

### Exp 1: V0 Baseline (Linear + mean cosine spread_loss)

**Config**: `nn.Linear(hidden_dim, embed_dim)` projector, spread_loss = mean pairwise cosine, weight=0.1.

| Stage | pred_loss | spread_loss | loss |
|-------|-----------|-------------|------|
| Init | 1.061 | 1.0 | 1.162 |
| E3 fit | 2.1e-5 | 1.0 | 0.100 |
| E3 val | 2.6e-5 | 1.0 | 0.100 |

Complete collapse. Predictor trivially outputs constant.

---

### Exp 2: + detach() on target

**Change**: `tgt_emb = emb[:, n_preds:].detach()`.

No improvement. Collapse is in the encoder, not gradient flow through target.

---

### Exp 3: InfoNCE spread_loss (τ=0.1)

**Change**: `logsumexp(sim/τ)`. InfoNCE scale: collapse=16.24, random=6.24.

| Stage | pred_loss | spread_loss |
|-------|-----------|-------------|
| Init | 1.080 | 16.236 (collapse) |
| E0 fit | 0.001 | 16.236 (still collapsed) |
| E1 fit | 0.966 | 9.286 (broke out) |
| E8 fit | 0.875 | 7.551 (near random) |
| E8 val | 0.879 | 7.573 |

Broke collapse at E1 via float noise accumulation, but pred_loss jumped to untrained level (0.97) and barely recovered. InfoNCE pushes temporal neighbours apart, directly opposing pred_loss.

---

### Exp 4: MLP projector with BatchNorm

**Change**: `MLP(hidden_dim, 2048, embed_dim, norm_fn=BatchNorm1d)`.

| Stage | pred_loss | spread_loss (cosine) |
|-------|-----------|---------------------|
| Init | 0.935 | 1.0 |
| E0 fit | 0.049 | **0.006** |
| E0 val | 0.017 | **1.0** |

Train/val split: BN decorrelates during training (batch stats) → spread sees no collapse → no corrective gradient. Eval with running stats → collapse exposed.

---

### Exp 5: LayerNorm before L2 normalize

| Stage | pred_loss | spread_loss (cosine) |
|-------|-----------|---------------------|
| Init | 1.104 | 1.0 |
| E0 fit | 4.5e-4 | 1.0 |

LayerNorm is per-sample. Doesn't fix cross-sample similarity.

---

### Exp 6: Noise injection (σ=1e-2)

| Stage | pred_loss | spread_loss (cosine) |
|-------|-----------|---------------------|
| Init | 0.880 | 1.0 |
| E0 fit | 6.0e-4 | 1.0 |

σ=1e-2 negligible vs embedding magnitude (~10). Angular perturbation ≈ 0.001 rad.

---

### Exp 7: variance_loss on pre-L2-norm embeddings

**Change**: `clamp(1 - std_per_dim, min=0).mean()` on emb_raw.

| Stage | pred_loss | spread_loss (variance) |
|-------|-----------|----------------------|
| Init | 1.138 | 0.996 |
| E0 fit | 7.9e-4 | 1.0 |

variance gradient = (x_i − mean)/(N·std) = 0/0 at collapse. PyTorch resolves to 0.

---

### Exp 8: DINO centering + uniformity_loss

**Change**: Subtract batch mean (train) / EMA center (eval) before L2 norm. uniformity_loss (t=2) on normalised embeddings.

| Stage | pred_loss | spread_loss (uniformity) |
|-------|-----------|--------------------------|
| Init | 1.053 | 6.236 (= collapse) |
| E0 fit | 0.986 | **3.935** (partially spread) |
| E0 val | 0.350 | **6.235** (= collapse) |

Same pattern as Exp 4: centering creates diversity during training (fit spread improved), but eval with EMA center still collapsed. Centering ≈ BN = batch-dependent operation that masks collapse from the loss.

---

### Exp 9: uniformity_loss only (no centering, Linear projector)

Isolate uniformity_loss without any architectural changes to confirm baseline behaviour.

| Stage | pred_loss | spread_loss (uniformity) |
|-------|-----------|--------------------------|
| Init | 0.932 | 6.236 (= collapse) |
| E0 fit | 6.2e-4 | 6.236 (= collapse) |
| E0 val | 8.3e-5 | 6.235 (= collapse) |

Complete collapse. uniformity_loss has zero gradient at collapse (same dead zone as other pairwise losses).

---

### Exp 10: sliced_spread_loss (sorting + quantile matching on sphere)

**Change**: replace pairwise/uniformity spread loss with `sliced_spread_loss()`:
- project all `B*T` unit vectors onto random directions
- sort projections independently per direction
- match sorted values against `N(0, 1/D)` quantiles

Training command:

```bash
python train_swm.py --config-name=swm.yaml \
    data=tworoom \
    subdir=ckpt/swm_v0_20260414_exp10_sliced_spread_loss \
    wandb.enabled=False
```

Config: `embed_dim=64`, `spread.weight=0.1`, `n_projections=256`.

| Stage | pred_loss | spread_loss (sliced) | loss |
|-------|-----------|----------------------|------|
| Init val sanity check | 0.9465 | 0.03296 | 0.94979 |
| E99 fit | 5.31e-6 | 0.03351 | 0.003356 |
| E99 val | 4.98e-6 | 0.03126 | 0.003131 |

Interpretation:
- `pred_loss` goes essentially to zero, so the predictor learns the trivial constant-latent solution perfectly.
- `spread_loss` stays almost exactly at the collapse baseline from start to finish.
- Unlike Exp 4 / 8, fit and val agree: this is **not** batch masking.
- Therefore sorting provides a non-zero gradient in principle, but in the current spherical pipeline that signal is too weak to move the model away from the collapsed basin.

Likely mechanism:
- The loss is applied **after** L2 normalisation on `S^{d-1}`.
- Earlier experiments already suggested raw ViT CLS features look like `v + ε`, where `||v|| >> ||ε||`.
- L2 normalisation maps `v + ε` close to `v / ||v||`, suppressing the useful variation before the spread loss sees it.
- `sliced_spread_loss` avoids the exact zero-gradient dead zone, but it still cannot overcome the trivial predictor solution once the representation has already been angularly flattened.

---

### Exp 11: MLP+BatchNorm projector + uniformity_loss

**Change**:
- use the original LeWM-style `MLP(..., BatchNorm1d)` projector and predictor projector
- keep spherical encoder / predictor outputs (`L2` normalised)
- use `uniformity_loss(t=2)` as spread loss

Training run:
- SwanLab run: `swm_v0_bn_uniform_lambda_0p1_t_2_20260415`
- URL: `https://swanlab.cn/@qunteam/worldmodels/runs/x3poay2amzei0vi6f1rlt/chart`

Config: `embed_dim=64`, `spread.weight=0.1`, `spread.type=uniformity`, `t=2.0`.

| Stage | pred_loss | spread_loss (uniformity) | loss |
|-------|-----------|--------------------------|------|
| Early fit (step 49) | 0.9712 | 2.8337 | 1.2545 |
| Early val (epoch 1) | 0.1814 | 4.2845 | 0.6098 |
| Mid fit peak spread (step 299) | 0.0247 | 6.0299 | 0.6277 |
| Final fit (step 128399) | 0.0060 | 2.4655 | 0.2526 |
| Final val (epoch 99) | 0.00220 | 2.4712 | 0.2493 |

Interpretation:
- This is **not** the Exp 4 / Exp 8 failure mode.
- Validation spread does **not** stay near the collapse baseline (`6.24`); it falls to `2.47`, close to the random-spread reference (`≈2.2`).
- Validation pred loss also goes very low (`0.181 -> 0.0022`), so the model is not merely decorrelated in training mode.
- Therefore BN + uniformity appears to be the first spherical variant that **really escapes collapse on validation**, not just on train.

Training dynamics:
- Early on, train and validation both fluctuate strongly.
- Fit spread initially rises toward the collapse scale (`≈6`), while validation spread is still high and noisy.
- After several epochs, validation rapidly improves and then stabilises around `2.47`.
- This suggests BN is doing more than masking: it injects enough cross-sample variation to break symmetry, but eval-mode running statistics need time to align before that benefit appears on validation.

Updated judgement on BatchNorm:
- Exp 4 was too pessimistic as a universal conclusion.
- BN still carries a train/eval mismatch risk.
- But at least in this long-run BN+uniformity setting, the mismatch is **transient**, not permanent masking.
- The most accurate description is: **slow-starting but genuinely non-collapsed**.

Remaining caveat:
- These metrics show that the representation escapes collapse.
- They do **not yet** prove better downstream planning / control performance.
- Exp 11 should therefore be treated as a promising positive result pending evaluation.

---

## Key Insight: The Gradient Dead Zone

At exact collapse (all z_i identical), gradient is zero for ALL tested losses:

| Loss type | Why zero at collapse |
|-----------|---------------------|
| Pairwise (cosine, uniformity, InfoNCE) | (z_i − z_j) = 0 |
| Statistical (variance, covariance) | (x_i − mean) = 0, std = 0 → 0/0 |
| Gram matrix on post-norm | Gradient ∝ (G−I)z₀, killed by L2 Jacobian (I−z₀z₀ᵀ)z₀ = 0 |

**Mechanisms now known to work**:
- sorting + quantile matching (used inside SIGReg's Epps-Pulley)
- BatchNorm-assisted uniformity, when trained long enough for eval running statistics to align

Sorting assigns different ranks to identical values → different target quantiles → non-zero gradient. Does not involve (z_i − z_j) or batch statistics.

### Updated interpretation after Exp 10

Exp 10 refines the conclusion:
- Sorting is likely **necessary** to escape exact collapse.
- But sorting **on post-L2 spherical embeddings is not sufficient**.
- The remaining bottleneck is probably the representation pipeline itself: useful variation is destroyed by early normalisation, so the spread objective receives too little signal too late.

This points to the next design direction:
- keep spherical prediction / planning if desired
- but apply anti-collapse regularisation on **pre-normalised** projector outputs, where the small non-collapsed variation still exists
- or otherwise preserve / amplify that variation before projecting to the sphere

### Updated again after Exp 11

Exp 11 further refines the picture:
- BatchNorm can be more than batch masking.
- When combined with uniformity and trained long enough, it can provide a real route out of collapse.
- The apparent contradiction with Exp 4 is explained by time scale: early validation can look worse before BN running statistics catch up.

So the current picture is:
- post-L2 linear projector + pairwise/statistical spread losses: fail
- post-L2 sliced spread alone: fail
- BN + uniformity: **promising success**
- SIGReg in Euclidean space: still the cleanest known robust baseline

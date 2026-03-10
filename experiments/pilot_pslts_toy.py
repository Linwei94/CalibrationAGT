"""
Pilot v3: PSLTS (Pseudo-Soft Label Temperature Scaling) on toy example.

Setting: only voted labels available; goal is to minimize ECE_true.

Method (PSLTS-v3, Confidence-Adaptive):
  β_i = 1 − f_i[y*]                              (model uncertainty about voted class)
  π̂_i = (1 − β_i) · e_{y*_i} + β_i / K          (instance-adaptive label smoothing)
       = f_i[y*] · e_{y*_i} + (1 − f_i[y*]) / K  (smooth to uniform by uncertainty)
  T*   = argmin_T  Σ_i KL(π̂_i ‖ softmax(z_i / T))

Key properties:
  π̂_i[y*] = f_i[y*] + (1−f_i[y*])/K  < 1.0    (since f_i[y*] < 1)
  E_{π̂_i}[z_i] = f_i[y*]·z_i[y*] + (1−f_i[y*])·mean(z_i) ≤ z_i[y*]
  ⟹  T*_PSLTS ≥ T*_TS          (provable: smaller stationarity target ⟹ larger T)
  ⟹  PSLTS softens predictions, moving them toward the annotator distribution.

Proof of T ordering:
  Stationarity: (1/n)Σ_i E_{π̂_i}[z_i] = (1/n)Σ_i E_{softmax(z_i/T*)}[z_i]
  LHS_PSLTS = (1/n)Σ_i [f_i[y*]·z_i[y*] + (1−f_i[y*])·mean(z_i)]
            ≤ (1/n)Σ_i z_i[y*]  =  LHS_TS      (since z_i[y*] ≥ mean(z_i))
  RHS is decreasing in T, so smaller LHS ⟹ larger T*.  QED.

Intuition:
  - Confident examples (f[y*] ≈ 1): β_i ≈ 0, π̂_i ≈ e_{y*}  (likely unambiguous → keep one-hot)
  - Uncertain examples (f[y*] ≈ 1/K): β_i ≈ (K-1)/K, π̂_i ≈ uniform  (likely ambiguous → smooth)

Baselines:
  TS              — standard (voted labels, one-hot target)
  LabelSmooth-TS  — global ε = mean(β_i), non-adaptive ablation
  PSLTS (ours)    — per-example β_i = 1 − f_i[y*]
  SLTS            — oracle (uses true annotator distributions)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'paper'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ── Exact same setup as make_figures.py ────────────────────────────────────
np.random.seed(11)
torch.manual_seed(11)

def generate_data(n_per_class=1500):
    """Identical to make_figures.py — uses global np.random state."""
    cov     = np.diag([1.0, 0.5])
    centers = [np.array([-3., 0.]), np.array([0., 0.]), np.array([3., 0.])]
    PI_AMB  = [0., 0.7, 0.3]
    X_list, yh_list, ys_list, amb_list = [], [], [], []
    for i, center in enumerate(centers):
        Xi = np.random.multivariate_normal(center, cov, n_per_class).astype(np.float32)
        if i == 0:
            yh = np.zeros(n_per_class, dtype=np.int64)
            ys = np.tile([1., 0., 0.], (n_per_class, 1)).astype(np.float32)
            am = np.zeros(n_per_class, dtype=bool)
        elif i == 1:
            yh = np.ones(n_per_class, dtype=np.int64)
            ys = np.tile(PI_AMB, (n_per_class, 1)).astype(np.float32)
            am = np.ones(n_per_class, dtype=bool)
        else:
            yh = np.full(n_per_class, 2, dtype=np.int64)
            ys = np.tile([0., 0., 1.], (n_per_class, 1)).astype(np.float32)
            am = np.zeros(n_per_class, dtype=bool)
        X_list.append(Xi); yh_list.append(yh)
        ys_list.append(ys); amb_list.append(am)
    return (np.vstack(X_list), np.concatenate(yh_list),
            np.vstack(ys_list).astype(np.float32), np.concatenate(amb_list))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )
    def forward(self, x): return self.net(x)

class TS(nn.Module):
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1)*1.5)
    def forward(self, z): return z / self.T.clamp(min=1e-3)
    def fit(self, z, yh):
        opt = torch.optim.LBFGS([self.T], lr=0.1, max_iter=500,
                                 tolerance_grad=1e-9, tolerance_change=1e-11)
        crit = nn.CrossEntropyLoss()
        def cl(): opt.zero_grad(); l = crit(self(z), yh); l.backward(); return l
        opt.step(cl); return self

class SoftTS(nn.Module):
    """Temperature scaling against any soft target π̂."""
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1)*1.5)
    def forward(self, z): return z / self.T.clamp(min=1e-3)
    def fit(self, z, pi):
        opt = torch.optim.LBFGS([self.T], lr=0.1, max_iter=500,
                                 tolerance_grad=1e-9, tolerance_change=1e-11)
        def cl():
            opt.zero_grad()
            l = -(pi * torch.log_softmax(self(z), 1)).sum(1).mean()
            l.backward(); return l
        opt.step(cl); return self

def make_pslts_targets(lg_cal, y_hard):
    """
    PSLTS-v3: confidence-adaptive pseudo-soft labels.
    β_i = 1 − f_i[y*]   (model's uncertainty about voted class)
    π̂_i = (1 − β_i) · e_{y*} + β_i / K
         = f_i[y*] · e_{y*} + (1 − f_i[y*]) / K
    Guarantees E_{π̂_i}[z_i] ≤ z_i[y*] ⟹ T*_PSLTS ≥ T*_TS.
    """
    K = lg_cal.shape[1]
    with torch.no_grad():
        f    = torch.softmax(lg_cal, 1)                          # (n, K)
        beta = 1 - f[torch.arange(len(y_hard)), y_hard]         # (n,) uncertainty
        yh   = torch.zeros(len(y_hard), K)
        yh.scatter_(1, y_hard.unsqueeze(1), 1.0)                 # one-hot
        pi_hat = (1 - beta).unsqueeze(1) * yh + beta.unsqueeze(1) / K
    return pi_hat, beta

def make_ls_targets(y_hard, K, eps):
    """Global label smoothing: π̂_i = (1-ε)·e_{y*} + ε/K."""
    yh = torch.zeros(len(y_hard), K)
    yh.scatter_(1, y_hard.unsqueeze(1), 1.0)
    return (1 - eps) * yh + eps / K

def ece_bins(probs, targets, n_bins=12, min_count=3):
    K = probs.shape[1]
    soft = np.eye(K)[targets.astype(int)] if targets.ndim == 1 else targets.astype(float)
    conf = probs.max(1); pred = probs.argmax(1)
    sacc = soft[np.arange(len(pred)), pred]
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        n = m.sum()
        if n >= min_count:
            ece += (n / len(probs)) * abs(float(conf[m].mean()) - float(sacc[m].mean()))
    return float(ece)

def ece_sampled(probs, soft_labels, n_bins=12, n_trials=100, seed=0):
    rng = np.random.default_rng(seed)
    K = soft_labels.shape[1]
    eces = []
    for _ in range(n_trials):
        sampled = np.array([
            rng.multinomial(1, np.clip(s, 0, None) / np.clip(s, 0, None).sum())
            for s in soft_labels
        ], dtype=float)
        eces.append(ece_bins(probs, sampled, n_bins=n_bins))
    return float(np.mean(eces))

def apply_cal(cal, lg_np):
    with torch.no_grad():
        return torch.softmax(cal(torch.tensor(lg_np)), 1).numpy()

# ── Experiment ────────────────────────────────────────────────────────────
print("=" * 65)
print("PSLTS-v3 Pilot: Toy Example (Confidence-Adaptive, Annotation-Free)")
print("=" * 65)

X, y_hard, y_soft, amb = generate_data(n_per_class=1500)
idx = np.arange(len(X))
idx_tr, idx_rest = train_test_split(idx, test_size=.4, random_state=0)
idx_cal, idx_te  = train_test_split(idx_rest, test_size=.5, random_state=0)

Xtr_t  = torch.FloatTensor(X[idx_tr]);   ytr_t  = torch.LongTensor(y_hard[idx_tr])
Xcl_t  = torch.FloatTensor(X[idx_cal]);  ycl_t  = torch.LongTensor(y_hard[idx_cal])
yscl_t = torch.FloatTensor(y_soft[idx_cal])
Xte_t  = torch.FloatTensor(X[idx_te])
ys_te  = y_soft[idx_te]
yh_te  = y_hard[idx_te]

# Train model (identical to make_figures.py)
model = MLP()
opt_m = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=128, shuffle=True)
for _ in range(250):
    model.train()
    for xb, yb in loader:
        opt_m.zero_grad(); nn.CrossEntropyLoss()(model(xb), yb).backward(); opt_m.step()
model.eval()
with torch.no_grad():
    lg_cal = model(Xcl_t)
    lg_te  = model(Xte_t)
lg_te_np = lg_te.numpy()

# Build pseudo-soft labels
K = 3
pi_pslts, beta_i = make_pslts_targets(lg_cal, ycl_t)
eps_global = float(beta_i.mean())               # global ε for LabelSmooth-TS
pi_ls      = make_ls_targets(ycl_t, K, eps_global)

print(f"\nβ_i = 1 − f[y*] statistics:")
print(f"  mean={beta_i.mean():.3f}  std={beta_i.std():.3f}  "
      f"min={beta_i.min():.3f}  max={beta_i.max():.3f}")
print(f"  global ε for LabelSmooth-TS = {eps_global:.3f}")

with torch.no_grad():
    pslts_y_conf = pi_pslts[torch.arange(len(ycl_t)), ycl_t].mean().item()
    slts_y_conf  = yscl_t[torch.arange(len(ycl_t)), ycl_t].mean().item()
print(f"\nMean target π̂[y*]:")
print(f"  TS:    1.000  (one-hot)")
print(f"  PSLTS: {pslts_y_conf:.3f}  = mean(f[y*] + (1−f[y*])/K)")
print(f"  SLTS:  {slts_y_conf:.3f}  (true annotator agreement)")
print(f"\n  Expected T ordering: T_TS < T_PSLTS ≤ T_SLTS")

# Fit all methods
ts_m    = TS().fit(lg_cal, ycl_t)
slts_m  = SoftTS().fit(lg_cal, yscl_t)
pslts_m = SoftTS().fit(lg_cal, pi_pslts)
ls_m    = SoftTS().fit(lg_cal, pi_ls)

T_ts    = float(ts_m.T)
T_slts  = float(slts_m.T)
T_pslts = float(pslts_m.T)
T_ls    = float(ls_m.T)

print(f"\nFitted temperatures:")
print(f"  T_TS={T_ts:.3f}  T_PSLTS={T_pslts:.3f}  T_SLTS={T_slts:.3f}  T_LS={T_ls:.3f}")
print(f"  T_TS < T_PSLTS? {T_ts < T_pslts}   T_PSLTS ≤ T_SLTS? {T_pslts <= T_slts}")

# Evaluate ECE
p_raw   = torch.softmax(lg_te, 1).numpy()
p_ts    = apply_cal(ts_m,    lg_te_np)
p_slts  = apply_cal(slts_m,  lg_te_np)
p_pslts = apply_cal(pslts_m, lg_te_np)
p_ls    = apply_cal(ls_m,    lg_te_np)

print(f"\n{'Method':<22} {'T':>6}  {'ECE-V':>8}  {'ECE-True':>10}  {'Δ':>8}")
print("-" * 62)
results = {}
yh1hot  = np.eye(K)[yh_te]
for name, p, T in [
    ("Uncalibrated",    p_raw,   None),
    ("TS",              p_ts,    T_ts),
    ("LabelSmooth-TS",  p_ls,    T_ls),
    ("PSLTS (ours)",    p_pslts, T_pslts),
    ("SLTS (oracle)",   p_slts,  T_slts),
]:
    ece_v = ece_bins(p, yh1hot) * 100
    ece_t = ece_sampled(p, ys_te) * 100
    delta = ece_t - ece_v
    T_str = f"{T:.3f}" if T else "   ---"
    print(f"{name:<22} {T_str:>6}  {ece_v:>7.2f}%  {ece_t:>9.2f}%  {delta:>+8.2f}")
    results[name] = dict(T=T, ece_v=ece_v, ece_t=ece_t, delta=delta)

print("\n" + "=" * 65)
print("KEY FINDINGS:")
ts_t    = results["TS"]["ece_t"]
pslts_t = results["PSLTS (ours)"]["ece_t"]
ls_t    = results["LabelSmooth-TS"]["ece_t"]
slts_t  = results["SLTS (oracle)"]["ece_t"]
print(f"  PSLTS vs TS:            {ts_t:.2f}% → {pslts_t:.2f}%  "
      f"({'✓ improvement' if pslts_t < ts_t else '✗ regression'}, {abs(ts_t-pslts_t):.2f}%)")
print(f"  PSLTS vs LabelSmooth:   {ls_t:.2f}% → {pslts_t:.2f}%  "
      f"({'✓ adaptive wins' if pslts_t < ls_t else '✗ adaptive loses'}, {abs(ls_t-pslts_t):.2f}%)")
print(f"  PSLTS gap from SLTS:    {pslts_t:.2f}% vs {slts_t:.2f}% (gap: {pslts_t-slts_t:.2f}%)")
print("=" * 65)

# Save
import json
os.makedirs("results", exist_ok=True)
out = {k: {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
       for k, v in results.items()}
out["_meta"] = {
    "beta_mean": float(beta_i.mean()), "beta_std": float(beta_i.std()),
    "eps_global": eps_global,
    "T_TS": T_ts, "T_PSLTS": T_pslts, "T_SLTS": T_slts,
    "pslts_target_ystar": pslts_y_conf, "slts_target_ystar": slts_y_conf,
}
json.dump(out, open("results/pilot_pslts_toy.json", "w"), indent=2)
print(f"Saved → results/pilot_pslts_toy.json")

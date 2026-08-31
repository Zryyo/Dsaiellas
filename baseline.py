"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, math, statistics, time
from dataclasses import replace, asdict
import numpy as np
from data import load, encode, FIELDS, FEATURE_FNS
from evaluate import evaluate
from losses import get_loss, LOSSES
import config

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y, uids=None, loss_fn=None):
        """loss_fn: (z, y, uids) -> (grad_z, loss_value), see losses.py.
        uids lets a loss group the batch by user when it needs to; the
        backprop through V/W below only depends on grad_z, so it's the
        same regardless of which loss produced it."""
        loss_fn = loss_fn or get_loss('logloss')
        z, E, S = self.logits(X)
        g, loss_val = loss_fn(z, y, uids)
        g = g.astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return loss_val

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def train(enc, dim, cfg=None, seed=0, verbose=True):
    """Shared FM training loop: early-stops on valid primary and returns the
    fitted model. This is the single place baseline.run_fm, ablation_features.py,
    and submit.py --make all train from — a change here (loss, features via
    the enc passed in, hparams) reaches all three instead of only one."""
    cfg = cfg or config.DEFAULT
    Xtr, ytr, utr = enc['train']
    utr = np.asarray(utr)
    Xva, yva, uva = enc['valid']
    m = FM(dim, k=cfg.k, lr=cfg.lr, seed=seed)
    loss_fn = get_loss(cfg.loss)
    rng = np.random.default_rng(seed)

    if cfg.sampler == 'user':
        user_rows = {}
        for i, u in enumerate(utr):
            user_rows.setdefault(u, []).append(i)
        user_rows = {u: np.asarray(rows, dtype=np.int64) for u, rows in user_rows.items()}
        user_list = list(user_rows.keys())

    def make_batches():
        """Row indices for one epoch's batches. 'row' (default): iid shuffle
        of all rows — unchanged from before the sampler seam existed, so it
        reproduces the exact same training trajectory bit-for-bit. 'user':
        shuffle user order, then fill batches by appending whole users'
        rows contiguously, so a user is never split across two batches —
        needed for any loss that compares examples within the same user
        (row-level iid shuffling on ~1.1M rows / ~27k users means an 8192
        row batch has almost no within-user repeats to compare)."""
        if cfg.sampler == 'user':
            rng.shuffle(user_list)
            batches, buf, buf_len = [], [], 0
            for u in user_list:
                rows = user_rows[u]
                buf.append(rows); buf_len += len(rows)
                if buf_len >= cfg.bs:
                    batches.append(np.concatenate(buf)); buf, buf_len = [], 0
            if buf:
                batches.append(np.concatenate(buf))
            return batches
        idx = rng.permutation(len(ytr))
        return [idx[i:i + cfg.bs] for i in range(0, len(idx), cfg.bs)]

    best, best_state, bad = -1, None, 0
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        batches = make_batches()
        losses = [m.step(Xtr[b], ytr[b], utr[b], loss_fn) for b in batches]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m

def run_fm(enc, dim, seed=0, verbose=True, **kw):
    """enc/dim come from data.encode(splits) — hoisted out so callers that need
    several runs (e.g. run_multiseed) only encode the data once.
    kw overrides fields of config.DEFAULT (k, lr, epochs, bs, patience, loss)."""
    cfg = replace(config.DEFAULT, **kw)
    Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = train(enc, dim, cfg=cfg, seed=seed, verbose=verbose)
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}

METRICS = ('GAUC', 'nDCG@5', 'primary')

def run_multiseed(splits, seeds=(0, 1, 2), **kw):
    """Run run_fm once per seed (FM's std across seeds is ~0.0008, so a single
    seed can't be trusted for deltas) and summarize with mean/pstdev per metric."""
    cfg = replace(config.DEFAULT, **kw)
    enc, dim = encode(splits, feature_fn=cfg.feature_fn)
    runs = [run_fm(enc, dim, seed=seed, verbose=False, **kw) for seed in seeds]
    out = {}
    for sp in ('valid', 'test'):
        out[sp] = {}
        for metric in METRICS:
            vals = [r[sp][metric] for r in runs]
            out[sp][metric] = {'mean': statistics.mean(vals), 'std': statistics.pstdev(vals)}
    return out

def is_real(new_mean, new_std, base_mean, base_std, n_seeds=3):
    """True if new_mean's improvement over base_mean clears noise: a 2-sigma
    band on the difference of means, floored at 0.001 absolute."""
    threshold = max(2 * (new_std + base_std) / math.sqrt(n_seeds), 0.001)
    return (new_mean - base_mean) > threshold

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=config.DEFAULT.k)
    ap.add_argument('--lr', type=float, default=config.DEFAULT.lr)
    ap.add_argument('--epochs', type=int, default=config.DEFAULT.epochs)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--loss', default=config.DEFAULT.loss, choices=list(LOSSES))
    ap.add_argument('--feature_fn', default=config.DEFAULT.feature_fn, choices=list(FEATURE_FNS))
    ap.add_argument('--sampler', default=config.DEFAULT.sampler, choices=['row', 'user'])
    ap.add_argument('--multiseed', action='store_true',
                    help='average FM over seeds 0,1,2 instead of a single seed run')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    cfg = replace(config.DEFAULT, k=a.k, lr=a.lr, epochs=a.epochs,
                  loss=a.loss, feature_fn=a.feature_fn, sampler=a.sampler)

    if a.multiseed:
        if a.model != 'fm':
            raise SystemExit('--multiseed is only supported for --model fm')
        seeds = (0, 1, 2)
        res = run_multiseed(splits, seeds=seeds, **asdict(cfg))
        print(f"\n=== fm multiseed (seeds={seeds}) ===")
        for sp in ('valid', 'test'):
            r = res[sp]
            print(f"  {sp:5s}  " + " | ".join(
                f"{m} {r[m]['mean']:.4f}±{r[m]['std']:.4f}" for m in METRICS))
    else:
        if a.model == 'fm':
            enc, dim = encode(splits, feature_fn=cfg.feature_fn)
            res = run_fm(enc, dim, seed=a.seed, **asdict(cfg))
        else:
            res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed)}[a.model](splits)
        print(f"\n=== {a.model} (seed={a.seed}) ===")
        for sp in ('valid', 'test'):
            r = res[sp]
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")

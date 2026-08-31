"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def build_features_base(x, ctx):
    """Default feature function — the 5 domains in FIELDS.
    x   : raw row tuple (date, user_id, video_id, author_id, tab, duration_ms, label)
    ctx : context precomputed once per encode() call, e.g. {'dur_edges': ...}
    Returns a list of raw (pre-vocab) feature values, one per domain.

    To change feature construction, write a new function with this same
    (x, ctx) -> list signature and register it in FEATURE_FNS — it can
    define an entirely different set of domains, not just append to this
    one, since encode() sizes everything off the returned list's length.
    """
    edges = ctx['dur_edges']
    return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

FEATURE_FNS = {'base': build_features_base}

def encode(splits, feature_fn='base'):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    feature_fn: FEATURE_FNS 里的名字，或直接传 (x, ctx) -> list 的 callable。
    返回 (X, y, users) per split，X 为 int32 (N, n_fields)，以及 field_dims 之和。"""
    fn = FEATURE_FNS[feature_fn] if isinstance(feature_fn, str) else feature_fn
    tr = splits['train']
    ctx = {'dur_edges': _bucket_edges([x[5] for x in tr])}
    n_fields = len(fn(tr[0], ctx))

    vocabs = [dict() for _ in range(n_fields)]
    for x in tr:
        for i, v in enumerate(fn(x, ctx)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), n_fields), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(fn(x, ctx)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))

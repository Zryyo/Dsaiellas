"""Swappable loss functions for FM training.

Each loss is a function (z, y, uids) -> (grad_z, loss_value):
    z          : (B,) raw FM logits for the batch
    y          : (B,) binary labels
    uids       : (B,) user id per example — a loss that scores each example
                 independently can ignore it; a loss that needs to compare
                 or combine examples belonging to the same user can group
                 the batch by it.
    grad_z     : (B,) dL/dz per example. FM.step backprops this through
                 V/W identically regardless of which loss produced it, so
                 a new loss can be added here without touching the FM class.
    loss_value : scalar, for logging only.

Select a loss by name via config.Config.loss / losses.get_loss(name).
"""
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def logloss(z, y, uids):
    """Pointwise binary cross-entropy (the original FM baseline loss)."""
    p = sigmoid(z)
    grad = ((p - y) / len(y)).astype(np.float32)
    loss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    return grad, loss

def pairwise_bpr_within_user(z, y, uids):
    """Within-user pairwise logistic (BPR-style) loss.

    For each user u, form all (pos, neg) pairs from that user's examples and
    minimize mean softplus(-(z_pos - z_neg)). This directly optimizes a smooth
    surrogate of within-user AUC. Gradients are averaged over all pairs in the
    batch so the step size remains comparable across batches.

    If the batch contains no user with both classes, falls back to pointwise
    logloss to keep training stable. Needs sampler='user' (see config.py) to
    actually see multiple rows per user in a batch — under the default
    sampler='row' it degenerates to logloss almost every batch.

    Agent-discovered and multi-seed validated: valid primary 0.6028 +/- 0.0002
    vs the 0.6014 +/- 0.0003 logloss/row baseline (+0.0014, real per is_real()),
    confirmed at 5 seeds. See runs/agent-openai-v3/iterations.jsonl iteration 2.
    """
    # If uids are not provided, revert to pointwise.
    if uids is None:
        return logloss(z, y, uids)

    B = len(z)
    grad = np.zeros(B, dtype=np.float32)
    loss_sum = 0.0
    total_pairs = 0

    # Group indices by user
    byu = {}
    for i, u in enumerate(uids):
        byu.setdefault(u, []).append(i)

    for _, idx_list in byu.items():
        if len(idx_list) < 2:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        labs = (y[idx] > 0.5).astype(np.int8)
        pos_rel = np.where(labs == 1)[0]
        neg_rel = np.where(labs == 0)[0]
        if len(pos_rel) == 0 or len(neg_rel) == 0:
            continue
        pos_idx = idx[pos_rel]
        neg_idx = idx[neg_rel]
        zp = z[pos_idx].astype(np.float32)
        zn = z[neg_idx].astype(np.float32)
        # Pairwise differences (P x N)
        # s = sigmoid(zn - zp) = sigmoid(-(zp - zn))
        s = sigmoid(zn[None, :] - zp[:, None]).astype(np.float32)
        # Loss: mean softplus(-(zp - zn)) = mean log(1 + exp(zn - zp))
        # Use logaddexp for numerical stability, in float64, then back to float.
        loss_sum += float(np.logaddexp(0.0, (zn[None, :] - zp[:, None]).astype(np.float64)).sum())
        pairs = s.size
        total_pairs += pairs
        # Gradients: d/dzp = -sum_j s_ij ; d/dzn = +sum_i s_ij
        gp = -s.sum(axis=1).astype(np.float32)
        gn = +s.sum(axis=0).astype(np.float32)
        grad[pos_idx] += gp
        grad[neg_idx] += gn

    if total_pairs == 0:
        # No informative pairs in this batch — fall back to pointwise BCE.
        return logloss(z, y, uids)

    grad /= float(total_pairs)
    loss = loss_sum / float(total_pairs)
    return grad, float(loss)

LOSSES = {
    'logloss': logloss,
    'pairwise': pairwise_bpr_within_user,
    # A loss that needs to compare or combine examples within the same
    # user can group this batch by uids and accumulate its gradient onto
    # each affected example's grad_z entry. It only needs to be added
    # here — FM.step is unchanged regardless of how grad_z was produced.
}

def get_loss(name):
    return LOSSES[name]

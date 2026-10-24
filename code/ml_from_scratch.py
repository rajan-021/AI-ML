"""From-scratch implementations for ML coding interviews.

Every function here is shown on the site's "ML Coding from Scratch" page (the page
includes the sections marked with --8<-- comments, so the site always shows this
tested code). Run the self-tests with:

    python code/ml_from_scratch.py

Only NumPy is required.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


# --8<-- [start:linear_regression]
def linear_regression_normal_equation(X, y):
    """Closed form: w = (X^T X)^-1 X^T y. A bias column is added. Returns (w, b)."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    theta = np.linalg.lstsq(Xb, y, rcond=None)[0]   # more stable than an explicit inverse
    return theta[:-1], theta[-1]


def linear_regression_gd(X, y, lr=0.1, epochs=1000):
    """Batch gradient descent on mean squared error. X: (n, d), y: (n,)."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        error = X @ w + b - y              # (n,)
        w -= lr * (2 / n) * (X.T @ error)  # dMSE/dw
        b -= lr * (2 / n) * error.sum()    # dMSE/db
    return w, b
# --8<-- [end:linear_regression]


# --8<-- [start:logistic_regression]
def sigmoid(z):
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def logistic_regression(X, y, lr=0.1, epochs=2000, l2=0.0):
    """Gradient descent on log loss with optional L2. Labels y in {0, 1}."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        w -= lr * (X.T @ (p - y) / n + l2 * w)   # gradient is (p - y) x
        b -= lr * np.mean(p - y)
    return w, b
# --8<-- [end:logistic_regression]


# --8<-- [start:softmax]
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)      # subtract the max: no overflow
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_with_grad(logits, labels):
    """logits: (n, k), labels: (n,) integer classes. Returns (loss, dL/dlogits)."""
    n = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))  # log-softmax
    loss = -log_probs[np.arange(n), labels].mean()
    grad = np.exp(log_probs)
    grad[np.arange(n), labels] -= 1               # p - y
    return loss, grad / n
# --8<-- [end:softmax]


# --8<-- [start:kmeans]
def kmeans(X, k, iters=100, seed=0):
    """Lloyd's algorithm with k-means++ initialization. Returns (centroids, labels)."""
    rng = np.random.default_rng(seed)
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):                          # k-means++: spread the starting points
        d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centroids], axis=0)
        centroids.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    centroids = np.array(centroids)
    for _ in range(iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)  # (n, k)
        labels = dists.argmin(axis=1)
        new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                        for j in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    return centroids, labels
# --8<-- [end:kmeans]


# --8<-- [start:knn]
def knn_predict(X_train, y_train, X_query, k=5):
    """Majority vote among the k nearest training points (Euclidean)."""
    d2 = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)  # (m, n)
    nearest = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    return np.array([Counter(y_train[row]).most_common(1)[0][0] for row in nearest])
# --8<-- [end:knn]


# --8<-- [start:tree_split]
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)


def best_split(X, y):
    """Return (feature, threshold, gain) of the best Gini split, or None if no split helps."""
    n, d = X.shape
    parent = gini(y)
    best = None
    for j in range(d):
        values = np.unique(X[:, j])
        for t in (values[:-1] + values[1:]) / 2:  # midpoints between distinct values
            left = X[:, j] <= t
            n_left = left.sum()
            child = (n_left * gini(y[left]) + (n - n_left) * gini(y[~left])) / n
            gain = parent - child
            if best is None or gain > best[2]:
                best = (j, t, gain)
    return best if best and best[2] > 0 else None
# --8<-- [end:tree_split]


# --8<-- [start:mlp]
def train_mlp(X, y, hidden=16, k=2, lr=0.5, epochs=2000, seed=0):
    """Two-layer network (ReLU hidden layer, softmax output) trained with full-batch GD."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2 / d), (d, hidden)); b1 = np.zeros(hidden)   # He init
    W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, k)); b2 = np.zeros(k)
    for _ in range(epochs):
        Z1 = X @ W1 + b1                 # (n, h)
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ W2 + b2                # (n, k)
        loss, dZ2 = cross_entropy_with_grad(Z2, y)
        dW2 = A1.T @ dZ2; db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)             # ReLU derivative
        dW1 = X.T @ dZ1; db1 = dZ1.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    predict = lambda Xn: (np.maximum(0, Xn @ W1 + b1) @ W2 + b2).argmax(axis=1)  # noqa: E731
    return predict, loss
# --8<-- [end:mlp]


# --8<-- [start:attention]
def scaled_dot_product_attention(Q, K, V, causal=False):
    """Q: (..., n, d_k), K: (..., m, d_k), V: (..., m, d_v) -> (..., n, d_v)."""
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if causal:
        n, m = scores.shape[-2:]
        scores = np.where(np.triu(np.ones((n, m), dtype=bool), k=1), -1e9, scores)
    return softmax(scores, axis=-1) @ V


def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads, causal=True):
    """X: (n, d_model); each W: (d_model, d_model)."""
    n, d_model = X.shape
    d_head = d_model // num_heads
    split = lambda T: T.reshape(n, num_heads, d_head).transpose(1, 0, 2)  # noqa: E731
    Q, K, V = split(X @ W_q), split(X @ W_k), split(X @ W_v)
    heads = scaled_dot_product_attention(Q, K, V, causal)            # (h, n, d_head)
    return heads.transpose(1, 0, 2).reshape(n, d_model) @ W_o
# --8<-- [end:attention]


# --8<-- [start:sampling]
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """Temperature, then top-k, then top-p (nucleus) filtering, then sample one token id."""
    rng = rng or np.random.default_rng()
    if temperature == 0:
        return int(np.argmax(logits))                 # greedy
    probs = softmax(np.asarray(logits, dtype=float) / temperature)
    order = np.argsort(probs)[::-1]                    # most likely first
    keep = np.ones(len(probs), dtype=bool)
    if top_k is not None:
        keep[order[top_k:]] = False
    if top_p is not None:
        cumulative = np.cumsum(probs[order])
        cutoff = np.searchsorted(cumulative, top_p) + 1   # smallest set reaching top_p
        keep[order[cutoff:]] = False
    probs = np.where(keep, probs, 0)
    return int(rng.choice(len(probs), p=probs / probs.sum()))
# --8<-- [end:sampling]


# --8<-- [start:bpe]
def train_bpe(word_counts, num_merges):
    """word_counts: {"low": 5, ...}. Returns the list of merges, most frequent pair first."""
    vocab = {tuple(word): count for word, count in word_counts.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, count in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += count
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))   # ties broken deterministically
        merges.append(best)
        new_vocab = {}
        for symbols, count in vocab.items():
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1]); i += 2
                else:
                    out.append(symbols[i]); i += 1
            new_vocab[tuple(out)] = count
        vocab = new_vocab
    return merges
# --8<-- [end:bpe]


# --8<-- [start:metrics]
def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def roc_auc(y_true, scores):
    """Probability a random positive outranks a random negative (ties count half)."""
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return (greater + 0.5 * ties) / (len(pos) * len(neg))


def ndcg_at_k(relevances, k):
    """relevances: graded relevance of results in ranked order."""
    rel = np.asarray(relevances, dtype=float)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(rel[:k] * discounts[: len(rel[:k])])
    ideal = np.sort(rel)[::-1][:k]
    idcg = np.sum(ideal * discounts[: len(ideal)])
    return dcg / idcg if idcg > 0 else 0.0
# --8<-- [end:metrics]


# --8<-- [start:retrieval]
def top_k_cosine(query, doc_matrix, k=3):
    """query: (d,), doc_matrix: (n, d). Returns indices and scores of the k most similar docs."""
    q = query / np.linalg.norm(query)
    D = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    scores = D @ q
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]
# --8<-- [end:retrieval]


# ----------------------------------------------------------------- self-tests

def _tests():
    rng = np.random.default_rng(0)

    X = rng.normal(size=(200, 3)); true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + 3 + rng.normal(0, 0.01, 200)
    w, b = linear_regression_normal_equation(X, y)
    assert np.allclose(w, true_w, atol=0.01) and abs(b - 3) < 0.01
    w2, b2 = linear_regression_gd(X, y, lr=0.1, epochs=2000)
    assert np.allclose(w2, true_w, atol=0.01) and abs(b2 - 3) < 0.01

    Xc = rng.normal(size=(400, 2)); yc = (Xc[:, 0] + 2 * Xc[:, 1] > 0).astype(int)
    w, b = logistic_regression(Xc, yc, lr=0.5, epochs=3000)
    acc = np.mean((sigmoid(Xc @ w + b) > 0.5) == yc)
    assert acc > 0.97, acc
    assert log_loss(yc, sigmoid(Xc @ w + b)) < 0.2

    logits = rng.normal(size=(5, 4)); labels = rng.integers(0, 4, 5)
    loss, grad = cross_entropy_with_grad(logits, labels)
    eps = 1e-6; num = np.zeros_like(logits)
    for i in range(5):
        for j in range(4):
            plus, minus = logits.copy(), logits.copy()
            plus[i, j] += eps; minus[i, j] -= eps
            num[i, j] = (cross_entropy_with_grad(plus, labels)[0]
                         - cross_entropy_with_grad(minus, labels)[0]) / (2 * eps)
    assert np.allclose(grad, num, atol=1e-6)            # analytic gradient matches numeric
    assert np.allclose(softmax(np.array([1000.0, 1000.0])), [0.5, 0.5])

    blobs = np.vstack([rng.normal(c, 0.2, (50, 2)) for c in ([0, 0], [5, 5], [0, 5])])
    cents, labs = kmeans(blobs, 3)
    assert sorted(np.round(cents).astype(int).tolist()) == [[0, 0], [0, 5], [5, 5]]

    preds = knn_predict(blobs, np.repeat([0, 1, 2], 50), np.array([[5.1, 4.9], [0.1, 5.2]]), k=5)
    assert preds.tolist() == [1, 2]

    Xs = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]]); ys = np.array([0, 0, 0, 1, 1, 1])
    feat, thr, gain = best_split(Xs, ys)
    assert feat == 0 and 3 < thr < 10 and abs(gain - 0.5) < 1e-9

    Xm = rng.normal(size=(300, 2)); ym = ((Xm[:, 0] * Xm[:, 1]) > 0).astype(int)  # XOR-like
    predict, final_loss = train_mlp(Xm, ym, hidden=32, lr=0.5, epochs=3000)
    assert np.mean(predict(Xm) == ym) > 0.95

    n, dm, h = 4, 8, 2
    Xa = rng.normal(size=(n, dm)); Ws = [rng.normal(size=(dm, dm)) for _ in range(4)]
    out = multi_head_attention(Xa, *Ws, num_heads=h, causal=True)
    assert out.shape == (n, dm)
    out2 = multi_head_attention(np.vstack([Xa[:2], rng.normal(size=(2, dm))]), *Ws, num_heads=h)
    assert np.allclose(out[:2], out2[:2])                # causal: later tokens can't change earlier outputs

    logits = np.array([2.0, 1.0, 0.5, -1.0])
    assert sample_next_token(logits, temperature=0) == 0
    draws = {sample_next_token(logits, top_k=2, rng=rng) for _ in range(200)}
    assert draws <= {0, 1}
    draws = {sample_next_token(logits, top_p=0.5, rng=rng) for _ in range(200)}
    assert draws == {0}

    merges = train_bpe({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 4)
    assert merges[:2] == [("s", "t"), ("e", "st")] or merges[:2] == [("e", "s"), ("es", "t")], merges

    yt = np.array([1, 0, 1, 1, 0]); yp = np.array([1, 0, 0, 1, 1])
    p, r, f = precision_recall_f1(yt, yp)
    assert abs(p - 2 / 3) < 1e-9 and abs(r - 2 / 3) < 1e-9 and abs(f - 2 / 3) < 1e-9
    assert roc_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.5, 0.1])) == 0.75
    assert abs(ndcg_at_k([3, 2, 0, 1], 4) - 0.9854) < 1e-3

    docs = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    idx, _ = top_k_cosine(np.array([1.0, 0.1]), docs, k=2)
    assert idx.tolist() == [0, 1]

    print("All self-tests passed.")


if __name__ == "__main__":
    _tests()

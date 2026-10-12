"""Summary-statistic Duncan test retained for FigS3 reproduction.

The historical equal-n assumption is n=3; this does not establish whether
replicates represent independent fermentation batches or technical repeats.
"""
import numpy as np
from scipy.stats import studentized_range

def duncan_letters(means, sds, n=3, alpha=0.05):
    means = np.asarray(means, float); sds = np.asarray(sds, float)
    k = len(means)
    mse = np.mean(sds ** 2)            # equal n: pooled within-group variance
    df = k * (n - 1)
    se = np.sqrt(mse / n)
    order = np.argsort(-means)          # descending
    m = means[order]
    def crit(p):
        ap = 1 - (1 - alpha) ** (p - 1)
        return studentized_range.ppf(1 - ap, p, df) * se
    # homogeneous[i][j]: range i..j (in sorted order) not significant (with step-down protection)
    hom = np.zeros((k, k), bool)
    for span in range(k, 0, -1):
        for i in range(0, k - span + 1):
            j = i + span - 1
            if span == 1:
                hom[i, j] = True; continue
            # protected if contained in a homogeneous larger range
            contained = any(hom[a, b] for a in range(0, i + 1) for b in range(j, k) if (b - a + 1) > span)
            hom[i, j] = contained or (m[i] - m[j]) <= crit(span)
    # maximal homogeneous subsets
    subsets = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if hom[i, jj])
        if not any(s[0] <= i and j <= s[1] for s in subsets):
            subsets.append((i, j))
    letters_sorted = [""] * k
    for li, (i, j) in enumerate(subsets):
        for t in range(i, j + 1):
            letters_sorted[t] += "abcdefghijklmnopqrstuvwxyz"[li]
    out = [""] * k
    for pos, idx in enumerate(order):
        out[idx] = letters_sorted[pos]
    return out, mse, df

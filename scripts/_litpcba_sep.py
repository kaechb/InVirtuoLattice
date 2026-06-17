"""Head-free per-target linear separability of LIT-PCBA actives vs decoys in a
given adapter's z_m space. 5-fold CV AUROC with a centroid (mean-difference)
direction and a ridge-LDA direction. Averaged over targets.

Usage: python scripts/_litpcba_sep.py <zm_cache_dir> <label>
"""
import json
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
np.random.seed(0)

CACHE = sys.argv[1]
LABEL = sys.argv[2]
TEST_PARQUET = "artifacts/processed/bindingdb/test_lit_pcba.parquet"
DECOY_CAP = 4000   # per target, to bound cost


def inchikey(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m) if m is not None else None


def load_cache(cache):
    m = json.load(open(f"{cache}/manifest.json"))
    cnt, dim, dt = m["count"], m["embedding_dim"], m["dtype"]
    mean = np.memmap(f"{cache}/mean.dat", dtype=dt, mode="r", shape=(cnt, dim))
    ik2row = {}
    with open(f"{cache}/pids.tsv") as fh:
        for line in fh:
            r, ik = line.rstrip("\n").split("\t")
            ik2row[ik] = int(r)
    return mean, ik2row


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    npos = labels.sum()
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def cv_auroc(A, D, k=5):
    """Pooled k-fold CV AUROC for centroid + ridge-LDA directions."""
    rng = np.random.default_rng(0)
    A = A[rng.permutation(len(A))]
    D = D[rng.permutation(len(D))]
    a_folds = np.array_split(np.arange(len(A)), k)
    d_folds = np.array_split(np.arange(len(D)), k)
    sc_c, sc_l, lab = [], [], []
    for i in range(k):
        a_te, d_te = a_folds[i], d_folds[i]
        a_tr = np.concatenate([a_folds[j] for j in range(k) if j != i])
        d_tr = np.concatenate([d_folds[j] for j in range(k) if j != i])
        muA, muD = A[a_tr].mean(0), D[d_tr].mean(0)
        diff = muA - muD
        Xtr = np.vstack([A[a_tr], D[d_tr]])
        C = np.cov(Xtr, rowvar=False) + 1e-2 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(C, diff)
        Xte = np.vstack([A[a_te], D[d_te]])
        yte = np.r_[np.ones(len(a_te)), np.zeros(len(d_te))]
        sc_c.append(Xte @ diff)
        sc_l.append(Xte @ w)
        lab.append(yte)
    sc_c, sc_l, lab = np.concatenate(sc_c), np.concatenate(sc_l), np.concatenate(lab)
    return auroc(sc_c, lab), auroc(sc_l, lab)


def main():
    mean, ik2row = load_cache(CACHE)
    df = pd.read_parquet(TEST_PARQUET, columns=["target_name", "smiles", "is_active"])
    df["target_name"] = df["target_name"].astype(str)
    df["is_active"] = df["is_active"].astype(int)

    rows = []
    for t, sub in df.groupby("target_name"):
        act = sub[sub.is_active == 1]
        dec = sub[sub.is_active == 0]
        if len(dec) > DECOY_CAP:
            dec = dec.sample(DECOY_CAP, random_state=0)
        keep = pd.concat([act, dec])
        keep = keep.assign(ik=[inchikey(s) for s in keep["smiles"]])
        keep = keep.dropna(subset=["ik"])
        keep = keep[keep["ik"].isin(ik2row)]
        a = keep[keep.is_active == 1]["ik"].map(ik2row).to_numpy()
        d = keep[keep.is_active == 0]["ik"].map(ik2row).to_numpy()
        if len(a) < 10 or len(d) < 10:
            print(f"  {t:8s} skip (a={len(a)} d={len(d)})")
            continue
        A = np.asarray(mean[np.sort(a)], dtype=np.float32)
        D = np.asarray(mean[np.sort(d)], dtype=np.float32)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
        ac, al = cv_auroc(A, D)
        rows.append((t, len(a), len(d), ac, al))
        print(f"  {t:8s} a={len(a):4d} d={len(d):5d} centroidAUROC={ac:.3f} ridgeLDA_AUROC={al:.3f}")

    cen = np.nanmean([r[3] for r in rows])
    lda = np.nanmean([r[4] for r in rows])
    print(f"\n[{LABEL}] targets={len(rows)}  mean centroid AUROC={cen:.4f}  mean ridgeLDA AUROC={lda:.4f}")


if __name__ == "__main__":
    main()

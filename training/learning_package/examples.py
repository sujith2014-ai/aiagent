"""LearningPackage from labelled examples supplied by the environment/user ("teach by example"): no teacher, no rule.
Exact duplicate feature vectors are removed first (they would leak between splits), then a stratified 60/20/20 split is made."""
from __future__ import annotations
import time
import numpy as np
from sklearn.model_selection import train_test_split
from training.learning_package.schema import LearningPackage


def package_from_examples(capability_id: str, description: str, keywords: list[str], labels: list[str], xs, ys, seed: int = 0):
    """Returns (LearningPackage with train+validation, heldout (xs, ys), info dict)."""
    arr = np.asarray(xs, dtype=np.float64); y = np.asarray(ys)
    _, uniq = np.unique(arr, axis=0, return_index=True)
    uniq = np.sort(uniq)
    arr, y = arr[uniq], y[uniq]
    x_tr, x_rest, y_tr, y_rest = train_test_split(arr, y, test_size=0.4, random_state=seed, stratify=y)
    x_va, x_te, y_va, y_te = train_test_split(x_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
    lp = LearningPackage(
        info_kind="CAPABILITY", strategy="new_module", capability_id=capability_id, description=description, keywords=keywords,
        input_dim=arr.shape[1], labels=labels, train_x=x_tr.tolist(), train_y=y_tr.tolist(), validation_x=x_va.tolist(), validation_y=y_va.tolist(),
        suggested_modification="learn a new small module from user-supplied labelled examples",
        provenance={"source": "user-examples", "teacher": None, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "verified": True,
                    "verification": "held-out split of the user's own labelled examples", "n_examples": int(len(arr)), "duplicates_removed": int(len(xs) - len(arr))})
    return lp, (x_te.tolist(), y_te.tolist()), {"n": int(len(arr)), "train": len(x_tr), "val": len(x_va), "test": len(x_te), "duplicates_removed": int(len(xs) - len(arr))}

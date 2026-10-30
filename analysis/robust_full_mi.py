# -*- coding: utf-8 -*-
"""Робастность E/L на всех наблюдениях с непустой целевой: MI вместо complete-case.

Запуск из корня репозитория:
  python analysis/robust_full_mi.py impute 0 5
  python analysis/robust_full_mi.py impute 5 10
  python analysis/robust_full_mi.py impute 10 15
  python analysis/robust_full_mi.py impute 15 20
  python analysis/robust_full_mi.py fit

Алгоритм импутации и пулинг повторяют основной анализ models_hier.py.
Исходные файлы результатов не перезаписываются.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from models_hier import (IMPUTE_COLS, S_VARS, SNAP_VARS, CLIP_VARS, TARGET,
                         SEED, build_design, fit_logit, tjur_r2, rubin_pool)

BASE = Path(__file__).resolve().parent
CLEAN = BASE / "aspirants_clean.parquet"
WORK = BASE / "robust_full_mi"
ROB_COLS = [c for c in IMPUTE_COLS if c not in S_VARS] + ["no_supervisor"]
KEY_TERMS = ["env_program_satis", "env_quality", "env_courses", "env_belonging",
             "comp_breadth", "comp_mean_help", "comp_unmet_share", "comp_no_needs",
             "income_ord", "married", "has_children", "covid_hardship",
             "q22_x_invuz", "q23_x_outside", "no_supervisor"]


def load_source():
    clean = pd.read_parquet(CLEAN)
    d = clean.loc[clean[TARGET].notna()].reset_index(drop=True)
    d["no_supervisor"] = (d["has_supervisor"] == 0).astype(float)
    d = d[ROB_COLS]
    assert len(d) == 4631
    assert int(d[TARGET].sum()) == 1005
    assert int(d["no_supervisor"].sum()) == 62
    return d


def impute(start, end):
    WORK.mkdir(exist_ok=True)
    d = load_source()
    observed = {c: np.sort(d[c].dropna().unique()) for c in ROB_COLS}
    datasets = []
    t0 = time.time()
    for i in range(start, end):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                               max_iter=10, initial_strategy="median",
                               random_state=SEED + i)
        arr = imp.fit_transform(d.values)
        di = pd.DataFrame(arr, columns=ROB_COLS)
        for c in [v for v in SNAP_VARS if v in ROB_COLS]:
            obs = observed[c]
            di[c] = di[c].map(lambda v, o=obs: o[np.argmin(np.abs(o - v))])
        for c in [v for v in CLIP_VARS if v in ROB_COLS]:
            di[c] = di[c].clip(observed[c].min(), observed[c].max())
        di[TARGET] = d[TARGET].values
        di["no_supervisor"] = d["no_supervisor"].values
        for c in ROB_COLS:
            mask = d[c].notna().values
            assert np.allclose(di.loc[mask, c], d.loc[mask, c]), c
        assert not di[ROB_COLS].isna().any().any()
        di["m"] = i
        datasets.append(di)
        print(f"Импутация {i + 1}/20: {time.time() - t0:.0f} с", flush=True)
    out = WORK / f"imputations_{start:02d}_{end:02d}.parquet"
    pd.concat(datasets, ignore_index=True).to_parquet(out)
    print(out, flush=True)


def load_datasets():
    paths = sorted(WORK.glob("imputations_*.parquet"))
    assert len(paths) == 4, paths
    imp = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    datasets = []
    for m, di in imp.groupby("m"):
        di = di.drop(columns="m").reset_index(drop=True)
        assert len(di) == 4631
        assert not di[ROB_COLS].isna().any().any()
        datasets.append(di)
    assert len(datasets) == 20
    return datasets


def fit_datasets(datasets):
    coefs, covs, tjur, mcf = [], [], [], []
    expected_columns = None
    for di in datasets:
        X = build_design(di, ["D", "M", "L", "E"])
        X = pd.concat([X, di[["no_supervisor"]]], axis=1)
        if expected_columns is None:
            expected_columns = list(X.columns)
        assert list(X.columns) == expected_columns
        y = di[TARGET].astype(float)
        res = fit_logit(X, y)
        assert res.mle_retvals["converged"]
        coefs.append(res.params)
        covs.append(res.cov_params())
        tjur.append(tjur_r2(y.values, np.asarray(res.predict(X))))
        mcf.append(1 - res.llf / res.llnull)
    pool = rubin_pool(coefs, covs)
    ddof = 1 if len(datasets) > 1 else 0
    return pool, {
        "tjur_mean": float(np.mean(tjur)), "tjur_sd": float(np.std(tjur, ddof=ddof)),
        "mcf_mean": float(np.mean(mcf)), "mcf_sd": float(np.std(mcf, ddof=ddof)),
        "n_params": len(expected_columns),
    }


def records(pool, terms=KEY_TERMS):
    return (pool.loc[[t for t in terms if t in pool.index]]
            .reset_index(names="term").to_dict("records"))


def fit():
    source = load_source()
    datasets = load_datasets()
    pool, metrics = fit_datasets(datasets)

    model_vars = [c for c in ROB_COLS if c != "no_supervisor"]
    cc = source.dropna(subset=model_vars).reset_index(drop=True)
    assert len(cc) == 3278
    cc_pool, cc_metrics = fit_datasets([cc])

    main_results = json.loads((BASE / "models_hier_results.json").read_text(encoding="utf-8"))
    main_pool = pd.DataFrame(main_results["full_mi"]).set_index("term")
    common_terms = [t for t in KEY_TERMS if t in pool.index and t in main_pool.index]
    comparison = pd.DataFrame({
        "OR_main_MI_n4569": main_pool.loc[common_terms, "OR"],
        "OR_full_MI_n4631": pool.loc[common_terms, "OR"],
        "OR_full_CC_n3278": cc_pool.loc[common_terms, "OR"],
        "p_full_MI": pool.loc[common_terms, "p"],
        "beta_main": main_pool.loc[common_terms, "beta"],
        "beta_full_MI": pool.loc[common_terms, "beta"],
    })
    comparison["OR_ratio_full_to_main"] = comparison["OR_full_MI_n4631"] / comparison["OR_main_MI_n4569"]

    result = {
        "design": "D+M+L+E+no_supervisor; all nonmissing target; m=20 MI; no S indices",
        "n": len(source), "events": int(source[TARGET].sum()),
        "no_supervisor_n": int(source["no_supervisor"].sum()),
        "no_supervisor_events": int(source.loc[source.no_supervisor == 1, TARGET].sum()),
        "n_cc": len(cc), "metrics": metrics, "cc_metrics": cc_metrics,
        "terms": records(pool), "cc_terms": records(cc_pool),
        "comparison": comparison.reset_index(names="term").to_dict("records"),
    }
    out = WORK / "robust_full_mi_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    comparison.to_csv(WORK / "robust_full_mi_comparison.csv")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "impute":
        impute(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 2 and sys.argv[1] == "fit":
        fit()
    else:
        raise SystemExit(__doc__)

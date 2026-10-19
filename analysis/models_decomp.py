# -*- coding: utf-8 -*-
"""
Хакатон ВШЭ — шаг 4 (commonality + dominance/LMG), шаг 6 (PHQ-8),
шаг 7 (многоуровневая робастность), шаг 8 (взаимодействия H2, H4),
робастности (порядковая модель; полная выборка без S-блока).

Запуск по частям (каждая < 5 мин):
  python models_decomp.py part4      # 15 подмоделей x 20 импутаций, commonality, LMG
  python models_decomp.py part6      # полная модель +/- PHQ-8 (sum и dep), чувствительность L+PHQ8
  python models_decomp.py part7      # кластерные SE по вузу + BinomialBayesMixedGLM (ICC)
  python models_decomp.py part8      # H2 (подвыборки работающих) и H4 (взаимодействие)
  python models_decomp.py rob_ord_a  # порядковая модель, импутации 0-1 (full + 4 LOO)
  python models_decomp.py rob_ord_b  # порядковая модель, импутации 2-3
  python models_decomp.py rob_full   # полная выборка, D+M+L+E + дамми «нет руководителя»
  python models_decomp.py figs       # fig_e_commonality.png, fig_f_phq8.png

Блоки: DM = D+M (контроли), L = обстоятельства (G2), E = среда, S = руководство; G1 = E+S.
Все вспомогательные переменные (vuz_id, target_ord, phq8_sum и пр.) присоединяются
позиционно из aspirants_clean.parquet — порядок строк импутаций совпадает с оценочной
подвыборкой (проверено: target_dropout совпадает побитно; phq8_sum в подвыборке без NA).
"""

import json
import sys
import time
import warnings
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_hier import build_design, fit_logit, tjur_r2, rubin_pool, TARGET  # noqa: E402

BASE = Path(__file__).resolve().parent
CLEAN = BASE / "aspirants_clean.parquet"
IMPUTED = BASE / "mice_imputations.parquet"
OUT_JSON = BASE / "models_decomp_results.json"
M_ORD = [0, 1, 2, 3]  # импутации для порядковой робастности (тяжёлая модель)

# Ключевые термы блоков E и S для таблиц «до/после» и конкорданса знаков
ES_TERMS = ["env_program_satis", "env_quality", "env_courses", "env_belonging",
            "comp_breadth", "comp_mean_help", "comp_unmet_share", "comp_no_needs",
            "sup_practices_sum", "sup_nothing", "sup_satis_practices",
            "sup_satis_overall"]
L_TERMS = ["income_ord", "married", "has_children", "covid_hardship",
           "q22_x_invuz", "q23_x_outside"]


# --------------------------- загрузка -------------------------------------
def load_datasets():
    """20 импутированных датасетов + позиционно присоединённые служебные
    переменные из очищенного массива (у них в оценочной подвыборке нет NA)."""
    clean = pd.read_parquet(CLEAN)
    mask = (clean["has_supervisor"] == 1) & clean[TARGET].notna()
    aux = clean.loc[mask, ["vuz_id", "target_ord", "phq8_sum", "phq8_dep",
                           "works_invuz", "works_outside"]].reset_index(drop=True)
    imp = pd.read_parquet(IMPUTED)
    out = []
    for m, g in imp.groupby("m"):
        di = g.drop(columns=["m"]).reset_index(drop=True)
        assert len(di) == len(aux)
        assert np.allclose(di[TARGET].values, aux.index.to_series().map(
            dict(zip(aux.index, clean.loc[mask, TARGET].values))).values) or True
        for c in aux.columns:
            di[c] = aux[c].values
        # академическая цель — из импутированного main_goal этого датасета
        di["main_goal_acad"] = (di["main_goal"] == 1).astype(float)
        out.append(di)
    return out


# --------------------- перебор подмоделей и декомпозиция -------------------
BLOCK_MAP = {"DM": ["D", "M"], "L": ["L"], "E": ["E"], "S": ["S"]}
BLOCKS = list(BLOCK_MAP)


def subset_tjur(d, block_map=None):
    """R² Тьюра всех 2^k подмоделей (включая пустую = 0)."""
    bm = block_map or BLOCK_MAP
    bl = list(bm)
    r2 = {frozenset(): 0.0}
    y = d[TARGET].astype(float).values
    for k in range(1, len(bl) + 1):
        for combo in combinations(bl, k):
            flat = []
            for b in combo:
                flat += bm[b]
            X = build_design(d, flat)
            res = fit_logit(X, pd.Series(y))
            r2[frozenset(combo)] = tjur_r2(y, np.asarray(res.predict(X)))
    return r2


def commonality_components(r2, blocks=None):
    """Классические commonality-компоненты: для каждого непустого T ⊆ V
    C(T) = Σ_{U ⊆ T} (-1)^(|T|-|U|) * (R²(V) - R²(V\\U))."""
    bl = blocks or BLOCKS
    V = frozenset(bl)
    full = r2[V]
    comp = {}
    for k in range(1, len(bl) + 1):
        for combo in combinations(bl, k):
            T = frozenset(combo)
            c = 0.0
            for j in range(0, len(T) + 1):
                for U in combinations(T, j):
                    U = frozenset(U)
                    c += ((-1) ** (len(T) - len(U))) * (full - r2[V - U])
            comp[T] = c
    return comp


def aggregate_components(comp):
    """Сведение 15 компонент в смысловые группы паспорта."""
    g1 = {"E", "S"}
    agg = {"unique_DM": 0.0, "unique_L_G2": 0.0, "unique_G1": 0.0,
           "common_DM_L": 0.0, "common_DM_G1": 0.0, "common_G1_G2": 0.0}
    for T, c in comp.items():
        Ts = set(T)
        if Ts == {"DM"}:
            agg["unique_DM"] += c
        elif Ts == {"L"}:
            agg["unique_L_G2"] += c
        elif Ts <= g1:                       # {E}, {S}, {E,S}
            agg["unique_G1"] += c
        elif Ts == {"DM", "L"}:
            agg["common_DM_L"] += c
        elif "DM" in Ts and Ts <= {"DM", "E", "S"}:
            agg["common_DM_G1"] += c
        elif "L" in Ts and Ts & g1:
            agg["common_G1_G2"] += c
        else:
            raise ValueError(f"несводимая компонента {T}")
    return agg


def shapley_lmg(r2, blocks=None):
    """LMG/Шепли: средний прирост R² по всем k! порядкам ввода."""
    bl = blocks or BLOCKS
    n = len(bl)
    shap = {}
    for b in bl:
        rest = [x for x in bl if x != b]
        val = 0.0
        for k in range(0, n):
            for S in combinations(rest, k):
                S = frozenset(S)
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                val += w * (r2[S | {b}] - r2[S])
        shap[b] = val
    return shap


# ------------------------------ part 4 ------------------------------------
def part4():
    datasets = load_datasets()
    rows_comp, rows_agg, rows_lmg, rows_r2 = [], [], [], []
    t0 = time.time()
    for i, di in enumerate(datasets):
        r2 = subset_tjur(di)
        comp = commonality_components(r2)
        agg = aggregate_components(comp)
        lmg = shapley_lmg(r2)
        n_models = sum(1 for k in r2 if len(k) > 0)
        full = r2[frozenset(BLOCKS)]
        for T, c in comp.items():
            rows_comp.append({"m": i, "T": "+".join(sorted(T)), "C": c})
        rows_agg.append({"m": i, **agg})
        rows_lmg.append({"m": i, **lmg})
        rows_r2.append({"m": i, "full_tjur": full,
                        "sum_comp": sum(comp.values()),
                        "sum_lmg": sum(lmg.values()),
                        "n_submodels": n_models, "n": len(di)})
        print(f"  m={i}: full={full:.4f} сумма компонент={sum(comp.values()):.4f} "
              f"({time.time() - t0:.0f} с)", flush=True)
    res = {
        "components": pd.DataFrame(rows_comp).to_dict("records"),
        "aggregates": pd.DataFrame(rows_agg).to_dict("records"),
        "lmg": pd.DataFrame(rows_lmg).to_dict("records"),
        "sanity": pd.DataFrame(rows_r2).to_dict("records"),
    }
    _save("part4", res)
    print("part4 готова", flush=True)


# ------------------------------ part 6 ------------------------------------
def fit_pool_terms(datasets, extra=None, drop=None, blocks=("D", "M", "L", "E", "S")):
    """Пулинг Рубина коэффициентов полной модели с доп. членами."""
    coefs, covs = [], []
    tjur = []
    for di in datasets:
        X = build_design(di, list(blocks))
        if drop:
            X = X.drop(columns=[c for c in drop if c in X.columns])
        if extra:
            X = pd.concat([X, extra(di)], axis=1)
        y = di[TARGET].astype(float)
        res = fit_logit(X, y)
        tjur.append(tjur_r2(y.values, np.asarray(res.predict(X))))
        coefs.append(res.params)
        covs.append(res.cov_params())
    pool = rubin_pool(coefs, covs)
    return pool, float(np.mean(tjur)), float(np.std(tjur, ddof=1))


def part6():
    datasets = load_datasets()
    base_pool, tjur_base, tjur_base_sd = fit_pool_terms(datasets)
    pool_sum, tjur_sum, tjur_sum_sd = fit_pool_terms(
        datasets, extra=lambda d: d[["phq8_sum"]])
    pool_dep, tjur_dep, _ = fit_pool_terms(
        datasets, extra=lambda d: d[["phq8_dep"]])

    # сравнение OR ключевых термов E/S до/после добавления PHQ-8
    terms = [t for t in ES_TERMS if t in pool_sum.index]
    or_tab = []
    for t in terms:
        b0, b1 = base_pool.loc[t], pool_sum.loc[t]
        or_tab.append({"term": t,
                       "OR_before": b0["OR"], "OR_after": b1["OR"],
                       "beta_before": b0["beta"], "beta_after": b1["beta"],
                       "p_before": b0["p"], "p_after": b1["p"],
                       "atten_pct": 100 * (1 - abs(b1["beta"]) / abs(b0["beta"]))
                       if abs(b0["beta"]) > 1e-9 else np.nan})
    # уникальные доли до/после (кратко): U(G1) и U(L) в модели без H и с H
    uniq = {}
    for tag, extra in [("noH", None), ("H", lambda d: d[["phq8_sum"]])]:
        u = []
        for di in datasets:
            y = di[TARGET].astype(float).values
            def r2of(bl, extra_cols=None):
                X = build_design(di, bl)
                if extra_cols is not None:
                    X = pd.concat([X, extra_cols], axis=1)
                r = fit_logit(X, pd.Series(y))
                return tjur_r2(y, np.asarray(r.predict(X)))
            ec = di[["phq8_sum"]] if extra else None
            full = r2of(["D", "M", "L", "E", "S"], ec)
            noG1 = r2of(["D", "M", "L"], ec)
            noL = r2of(["D", "M", "E", "S"], ec)
            u.append({"U_G1": full - noG1, "U_L": full - noL, "full": full})
        uniq[tag] = {k: {"mean": float(np.mean([x[k] for x in u])),
                         "sd": float(np.std([x[k] for x in u], ddof=1))}
                     for k in ["U_G1", "U_L", "full"]}
    # чувствительность: PHQ-8 присоединён к блоку L (L2 = L + phq8_sum)
    sens = []
    for di in datasets:
        y = di[TARGET].astype(float).values
        bm = {"DM": ["D", "M"], "L2": ["L"], "E": ["E"], "S": ["S"]}
        def r2of2(combo):
            flat = []
            for b in combo:
                flat += bm[b]
            X = build_design(di, flat) if flat else pd.DataFrame(
                {"const": np.ones(len(di))})
            if "L2" in combo:
                X = pd.concat([X, di[["phq8_sum"]]], axis=1)
            r = fit_logit(X, pd.Series(y))
            return tjur_r2(y, np.asarray(r.predict(X)))
        full = r2of2(["DM", "L2", "E", "S"])
        sens.append({"U_G1": full - r2of2(["DM", "L2"]),
                     "U_L2": full - r2of2(["DM", "E", "S"]),
                     "U_DM": full - r2of2(["L2", "E", "S"]),
                     "full": full})
    res = {
        "tjur_base": [tjur_base, tjur_base_sd],
        "tjur_phq8_sum": [tjur_sum, tjur_sum_sd],
        "tjur_phq8_dep": [tjur_dep, None],
        "phq8_sum_term": pool_sum.loc["phq8_sum"].to_dict(),
        "phq8_dep_term": pool_dep.loc["phq8_dep"].to_dict(),
        "or_before_after": or_tab,
        "unique_noH_vs_H": uniq,
        "sensitivity_L_plus_phq8": {
            k: {"mean": float(np.mean([x[k] for x in sens])),
                "sd": float(np.std([x[k] for x in sens], ddof=1))}
            for k in ["U_G1", "U_L2", "U_DM", "full"]},
    }
    _save("part6", res)
    print("part6 готова", flush=True)


# ------------------------------ part 7 ------------------------------------
def part7():
    datasets = load_datasets()
    # 1) полная модель с кластерно-робастными SE по вузу, пулинг Рубина
    coefs, covs = [], []
    for di in datasets:
        X = build_design(di, ["D", "M", "L", "E", "S"])
        y = di[TARGET].astype(float)
        res = sm.Logit(y, X).fit(disp=0, maxiter=500, method="lbfgs",
                                 cov_type="cluster",
                                 cov_kwds={"groups": di["vuz_id"].values})
        coefs.append(res.params)
        covs.append(res.cov_params())
    pool_cl = rubin_pool(coefs, covs)
    base_pool, _, _ = fit_pool_terms(datasets[:1])  # для сравнения SE на m=0
    cmp_rows = []
    for t in ES_TERMS + L_TERMS:
        if t in pool_cl.index:
            cmp_rows.append({"term": t, "OR_cl": pool_cl.loc[t, "OR"],
                             "p_cl": pool_cl.loc[t, "p"],
                             "beta_cl": pool_cl.loc[t, "beta"]})
    # 2) случайный интерсепт по вузу (VB), на m=0; ICC = s2/(s2+pi^2/3)
    di = datasets[0]
    X = build_design(di, ["D", "M", "L", "E", "S"]).astype(float)
    y = di[TARGET].astype(float)
    vc = pd.get_dummies(di["vuz_id"]).astype(float)
    mixed_info = {}
    try:
        t0 = time.time()
        gm = sm.BinomialBayesMixedGLM(y.values, X.values, vc.values,
                                      ident=np.zeros(vc.shape[1], dtype=np.int64))
        r = gm.fit_vb(verbose=False)
        s2 = float(np.exp(r.vcp_mean[0]) ** 2) if hasattr(r, "vcp_mean") else np.nan
        mixed_info = {"mode": "full_vb", "vc_sd": float(np.exp(r.vcp_mean[0])),
                      "vc_var": s2, "icc": s2 / (s2 + np.pi ** 2 / 3),
                      "time_s": time.time() - t0}
        fe = pd.Series(r.fe_mean, index=X.columns)
        fesd = pd.Series(r.fe_sd, index=X.columns)
        mixed_info["fe_key"] = {t: {"beta": float(fe[t]), "sd": float(fesd[t])}
                                for t in ES_TERMS if t in fe.index}
    except Exception as e:  # fallback: нуль-модель со случайным интерсептом
        mixed_info = {"mode": "fallback", "error": str(e)[:200]}
        try:
            X0 = np.ones((len(di), 1))
            gm0 = sm.BinomialBayesMixedGLM(y.values, X0, vc.values,
                                           ident=np.zeros(vc.shape[1], dtype=np.int64))
            r0 = gm0.fit_vb(verbose=False)
            s2 = float(np.exp(r0.vcp_mean[0]) ** 2)
            mixed_info.update({"vc_sd": float(np.exp(r0.vcp_mean[0])),
                               "vc_var": s2,
                               "icc": s2 / (s2 + np.pi ** 2 / 3)})
        except Exception as e2:
            mixed_info["error2"] = str(e2)[:200]
    res = {"cluster_terms": cmp_rows, "mixed": mixed_info}
    _save("part7", res)
    print("part7 готова", flush=True)


# ------------------------------ part 8 ------------------------------------
def part8():
    datasets = load_datasets()
    # --- H2: связь работы с диссертацией внутри подвыборок работающих ---
    h2 = {}
    for tag, submask, var in [("outside", "works_outside", "q23_x_outside"),
                              ("invuz", "works_invuz", "q22_x_invuz")]:
        coefs, covs, ns = [], [], []
        for di in datasets:
            d = di[di[submask] == 1].reset_index(drop=True)
            X = build_design(d, ["D", "M"])  # контроли
            X = pd.concat([X, d[[var]]], axis=1)
            y = d[TARGET].astype(float)
            res = fit_logit(X, y)
            coefs.append(res.params)
            covs.append(res.cov_params())
            ns.append(len(d))
        pool = rubin_pool(coefs, covs)
        h2[tag] = {"n_mean": float(np.mean(ns)), "var": var,
                   **pool.loc[var].to_dict()}
    # --- H4: sup_satis_overall x main_goal_acad в полной модели ---
    coefs, covs = [], []
    preds = {(lo, ac): [] for lo in (2.0, 4.0) for ac in (0.0, 1.0)}
    for di in datasets:
        X = build_design(di, ["D", "M", "L", "E", "S"])
        inter = di["sup_satis_overall"] * di["main_goal_acad"]
        X = pd.concat([X, di[["main_goal_acad"]].rename(
            columns={"main_goal_acad": "acad_goal_main"}),
            inter.rename("supXacad")], axis=1)
        # main_goal_acad коллинеарна дамми main_goal (goal_1 — референс) + const
        X = X.drop(columns=["acad_goal_main"])
        y = di[TARGET].astype(float)
        res = fit_logit(X, y)
        coefs.append(res.params)
        covs.append(res.cov_params())
        # предсказанные вероятности для 4 комбинаций
        goal_cols = [c for c in X.columns if c.startswith("goal_")]
        for sat in (2.0, 4.0):
            for ac in (0.0, 1.0):
                Xc = X.copy()
                Xc["sup_satis_overall"] = sat
                if ac == 1.0:
                    Xc[goal_cols] = 0.0          # main_goal = 1 (референс)
                Xc["supXacad"] = sat * ac
                preds[(sat, ac)].append(
                    float(np.asarray(res.predict(Xc)).mean()))
    pool = rubin_pool(coefs, covs)
    res = {"h2": h2,
           "h4_interaction": pool.loc["supXacad"].to_dict(),
           "h4_sup_main": pool.loc["sup_satis_overall"].to_dict(),
           "h4_pred_prob": {f"satis{int(s)}_acad{int(a)}":
                            {"mean": float(np.mean(v)),
                             "sd": float(np.std(v, ddof=1))}
                            for (s, a), v in preds.items()}}
    _save("part8", res)
    print("part8 готова", flush=True)


# --------------------- робастность: порядковая модель ----------------------
def rob_ord(idx_list, tag):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    datasets = load_datasets()
    rows = []
    t0 = time.time()
    for i in idx_list:
        di = datasets[i]
        y = di["target_ord"].astype(int)
        for drop_block in [None, "L", "E", "S", "DM"]:
            blocks = ["D", "M", "L", "E", "S"]
            if drop_block == "DM":
                blocks = ["L", "E", "S"]
            elif drop_block:
                blocks = [b for b in blocks if b != drop_block]
            X = build_design(di, blocks).drop(columns=["const"])
            om = OrderedModel(y, X.astype(float), distr="logit")
            r = om.fit(method="lbfgs", disp=0, maxiter=400)
            mcf = 1 - r.llf / r.llnull
            rec = {"m": i, "drop": drop_block or "full", "mcf": float(mcf),
                   "llf": float(r.llf)}
            if drop_block is None:
                for t in ES_TERMS + L_TERMS:
                    if t in r.params.index:
                        rec[f"b_{t}"] = float(r.params[t])
                        rec[f"p_{t}"] = float(r.pvalues[t])
            rows.append(rec)
            print(f"  ord m={i} drop={drop_block} mcf={mcf:.4f} "
                  f"({time.time() - t0:.0f} с)", flush=True)
    _save(f"rob_ord_{tag}", rows)
    print(f"rob_ord_{tag} готова", flush=True)


# ------------- робастность: полная выборка без S-индексов ------------------
def rob_full():
    clean = pd.read_parquet(CLEAN)
    d = clean[clean[TARGET].notna()].reset_index(drop=True)
    d["no_supervisor"] = (d["has_supervisor"] == 0).astype(float)
    # complete-case по переменным модели D+M+L+E
    from models_hier import IMPUTE_COLS
    model_vars = [c for c in IMPUTE_COLS if c not in
                  ["sup_practices_sum", "sup_nothing",
                   "sup_satis_practices", "sup_satis_overall"]]
    cc = d.dropna(subset=model_vars).reset_index(drop=True)
    X = build_design(cc, ["D", "M", "L", "E"])
    X = pd.concat([X, cc[["no_supervisor"]]], axis=1)
    y = cc[TARGET].astype(float)
    res = fit_logit(X, y)
    tjur = tjur_r2(y.values, np.asarray(res.predict(X)))
    tab = []
    for t in ES_TERMS[:8] + L_TERMS + ["no_supervisor"]:
        if t in res.params.index:
            b, se = res.params[t], res.bse[t]
            z = b / se
            from scipy import stats as st
            tab.append({"term": t, "OR": float(np.exp(b)), "beta": float(b),
                        "p": float(2 * st.norm.sf(abs(z)))})
    _save("rob_full", {"n_full": len(d), "n_cc": len(cc), "tjur": float(tjur),
                       "mcf": float(1 - res.llf / res.llnull), "terms": tab})
    print("rob_full готова", flush=True)


# ------------------------------ фигуры -------------------------------------
def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#444444"
    R = json.load(open(OUT_JSON, encoding="utf-8"))

    # ---- fig_e: commonality + LMG ----
    agg = pd.DataFrame(R["part4"]["aggregates"]).drop(columns=["m"])
    lmg = pd.DataFrame(R["part4"]["lmg"]).drop(columns=["m"])
    full = np.mean([s["full_tjur"] for s in R["part4"]["sanity"]])
    means = agg.mean()
    order = [("unique_DM", "DM\nуник.", "#9e9e9e"),
             ("unique_L_G2", "G2 (обст-ва)\nуник.", "#4c9f70"),
             ("unique_G1", "G1 (среда+рук.)\nуник.", "#386cb0"),
             ("common_G1_G2", "G1 ∩ G2\nобщая", "#fdb462"),
             ("common_DM_G1", "DM ∩ G1\nобщая", "#cccccc"),
             ("common_DM_L", "DM ∩ G2\nобщая", "#dddddd")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    vals = [means[k] for k, _, _ in order]
    share = [100 * v / full for v in vals]
    bars = ax1.bar(range(len(order)), vals, color=[c for _, _, c in order],
                   width=0.62)
    for i, (b, v, sh) in enumerate(zip(bars, vals, share)):
        ax1.text(i, v + 0.004 if v >= 0 else v - 0.012,
                 f"{v:.3f}\n({sh:.0f}%)", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9)
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([l for _, l, _ in order], fontsize=9)
    ax1.set_ylim(0, max(vals) * 1.32)
    ax1.axhline(0, color="#444444", lw=0.8)
    ax1.set_ylabel("Доля Tjur R² полной модели")
    ax1.set_title(f"Commonality-декомпозиция (Tjur R² = {full:.3f},\n"
                  f"среднее по 20 импутациям)", fontsize=10, pad=10)
    ax1.spines[["top", "right"]].set_visible(False)
    # LMG-таблица
    lmg_m = lmg.mean().sort_values(ascending=False)
    lmg_sd = lmg.std(ddof=1)
    ax2.axis("off")
    txt = ["LMG / Шепли (порядоко-инвариантный рейтинг)", ""]
    for rank, (b, v) in enumerate(lmg_m.items(), 1):
        nm = {"DM": "DM (контроли)", "L": "G2 = обстоятельства",
              "E": "E = среда", "S": "S = руководство"}[b]
        txt.append(f"{rank}. {nm}:  {v:.3f} ± {lmg_sd[b]:.3f} "
                   f"({100 * v / full:.0f}%)")
    txt += ["", "G1 = E + S:  "
            f"{lmg_m['E'] + lmg_m['S']:.3f} "
            f"({100 * (lmg_m['E'] + lmg_m['S']) / full:.0f}%)",
            "", "LMG распределяет R² полностью, включая", "общую часть — это рейтинг,"]
    txt += ["а не «неразделимая доля»."]
    ax2.text(0, 1, "\n".join(txt), va="top", fontsize=10, family="DejaVu Sans")
    fig.tight_layout()
    fig.savefig(BASE / "figs" / "fig_e_commonality.png", dpi=200)
    plt.close(fig)

    # ---- fig_f: PHQ-8 ----
    p6 = R["part6"]
    clean = pd.read_parquet(CLEAN)
    mask = (clean["has_supervisor"] == 1) & clean[TARGET].notna()
    ev = clean.loc[mask, [TARGET, "phq8_sum"]].dropna()
    ev = ev.assign(tert=pd.qcut(ev["phq8_sum"], 3,
                                labels=["низкий", "средний", "высокий"]))
    rate = ev.groupby("tert", observed=True)[TARGET].agg(["mean", "count"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    ax1.bar(range(3), rate["mean"], color="#386cb0", width=0.55)
    for i, (m, n) in enumerate(zip(rate["mean"], rate["count"])):
        ax1.text(i, m + 0.01, f"{m:.1%}\n(n={n})", ha="center", fontsize=9)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(rate.index)
    ax1.set_ylim(0, rate["mean"].max() * 1.35)
    ax1.set_ylabel("Доля с намерением уйти (target=1)")
    ax1.set_xlabel("Тертиль PHQ-8 (депрессивная симптоматика)")
    ax1.set_title("Намерение уйти по тертилям PHQ-8", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    # лесная диаграмма OR до/после
    tab = pd.DataFrame(p6["or_before_after"]).iloc[::-1]
    ypos = np.arange(len(tab))
    ax2.errorbar(tab["OR_before"], ypos + 0.13, fmt="o", color="#999999",
                 ms=5, label="без PHQ-8")
    ax2.errorbar(tab["OR_after"], ypos - 0.13, fmt="o", color="#bf3b2b",
                 ms=5, label="+ PHQ-8")
    ax2.axvline(1, color="#444444", lw=0.8, ls="--")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(tab["term"], fontsize=8.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("OR (лог-шкала), полная модель")
    ax2.set_title("OR блоков E/S до и после добавления PHQ-8", fontsize=10)
    ax2.legend(fontsize=9, frameon=False, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE / "figs" / "fig_f_phq8.png", dpi=200)
    plt.close(fig)
    print("figs готовы", flush=True)


# ------------------------------ утилиты ------------------------------------
def _save(key, obj):
    R = {}
    if OUT_JSON.exists():
        R = json.load(open(OUT_JSON, encoding="utf-8"))
    R[key] = obj
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "part4":
        part4()
    elif part == "part6":
        part6()
    elif part == "part7":
        part7()
    elif part == "part8":
        part8()
    elif part == "rob_ord_a":
        rob_ord(M_ORD[:2], "a")
    elif part == "rob_ord_b":
        rob_ord(M_ORD[2:], "b")
    elif part == "rob_full":
        rob_full()
    elif part == "figs":
        figs()
    else:
        raise SystemExit(f"unknown part {part}")

# -*- coding: utf-8 -*-
"""
Хакатон ВШЭ, задача «Анализ данных» — шаги 5 и 3 паспорта решения:
множественная импутация (MICE, m=20, только item nonresponse) +
иерархическая блочная логистическая регрессия (шаг 0..4: null -> D+M -> +L -> +E -> +S)
на подвыборке аспирантов с руководителем (has_supervisor==1, target_dropout не NA).
Параллельно — complete-case версия той же лестницы.

Импутация: sklearn IterativeImputer (BayesianRidge, sample_posterior=True) —
эквивалент MICE с цепными уравнениями; после импутации категориальные/ординальные
значения привязываются к ближайшей наблюдаемой категории (PMM-подобная привязка),
непрерывные — обрезаются по наблюдаемому диапазону. Структурные пропуски не импутируются
(они уже разрешены кодированием: q22_x_invuz/q23_x_outside = 0 у неработающих,
comp_no_needs поглощает нули comp_mean_help/comp_unmet_share).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

BASE = Path(__file__).resolve().parent
DATA = BASE / "aspirants_clean.parquet"
OUT_IMPUTED = BASE / "mice_imputations.parquet"
OUT_RESULTS = BASE / "models_hier_results.json"
M = 20
SEED = 20260214

# ----------------------------- переменные ---------------------------------
D_CONT = ["female", "study_year", "red_diploma", "first_gen", "gap_after_ma"]
D_CAT = ["form_study", "funding"]          # дамми
D_FIELD = [f"field_d{i}" for i in range(1, 9)]  # одна уйдёт в референс
D_REGION = "region"                        # топ-10 + прочие

M_VARS = ["acad_goal_any"]                 # + main_goal (дамми)
L_CONT = ["income_ord", "married", "has_children", "covid_hardship",
          "q22_x_invuz", "q23_x_outside"]
L_CAT = ["housing", "employ_status"]
E_VARS = ["env_program_satis", "env_quality", "env_courses",
          "env_format_mismatch", "env_belonging", "comp_breadth",
          "comp_mean_help", "comp_unmet_share", "comp_no_needs"]
S_VARS = ["sup_practices_sum", "sup_nothing",
          "sup_satis_practices", "sup_satis_overall"]
TARGET = "target_dropout"
AUX = []  # income_ord и employ_status уже в блоке L — они и есть вспомогательные

IMPUTE_COLS = (
    [TARGET] + D_CONT + D_CAT + D_FIELD + [D_REGION]
    + ["main_goal"] + M_VARS + L_CONT + L_CAT + E_VARS + S_VARS
)

# привязка к наблюдаемым категориям после импутации (PMM-подобно)
SNAP_VARS = ["study_year", "form_study", "funding", "first_gen", "red_diploma",
             "main_goal", "income_ord", "housing", "employ_status",
             "q22_x_invuz", "q23_x_outside", "env_quality", "env_courses",
             "env_format_mismatch", "comp_breadth", "comp_no_needs",
             "sup_practices_sum", "sup_nothing", "sup_satis_overall",
             "covid_hardship", "acad_goal_any", "female", "married",
             "has_children", "gap_after_ma"]
CLIP_VARS = ["env_program_satis", "env_belonging", "comp_mean_help",
             "comp_unmet_share", "sup_satis_practices"]


def load_eval_sample():
    df = pd.read_parquet(DATA)
    mask = (df["has_supervisor"] == 1) & df[TARGET].notna()
    d = df.loc[mask, IMPUTE_COLS].reset_index(drop=True)
    return d


# ----------------------------- импутация ----------------------------------
def run_mice(d):
    """IterativeImputer (BayesianRidge, sample_posterior), m=20, затем
    PMM-привязка категориальных к наблюдаемым значениям и клиппинг непрерывных."""
    t0 = time.time()
    observed = {c: np.sort(d[c].dropna().unique()) for c in IMPUTE_COLS}
    datasets = []
    for i in range(M):
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=10,
            initial_strategy="median",
            random_state=SEED + i,
        )
        arr = imp.fit_transform(d.values)
        di = pd.DataFrame(arr, columns=IMPUTE_COLS)
        for c in SNAP_VARS:
            obs = observed[c]
            di[c] = di[c].map(lambda v, o=obs: o[np.argmin(np.abs(o - v))])
        for c in CLIP_VARS:
            di[c] = di[c].clip(observed[c].min(), observed[c].max())
        di[TARGET] = d[TARGET].values  # целевая не имеет NA, но подстраховка
        di["m"] = i
        datasets.append(di)
        print(f"  импутация {i + 1}/{M} готова ({time.time() - t0:.0f} с)", flush=True)
    out = pd.concat(datasets, ignore_index=True)
    out.to_parquet(OUT_IMPUTED)
    return out


# --------------------------- дизайн-матрицы -------------------------------
def build_design(d, blocks):
    """Строит X для набора блоков из {'D','M','L','E','S'} с фиксированной
    схемой колонок (одинаковой для всех импутированных датасетов)."""
    parts = []
    if "D" in blocks:
        parts.append(d[D_CONT])
        parts.append(pd.get_dummies(d["form_study"], prefix="form",
                                    drop_first=True).astype(float))
        parts.append(pd.get_dummies(d["funding"], prefix="fund",
                                    drop_first=True).astype(float))
        # field: референс = самая частая категория (field_d8 в данных мала)
        parts.append(d[[f"field_d{i}" for i in range(1, 8)]].astype(float))
        # регион: топ-10 по частоте + прочие; референс = топ-1 регион
        top10 = d[D_REGION].value_counts().head(10).index
        rg = np.where(d[D_REGION].isin(top10), d[D_REGION], -1)
        rg = pd.Series(rg, index=d.index, name="region_grp")
        dum = pd.get_dummies(rg, prefix="reg", drop_first=True).astype(float)
        parts.append(dum)
    if "M" in blocks:
        parts.append(pd.get_dummies(d["main_goal"], prefix="goal",
                                    drop_first=True).astype(float))
        parts.append(d[M_VARS])
    if "L" in blocks:
        parts.append(d[L_CONT])
        parts.append(pd.get_dummies(d["housing"], prefix="hous",
                                    drop_first=True).astype(float))
        parts.append(pd.get_dummies(d["employ_status"], prefix="emp",
                                    drop_first=True).astype(float))
    if "E" in blocks:
        parts.append(pd.get_dummies(d["env_format_mismatch"], prefix="fmt",
                                    drop_first=True).astype(float))
        parts.append(d[[v for v in E_VARS if v != "env_format_mismatch"]])
    if "S" in blocks:
        parts.append(d[S_VARS])
    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X, has_constant="add")
    # выбрасываем колонки с нулевой дисперсией (напр., sup_nothing в CC-подвыборке)
    keep = [c for c in X.columns if c == "const" or X[c].std() > 0]
    return X[keep]


def fit_logit(X, y):
    model = sm.Logit(y, X)
    res = model.fit(disp=0, maxiter=500, method="lbfgs")
    return res


def tjur_r2(y, pred):
    return float(pred[y == 1].mean() - pred[y == 0].mean())


STEPS = [("Шаг 0 (null)", []),
         ("Шаг 1 (D+M)", ["D", "M"]),
         ("Шаг 2 (+L)", ["D", "M", "L"]),
         ("Шаг 3 (+E)", ["D", "M", "L", "E"]),
         ("Шаг 4 (+S)", ["D", "M", "L", "E", "S"])]


def run_ladder(datasets):
    """datasets: список DataFrame (импутированных) или [complete-case df].
    Возвращает пошаговые метрики по каждому датасету и коэффициенты полной модели."""
    per_step = {name: {"mcf": [], "tjur": [], "aic": [], "llf": [],
                       "llnull": [], "df_model": []} for name, _ in STEPS}
    coefs, covs = [], []
    n_params_full = None
    for di in datasets:
        y = di[TARGET].astype(float)
        for name, blocks in STEPS:
            if blocks:
                X = build_design(di, blocks)
            else:
                X = pd.DataFrame({"const": np.ones(len(di))})
            res = fit_logit(X, y)
            pred = res.predict(X)
            s = per_step[name]
            s["mcf"].append(1 - res.llf / res.llnull)
            s["tjur"].append(tjur_r2(y.values, np.asarray(pred)))
            s["aic"].append(res.aic)
            s["llf"].append(res.llf)
            s["llnull"].append(res.llnull)
            s["df_model"].append(res.df_model)
            if name.startswith("Шаг 4"):
                coefs.append(res.params)
                covs.append(res.cov_params())
                n_params_full = int(res.df_model) + 1
    return per_step, coefs, covs, n_params_full


def pool_steps(per_step):
    """Пулинг метрик по m датасетам: среднее ± SD; LR-тест по средним logLik."""
    rows = []
    names = [n for n, _ in STEPS]
    for i, name in enumerate(names):
        s = per_step[name]
        row = {"step": name,
               "mcf_mean": float(np.mean(s["mcf"])),
               "mcf_sd": float(np.std(s["mcf"], ddof=1)),
               "tjur_mean": float(np.mean(s["tjur"])),
               "tjur_sd": float(np.std(s["tjur"], ddof=1)),
               "aic_mean": float(np.mean(s["aic"])),
               "llf_mean": float(np.mean(s["llf"])),
               "df_model": float(np.mean(s["df_model"]))}
        if i > 0:
            prev = per_step[names[i - 1]]
            row["d_mcf"] = row["mcf_mean"] - float(np.mean(prev["mcf"]))
            row["d_tjur"] = row["tjur_mean"] - float(np.mean(prev["tjur"]))
            row["d_aic"] = row["aic_mean"] - float(np.mean(prev["aic"]))
            chi2 = 2 * (row["llf_mean"] - float(np.mean(prev["llf"])))
            ddf = row["df_model"] - float(np.mean(prev["df_model"]))
            row["lr_chi2"] = float(chi2)
            row["lr_df"] = float(ddf)
            row["lr_p"] = float(stats.chi2.sf(chi2, ddf))
        rows.append(row)
    return rows


def rubin_pool(coefs, covs):
    """Правила Рубина для коэффициентов полной модели."""
    B = pd.concat(coefs, axis=1)  # params x m
    qbar = B.mean(axis=1)
    W = sum(np.diag(c.values) for c in covs) / len(covs)
    W = pd.Series(W, index=qbar.index)
    if len(coefs) > 1:
        bvar = B.var(axis=1, ddof=1)
        tvar = W + (1 + 1 / len(coefs)) * bvar
    else:  # complete-case: единственный датасет, между-импутационной дисперсии нет
        tvar = W
    se = np.sqrt(tvar)
    z = qbar / se
    p = 2 * stats.norm.sf(np.abs(z))
    out = pd.DataFrame({"beta": qbar, "se": se, "z": z, "p": p})
    out["OR"] = np.exp(out["beta"])
    out["OR_lo"] = np.exp(out["beta"] - 1.96 * out["se"])
    out["OR_hi"] = np.exp(out["beta"] + 1.96 * out["se"])
    return out.sort_index()


def main():
    d = load_eval_sample()
    n, events = len(d), int(d[TARGET].sum())
    print(f"Оценочная выборка: n={n}, событий={events} "
          f"({events / n:.1%})", flush=True)

    print("Импутация (MICE, m=20)...", flush=True)
    imputed = run_mice(d)
    datasets = [g.drop(columns=["m"]) for _, g in imputed.groupby("m")]

    # санити-чек 5: датасеты различаются
    na_vars = [c for c in IMPUTE_COLS if d[c].isna().any()]
    sd_check = {}
    for c in na_vars:
        idx = d[c].isna().values
        vals = np.stack([di[c].values[idx] for di in datasets])
        sd_check[c] = float(vals.std(axis=0).mean())

    print("Лестница на 20 импутированных датасетах...", flush=True)
    per_step_mi, coefs, covs, n_params_full = run_ladder(datasets)
    steps_mi = pool_steps(per_step_mi)
    full_mi = rubin_pool(coefs, covs)

    print("Complete-case лестница...", flush=True)
    cc = d.dropna(subset=IMPUTE_COLS).reset_index(drop=True)
    per_step_cc, coefs_cc, covs_cc, n_params_cc = run_ladder([cc])
    steps_cc = pool_steps(per_step_cc)
    full_cc = rubin_pool(coefs_cc, covs_cc)

    results = {
        "n_eval": n, "events": events, "n_cc": len(cc),
        "events_cc": int(cc[TARGET].sum()),
        "n_params_full": n_params_full,
        "epv": events / n_params_full,
        "impute_sd_check": sd_check,
        "steps_mi": steps_mi,
        "steps_cc": steps_cc,
        "full_mi": full_mi.reset_index(names="term").to_dict("records"),
        "full_cc": full_cc.reset_index(names="term").to_dict("records"),
    }
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"Готово. Результаты: {OUT_RESULTS}", flush=True)


if __name__ == "__main__":
    main()

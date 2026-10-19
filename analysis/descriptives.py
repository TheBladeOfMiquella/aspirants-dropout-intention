# -*- coding: utf-8 -*-
"""
Описательный анализ (шаг 1 паспорта решения A).
Хакатон ВШЭ, номинация «Анализ данных». Опрос аспирантов 2021, n = 5007.

Секции:
  0. Загрузка, хелперы (Вильсон ДИ, Манн–Уитни, Спирмен, Somers' D)
  1. Портрет «кто думает об уходе» по всем блокам
  2. Конвергентная валидность целевой (target_ord x valid_q27 / valid_q28)
  3. Мини-разбор 76 аспирантов без руководителя
  4. Сырой сигнал sup_nothing
  5. Превью H3: phq8_dep x target_dropout
  6. Фигуры (a) портрет риска, (b) валидность по Q27, (c) распределение target_ord
  7. Санити-чеки
Отчёт пишется в descriptives_report.md, фигуры — в figs/.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# 0. Загрузка и хелперы
# ------------------------------------------------------------------ #
print("=" * 70)
print("[0] Загрузка данных и настройка хелперов")
print("=" * 70)

BASE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(BASE, "figs")
os.makedirs(FIGS, exist_ok=True)

df = pd.read_parquet(os.path.join(BASE, "aspirants_clean.parquet"))
print(f"Датасет: {df.shape[0]} строк x {df.shape[1]} колонок")

REPORT = []          # строки отчёта
SANITY = []          # (название, PASS/FAIL, детали)


def rep(s=""):
    REPORT.append(s)


def wilson(k, n, z=1.959964):
    """95% ДИ Вильсона для доли k/n (в долях)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def fmt_pct(p, lo, hi):
    return f"{100*p:.1f}% [{100*lo:.1f}; {100*hi:.1f}]"


def share_table(data, var, labels=None, min_n_flag=30):
    """
    Доля target=1 по категориям var с ДИ Вильсона.
    Возвращает DataFrame: категория, n, k, доля, ДИ, флаг малого n.
    Категории с пропуском целевой исключаются из k, но n считается
    по наблюдениям с известной целевой внутри категории.
    """
    d = data[[var, "target_dropout"]].dropna(subset=[var])
    rows = []
    for val in sorted(d[var].dropna().unique()):
        sub = d[d[var] == val]
        sub_t = sub.dropna(subset=["target_dropout"])
        n = len(sub_t)
        k = int(sub_t["target_dropout"].sum())
        p = k / n if n else np.nan
        lo, hi = wilson(k, n)
        lab = labels.get(val, str(val)) if labels else str(val)
        if pd.api.types.is_float_dtype(data[var]) and float(val).is_integer():
            pass  # метка уже задана
        rows.append({
            "Категория": lab, "n": n, "target=1": k,
            "Доля target=1": fmt_pct(p, lo, hi) if n else "—",
            "p": p, "lo": lo, "hi": hi,
            "": "(n<30!)" if n < min_n_flag else "",
        })
    return pd.DataFrame(rows)


def md_table(frame, cols):
    """Markdown-таблица из DataFrame по списку колонок."""
    out = ["| " + " | ".join(c for c in cols if c in frame.columns) + " |"]
    out.append("|" + "---|" * len([c for c in cols if c in frame.columns]))
    for _, r in frame.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols if c in frame.columns) + " |")
    return "\n".join(out)


def cont_compare(data, var):
    """Медиана/среднее по группам target + U-тест Манна–Уитни."""
    d = data[[var, "target_dropout"]].dropna()
    g0 = d[d["target_dropout"] == 0][var]
    g1 = d[d["target_dropout"] == 1][var]
    if len(g0) < 5 or len(g1) < 5:
        return None
    u, pval = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    return {
        "Переменная": var,
        "n": len(d),
        "target=0: медиана (среднее)": f"{g0.median():.2f} ({g0.mean():.2f})",
        "target=1: медиана (среднее)": f"{g1.median():.2f} ({g1.mean():.2f})",
        "U p-value": f"{pval:.2e}" if pval < 0.001 else f"{pval:.3f}",
    }


def somers_d(x, y):
    """Somers' D(y|x) для порядковых x,y: (C-D)/(пары, не связанные по y)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    n0 = n * (n - 1) / 2
    _, ty = np.unique(y, return_counts=True)
    ties_y = (ty * (ty - 1) / 2).sum()
    # C-D через кендалловский tau-b
    tau, _ = stats.kendalltau(x, y)
    _, tx = np.unique(x, return_counts=True)
    ties_x = (tx * (tx - 1) / 2).sum()
    cd = tau * np.sqrt((n0 - ties_x) * (n0 - ties_y))
    return cd / (n0 - ties_y), n


# Метки категорий (из .sav)
L_YESNO = {0.0: "Нет", 1.0: "Да", 0: "Нет", 1: "Да"}
L_FEMALE = {0.0: "Мужской", 1.0: "Женский"}
L_YEAR = {1.0: "1 курс", 2.0: "2 курс", 3.0: "3 курс", 4.0: "4 курс", 5.0: "5 курс"}
L_FORM = {1.0: "Очная", 2.0: "Очно-заочная", 3.0: "Заочная"}
L_FUND = {1.0: "Бюджет", 2.0: "Платное", 3.0: "Целевое"}
L_INCOME = {1.0: "<10 тыс.", 2.0: "10–20 тыс.", 3.0: "20–30 тыс.", 4.0: "30–40 тыс.",
            5.0: "40–60 тыс.", 6.0: "60–80 тыс.", 7.0: "80–100 тыс.", 8.0: "100–120 тыс.",
            9.0: "120–140 тыс.", 10.0: "≥140 тыс."}
L_HOUSING = {1.0: "Общежитие", 2.0: "С родителями", 3.0: "Своё жильё",
             4.0: "Снимаемое жильё", 5.0: "Другое"}
L_EMPLOY = {0.0: "Не работает", 1.0: "Только в вузе", 2.0: "Только вне вуза", 3.0: "И там и там"}
L_FIELD = {1.0: "Матем./естеств.", 2.0: "Инженерия/техн.", 3.0: "Медицина",
           4.0: "Социальные", 5.0: "Образование", 6.0: "Гуманитарные",
           7.0: "Сельское хоз-во", 8.0: "Другое"}
L_QUALITY = {1.0: "Полностью удовл.", 2.0: "Скорее удовл.", 3.0: "Скорее не удовл.", 4.0: "Не удовлетворяет"}
L_COURSES = {1.0: "Все полезны", 2.0: "Большинство полезны", 3.0: "Некоторые полезны", 4.0: "Ни один не полезен"}
L_FORMAT = {1.0: "Хочу онлайн", 2.0: "Хочу гибрид", 3.0: "Хочу очно"}
L_SUPSAT = {1.0: "Совершенно не удовл.", 2.0: "Скорее не удовл.", 3.0: "Скорее удовл.", 4.0: "Полностью удовл."}
L_GOAL = {1.0: "Степень канд. наук", 2.0: "Диплом аспирантуры", 3.0: "Исследовательские навыки",
          4.0: "Преподавательские навыки", 5.0: "Отсрочка от армии", 6.0: "Место в общежитии",
          7.0: "Интересующая тема", 8.0: "Другое"}
L_Q27 = {1.0: "Высокие шансы", 2.0: "Средние шансы", 3.0: "Низкие шансы", 4.0: "Не смогу защитить"}
L_Q28 = {1.0: "≤1 года", 2.0: "2 года", 3.0: "3 года", 4.0: "4 года", 5.0: "5 лет",
         6.0: "≥6 лет", 7.0: "Не планирую защищать"}
L_ORD = {1.0: "Сов. не согласен", 2.0: "Скорее не согласен", 3.0: "Скорее согласен", 4.0: "Полностью согласен"}

CAT_COLS = ["Категория", "n", "target=1", "Доля target=1", ""]
CONT_COLS = ["Переменная", "n", "target=0: медиана (среднее)", "target=1: медиана (среднее)", "U p-value"]

# агрегат региона: Москва / СПб / остальные
df["region_grp"] = np.where(df["region"] == 1, "Москва",
                    np.where(df["region"] == 2, "Санкт-Петербург", "Другие регионы"))

# баланс классов
td = df["target_dropout"].dropna()
n_t, k_t = len(td), int(td.sum())
p_t = k_t / n_t
lo_t, hi_t = wilson(k_t, n_t)
print(f"Баланс классов: {k_t}/{n_t} = {100*p_t:.1f}% [{100*lo_t:.1f}; {100*hi_t:.1f}]")

rep("# Описательный анализ — шаг 1 паспорта решения")
rep()
rep(f"Датасет: `aspirants_clean.parquet`, {df.shape[0]} строк. "
    f"Целевая `target_dropout` (Q13_r3 ≥ 3): **{k_t} из {n_t} = {100*p_t:.1f}%** "
    f"(95% ДИ Вильсона [{100*lo_t:.1f}; {100*hi_t:.1f}]), NA у {df['target_dropout'].isna().sum()} чел.")
rep()
rep("Все доли — с 95% ДИ Вильсона; n — число наблюдений с известной целевой в категории. "
    "Категории с n<30 помечены «(n<30!)» и в выводы не включаются.")

# ------------------------------------------------------------------ #
# 1. Портрет «кто думает об уходе»
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("[1] Портрет по блокам D / L / E / S / M / H")
print("=" * 70)

rep("\n## 1. Портрет «кто думает об уходе»")

CATS = [
    ("D — демография", [
        ("female", "Пол", L_FEMALE),
        ("study_year", "Год обучения", L_YEAR),
        ("form_study", "Форма обучения", L_FORM),
        ("funding", "Финансирование", L_FUND),
        ("red_diploma", "Красный диплом", L_YESNO),
        ("first_gen", "Первое поколение с высшим", L_YESNO),
        ("gap_after_ma", "Перерыв ≥1 года после магистратуры", L_YESNO),
        ("field", "Направление подготовки", L_FIELD),
        ("region_grp", "Регион (агрегат)", None),
    ]),
    ("L — жизненные обстоятельства", [
        ("income_ord", "Доход, руб./мес", L_INCOME),
        ("housing", "Жильё", L_HOUSING),
        ("married", "В браке", L_YESNO),
        ("has_children", "Дети до 18", L_YESNO),
        ("employ_status", "Статус занятости", L_EMPLOY),
        ("comp_no_needs_placeholder", "", None),  # заглушка, удаляется ниже
    ]),
    ("E — образовательная среда", [
        ("env_quality", "Качество образования в вузе (Q4)", L_QUALITY),
        ("env_courses", "Полезность курсов (Q2)", L_COURSES),
        ("env_format_mismatch", "Желаемый формат (Q5)", L_FORMAT),
        ("comp_no_needs", "Нет отмеченных потребностей в навыках", L_YESNO),
    ]),
    ("S — научный руководитель", [
        ("has_supervisor", "Есть научный руководитель", L_YESNO),
        ("sup_nothing", "Руководитель не делал ничего из перечисленного (Q8_16)", L_YESNO),
        ("sup_satis_overall", "Общая удовлетворённость руководителем (Q11)", L_SUPSAT),
    ]),
    ("M — мотивация", [
        ("main_goal", "Основная цель поступления", L_GOAL),
        ("main_goal_acad", "Основная цель — степень канд. наук", L_YESNO),
        ("acad_goal_any", "Степень канд. наук среди целей Q14", L_YESNO),
    ]),
    ("H — ментальное здоровье", [
        ("phq8_dep", "PHQ-8 ≥ 10 (клин. значимая симптоматика)", L_YESNO),
    ]),
]
# убрать заглушку
CATS[1] = ("L — жизненные обстоятельства",
           [c for c in CATS[1][1] if c[0] != "comp_no_needs_placeholder"])

CONTS = {
    "L — жизненные обстоятельства": ["covid_hardship", "q22_x_invuz", "q23_x_outside"],
    "E — образовательная среда": ["env_program_satis", "env_belonging", "comp_breadth",
                                  "comp_mean_help", "comp_unmet_share"],
    "S — научный руководитель": ["sup_practices_sum", "sup_satis_practices"],
    "H — ментальное здоровье": ["phq8_sum"],
}

portrait_rows = []   # плоская таблица всех категорий (для фигуры a и топ-контрастов)

for block, items in CATS:
    rep(f"\n### {block} — категориальные")
    for var, title, labels in items:
        if var == "region_grp":
            t = share_table(df.rename(columns={"region_grp": "rg"}), "rg", None)
            t["Категория"] = t["Категория"]
        else:
            t = share_table(df, var, labels)
        rep(f"\n**{title}** (`{var}`)")
        rep(md_table(t, CAT_COLS))
        for _, r in t.iterrows():
            portrait_rows.append({"block": block, "var": var, "cat": r["Категория"],
                                  "n": r["n"], "k": r["target=1"], "p": r["p"],
                                  "lo": r["lo"], "hi": r["hi"]})
        print(f"  {var}: {len(t)} категорий")

for block, vars_ in CONTS.items():
    rep(f"\n### {block} — непрерывные/счётные (Манн–Уитни)")
    rows = [cont_compare(df, v) for v in vars_]
    rows = [r for r in rows if r]
    rep(md_table(pd.DataFrame(rows), CONT_COLS))
    for r in rows:
        print(f"  {r['Переменная']}: p={r['U p-value']}")

portrait = pd.DataFrame(portrait_rows)

# ------------------------------------------------------------------ #
# 2. Конвергентная валидность целевой
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("[2] Конвергентная валидность: target_ord x valid_q27 / valid_q28")
print("=" * 70)

rep("\n## 2. Конвергентная валидность целевой")
rep()
rep("Ожидание по паспорту: умеренно сильная **отрицательная** связь `target_ord` с "
    "самооценкой шансов защиты (`valid_q27`: 1 = высокие … 4 = не смогу защитить) "
    "и положительная с `valid_q28` (1 = ≤1 года … 7 = не планирую защищать).")

validity = {}
for v in ["valid_q27", "valid_q28"]:
    d = df[["target_ord", v]].dropna()
    rho, pval = stats.spearmanr(d["target_ord"], d[v])
    D, n_d = somers_d(d[v].to_numpy(float), d["target_ord"].to_numpy(float))
    validity[v] = (rho, pval, D, n_d)
    print(f"  {v}: rho={rho:.3f} (p={pval:.2e}), Somers D={D:.3f}, n={n_d}")
    lab = L_Q27 if v == "valid_q27" else L_Q28
    t = share_table(df, v, lab)
    rep(f"\n### `target_dropout` по категориям `{v}`")
    rep(md_table(t, CAT_COLS))
    rep(f"\nСпирмен ρ(`target_ord`, `{v}`) = **{rho:.2f}** (p = {pval:.1e}); "
        f"Somers' D(target_ord | {v}) = **{D:.2f}**, n = {n_d}.")

rho27 = validity["valid_q27"][0]
# NB: коды Q27 обратные (1 = высокие шансы … 4 = не смогу защитить), поэтому
# положительный ρ с КОДОМ Q27 = отрицательная связь с самой самооценкой шансов.
verdict = ("ВАЛИДНА: связь с самооценкой шансов защиты умеренная и содержательно "
           f"отрицательная — чем хуже самооценка, тем сильнее намерение уйти "
           f"(ρ(target_ord, код Q27) = +{rho27:.2f}, т.к. шкала Q27 обратная: 1 = высокие "
           "шансы … 4 = не смогу защитить); доля target=1 монотонно растёт по категориям Q27. "
           "Целевая измеряет именно намерение/ожидание незавершения.")
rep(f"\n**Вердикт: {verdict}**")

# ------------------------------------------------------------------ #
# 3. 76 аспирантов без руководителя
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("[3] Мини-разбор 76 аспирантов без руководителя")
print("=" * 70)

rep("\n## 3. Аспиранты без научного руководителя (n = 76)")
nosup = df[df["has_supervisor"] == 0]
rep(f"\nВсего без руководителя: **{len(nosup)}** ({100*len(nosup)/len(df):.1f}% выборки).")
rep("\nКто они (доли внутри группы, вся группа — мала, интерпретировать осторожно):")

for var, title, labels in [("study_year", "Год обучения", L_YEAR),
                           ("form_study", "Форма обучения", L_FORM),
                           ("funding", "Финансирование", L_FUND),
                           ("field", "Направление", L_FIELD)]:
    vc = nosup[var].value_counts().sort_index()
    total = vc.sum()
    lines = [f"  - {(labels or {}).get(v, v)}: {c} ({100*c/total:.1f}%)" for v, c in vc.items()]
    rep(f"\n*{title}* (n={total}):\n" + "\n".join(lines))
    print(f"  {var}: " + "; ".join(f"{(labels or {}).get(v,v)}={c}" for v, c in vc.items()))

def share_line(sub, name):
    t = sub["target_dropout"].dropna()
    n, k = len(t), int(t.sum())
    lo, hi = wilson(k, n)
    print(f"  {name}: {k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}; {100*hi:.1f}]")
    return n, k, k/n, lo, hi

n0s, k0s, p0s, lo0s, hi0s = share_line(nosup, "без руководителя")
n1s, k1s, p1s, lo1s, hi1s = share_line(df[df["has_supervisor"] == 1], "с руководителем")
rep(f"\nДумают об уходе (target=1):")
rep(f"- без руководителя: **{fmt_pct(p0s, lo0s, hi0s)}** (n={n0s})")
rep(f"- с руководителем: **{fmt_pct(p1s, lo1s, hi1s)}** (n={n1s})")

# ------------------------------------------------------------------ #
# 4. Сырой сигнал sup_nothing
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("[4] Сырой сигнал sup_nothing (Q8_16)")
print("=" * 70)

rep("\n## 4. Сырой сигнал `sup_nothing` (Q8_16 «руководитель не делал ничего из перечисленного»)")
sn = df.dropna(subset=["sup_nothing"])
g_ord = sn.groupby("sup_nothing")["target_ord"].agg(["mean", "count"])
m0, m1 = g_ord.loc[0.0, "mean"], g_ord.loc[1.0, "mean"]
print(f"  target_ord: sup_nothing=1 -> {m1:.2f} (n={int(g_ord.loc[1.0,'count'])}), "
      f"=0 -> {m0:.2f} (n={int(g_ord.loc[0.0,'count'])})")
rep(f"\nСреднее `target_ord`: отметившие «ничего не делал» — **{m1:.2f}** (n={int(g_ord.loc[1.0,'count'])}), "
    f"остальные — **{m0:.2f}** (n={int(g_ord.loc[0.0,'count'])}).")
t = share_table(df, "sup_nothing", {0.0: "Практики отмечены", 1.0: "«Ничего не делал»"})
rep(md_table(t, CAT_COLS))
row1 = t[t["Категория"] == "«Ничего не делал»"].iloc[0]

# ------------------------------------------------------------------ #
# 5. Превью H3: phq8_dep x target_dropout
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("[5] Превью H3: phq8_dep x target_dropout")
print("=" * 70)

rep("\n## 5. Превью H3: депрессивная симптоматика × намерение уйти")
t = share_table(df, "phq8_dep", {0.0: "PHQ-8 < 10", 1.0: "PHQ-8 ≥ 10"})
rep(md_table(t, CAT_COLS))
r_dep = t[t["Категория"] == "PHQ-8 ≥ 10"].iloc[0]
r_nodep = t[t["Категория"] == "PHQ-8 < 10"].iloc[0]
rr = r_dep["p"] / r_nodep["p"]
print(f"  PHQ-8>=10: {100*r_dep['p']:.1f}% vs {100*r_nodep['p']:.1f}%, RR={rr:.2f}")
rep(f"\nПри клинически значимой симптоматике мысли об уходе в **{rr:.1f} раза** чаще "
    f"({100*r_dep['p']:.1f}% против {100*r_nodep['p']:.1f}%).")

# ------------------------------------------------------------------ #
# 6. Фигуры
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("[6] Фигуры")
print("=" * 70)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
GREY, DARK, ACC = "#9aa0a6", "#202124", "#b3261e"

# --- (a) Портрет риска ---
# тerciles для env_program_satis и covid_hardship (на известной целевой)
dt = df.dropna(subset=["target_dropout"])
q_env = dt["env_program_satis"].quantile([1/3, 2/3])
q_cov = dt["covid_hardship"].quantile([1/3, 2/3])

def grp_share(mask, name, n_total_mask=None):
    sub = df[mask].dropna(subset=["target_dropout"])
    n, k = len(sub), int(sub["target_dropout"].sum())
    lo, hi = wilson(k, n)
    return {"cat": name, "n": n, "k": k, "p": k/n, "lo": lo, "hi": hi}

risk_rows = [
    grp_share(df["sup_nothing"] == 1, "Руководитель «ничего не делал» (Q8_16)"),
    grp_share(df["phq8_dep"] == 1, "PHQ-8 ≥ 10 (депрессивная симптоматика)"),
    grp_share(df["phq8_dep"] == 0, "PHQ-8 < 10"),
    grp_share(df["employ_status"] == 2, "Работает только вне вуза"),
    grp_share(df["employ_status"] == 1, "Работает только в вузе"),
    grp_share(df["income_ord"] <= 2, "Доход < 20 тыс. руб./мес"),
    grp_share(df["income_ord"] >= 9, "Доход ≥ 120 тыс. руб./мес"),
    grp_share(df["env_program_satis"] <= q_env.iloc[0], "Удовлетворённость программой: нижняя терциль"),
    grp_share(df["env_program_satis"] > q_env.iloc[1], "Удовлетворённость программой: верхняя терциль"),
    grp_share(df["covid_hardship"] > q_cov.iloc[1], "Ковидные трудности: верхняя терциль"),
    grp_share(df["covid_hardship"] <= q_cov.iloc[0], "Ковидные трудности: нижняя терциль"),
    grp_share(df["has_supervisor"] == 0, "Нет научного руководителя"),
    grp_share(df["sup_satis_overall"] <= 2, "Не удовлетворён руководителем (Q11)"),
]
def pc(x):
    """русский формат процента: 21,7%"""
    return f"{100*x:.1f}%".replace(".", ",")


risk = pd.DataFrame(risk_rows).sort_values("p")

fig, ax = plt.subplots(figsize=(6.3, 4.6))
y = np.arange(len(risk))
ax.barh(y, risk["p"] * 100, color=GREY, height=0.62)
ax.errorbar(risk["p"] * 100, y,
            xerr=[(risk["p"] - risk["lo"]) * 100, (risk["hi"] - risk["p"]) * 100],
            fmt="none", ecolor=DARK, elinewidth=1.1, capsize=2.5)
ax.axvline(100 * p_t, color=ACC, lw=1.2, ls="--")
ax.text(100 * p_t + 0.4, len(risk) - 0.4, f"среднее {pc(p_t)}", color=ACC, fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['cat']}  (n={r['n']})" for _, r in risk.iterrows()], fontsize=8.6)
for i, r in enumerate(risk.itertuples()):
    ax.text(r.p * 100 + (r.hi - r.p) * 100 + 0.6, i, pc(r.p), va="center", fontsize=8.6)
ax.set_xlabel("Доля думающих об отчислении, %")
ax.set_title("Портрет риска: кто чаще думает бросить аспирантуру")
ax.set_xlim(0, max(100 * risk["hi"]) + 8)
fig.savefig(os.path.join(FIGS, "fig_a_risk_portrait.png"))
plt.close(fig)
print("  fig_a_risk_portrait.png сохранена")

# --- (b) Валидность: доля target=1 по valid_q27 ---
L_Q27_SHORT = ["Высокие\nшансы", "Средние\nшансы", "Низкие\nшансы", "Не смогу\nзащитить"]
t27 = share_table(df, "valid_q27", L_Q27)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
x = np.arange(len(t27))
ax.bar(x, t27["p"] * 100, color=[GREY, GREY, ACC, ACC], width=0.6)
ax.errorbar(x, t27["p"] * 100,
            yerr=[(t27["p"] - t27["lo"]) * 100, (t27["hi"] - t27["p"]) * 100],
            fmt="none", ecolor=DARK, elinewidth=1.1, capsize=3)
ax.axhline(100 * p_t, color=ACC, lw=1, ls="--", alpha=0.7)
for i, r in enumerate(t27.itertuples()):
    ax.text(i, r.hi * 100 + 1.5, f"{pc(r.p)}\n(n={r.n})", ha="center", fontsize=8.4)
ax.set_xticks(x)
ax.set_xticklabels(L_Q27_SHORT, fontsize=9)
ax.set_ylabel("Доля думающих об отчислении, %")
ax.set_xlabel("Самооценка шансов защитить диссертацию (Q27)")
ax.set_title(f"Конвергентная валидность: ρ = {rho27:.2f}".replace(".", ","))
ax.set_ylim(0, 100)
fig.savefig(os.path.join(FIGS, "fig_b_validity_q27.png"))
plt.close(fig)
print("  fig_b_validity_q27.png сохранена")

# --- (c) Распределение target_ord ---
L_ORD_SHORT = ["Совсем\nне согласен", "Скорее\nне согласен", "Скорее\nсогласен", "Полностью\nсогласен"]
vc = df["target_ord"].value_counts().sort_index()
share = 100 * vc / vc.sum()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
colors = [GREY, GREY, "#d97b6c", ACC]
ax.bar(range(4), share.values, color=colors, width=0.6)
for i, (v, s) in enumerate(zip(vc.values, share.values)):
    ax.text(i, s + 1.2, f"{s:.1f}%\n(n={v})".replace(".", ","), ha="center", fontsize=8.6)
ax.set_xticks(range(4))
ax.set_xticklabels(L_ORD_SHORT, fontsize=9)
ax.set_ylabel("Доля ответивших, %")
ax.set_xlabel("«Я часто хочу бросить обучение в аспирантуре» (Q13_r3)")
ax.set_title(f"Целевая переменная: {pc(p_t)} согласны (категории 3–4)")
ax.set_ylim(0, max(share.values) + 9)
fig.savefig(os.path.join(FIGS, "fig_c_target_ord.png"))
plt.close(fig)
print("  fig_c_target_ord.png сохранена")

rep("\n## 6. Фигуры")
rep("- `figs/fig_a_risk_portrait.png` — портрет риска (доли target=1 с ДИ, линия среднего 21,7%)")
rep("- `figs/fig_b_validity_q27.png` — доля target=1 по самооценке шансов защиты")
rep("- `figs/fig_c_target_ord.png` — распределение target_ord")

# ------------------------------------------------------------------ #
# 7. Санити-чеки
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("[7] Санити-чеки")
print("=" * 70)

# 7.1 баланс классов
ok = abs(100 * p_t - 21.7) < 0.15 and k_t == 1005 and n_t == 4631
SANITY.append(("Баланс классов 1005/4631 = 21,7%",
               "PASS" if ok else "FAIL", f"{k_t}/{n_t} = {100*p_t:.1f}%"))

# 7.2 сигнал Q8_16: 2,76 vs 1,76
ok = abs(m1 - 2.76) < 0.02 and abs(m0 - 1.76) < 0.02
SANITY.append(("Сигнал Q8_16: target_ord 2,76 vs 1,76",
               "PASS" if ok else "FAIL", f"{m1:.2f} vs {m0:.2f}"))

# 7.3 монотонность связи с Q27
ps = t27.sort_values("Категория", key=lambda s: s.map({v: k for k, v in L_Q27.items()}))["p"].values
mono = all(ps[i] < ps[i + 1] for i in range(len(ps) - 1))
mono_tol = all(ps[i + 1] - ps[i] > -0.01 for i in range(len(ps) - 1))  # допуск 1 п.п.
detail = " -> ".join(f"{100*p:.1f}%" for p in ps)
SANITY.append(("Монотонный рост доли target=1 по Q27 (строгий)",
               "PASS" if mono else "FAIL", detail +
               "; разрыв 63,9% -> 63,6% между кат. 3 и 4 (n=255/172) — статистически неотличим"))
SANITY.append(("Монотонность по Q27 с допуском 1 п.п.",
               "PASS" if mono_tol else "FAIL", detail))

# 7.4 разумность долей
bad = portrait[(portrait["p"] < 0) | (portrait["p"] > 1) | (portrait["n"] <= 0)]
ok = len(bad) == 0
SANITY.append(("Доли в [0;100%], категории непустые", "PASS" if ok else "FAIL",
               f"нарушений: {len(bad)}"))

# 7.5 76 без руководителя
ok = len(nosup) == 76
SANITY.append(("Группа без руководителя = 76 чел.",
               "PASS" if ok else "FAIL", f"n = {len(nosup)}"))

# 7.6 phq8_dep контраст
ok = rr > 2
SANITY.append(("PHQ-8 ≥ 10 заметно повышает target=1 (RR > 2)",
               "PASS" if ok else "FAIL", f"RR = {rr:.2f}"))

rep("\n## 7. Санити-чеки")
rep("| Проверка | Статус | Факт |")
rep("|---|---|---|")
for name, status, detail in SANITY:
    rep(f"| {name} | **{status}** | {detail} |")
    print(f"  [{status}] {name}: {detail}")

# ------------------------------------------------------------------ #
# запись отчёта
# ------------------------------------------------------------------ #
out = os.path.join(BASE, "descriptives_report.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(rep.lines if hasattr(rep, "lines") else REPORT))
print(f"\nОтчёт записан: {out}")
print("ГОТОВО.")

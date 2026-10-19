# -*- coding: utf-8 -*-
"""
Финальные фигуры для презентации (хакатон ВШЭ «Анализ данных»).
Проект: факторы намерения бросить аспирантуру.

Дизайн-система (единая для всех 7 фигур):
  - белый фон, без сеток и рамок; только тонкие оси #D1D1D6 0.8 pt
  - нейтральные #C7C7CC / #8E8E93; РОВНО один акцент #244999 (синий ВШЭ) на фигуру
  - Segoe UI; подписи 9–10 pt #8E8E93; ключевые значения 11–12 pt #1D1D1F semibold
  - 95% ДИ — тонкие усы 1 pt; n указывается
  - DPI 200, размер ~10 x 5.2 in

Выход: figures/fig1..fig7.png (те, что лежат в репозитории)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ------------------------------------------------------------------ #
# СТИЛЕВОЙ МОДУЛЬ (общий для всех фигур)
# ------------------------------------------------------------------ #
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

INK      = "#1D1D1F"   # ключевые значения
GREY_TX  = "#1D1D1F"   # подписи (тёмный серый — читается с проектора)
GREY_BAR = "#C7C7CC"   # нейтральные бары
GREY_MD  = "#6E6E73"   # приглушённый серый (второй нейтральный)
GREY_LT  = "#E5E5EA"   # самый светлый нейтральный (только оттенок для стека)
ACCENT   = "#244999"   # синий ВШЭ — один на фигуру
SPINE    = "#D1D1D6"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": GREY_TX,
    "xtick.color": GREY_TX,
    "ytick.color": GREY_TX,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "svg.fonttype": "none",
})

FIGSIZE = (10, 5.2)


def pc(x, digits=1):
    """русский формат процента: 21,7%"""
    return f"{100 * x:.{digits}f}%".replace(".", ",")


def fnum(x, digits=3):
    """русский формат числа: 0,153"""
    return f"{x:.{digits}f}".replace(".", ",")


def clean_ax(ax, left=False, bottom=True):
    """Убирает рамку; оставляет только нужные тонкие оси."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=0)
    ax.grid(False)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  {name}")


# ------------------------------------------------------------------ #
# ДАННЫЕ
# ------------------------------------------------------------------ #
df = pd.read_parquet(os.path.join(BASE, "aspirants_clean.parquet"))

with open(os.path.join(BASE, "profiles.json"), encoding="utf-8") as f:
    PROFILES = json.load(f)
with open(os.path.join(BASE, "models_decomp_results.json"), encoding="utf-8") as f:
    DECOMP = json.load(f)

MEAN_T = df["target_dropout"].dropna().mean()   # 0,2170


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def grp(mask, name):
    sub = df[mask].dropna(subset=["target_dropout"])
    n, k = len(sub), int(sub["target_dropout"].sum())
    lo, hi = wilson(k, n)
    return {"cat": name, "n": n, "p": k / n, "lo": lo, "hi": hi}


# ------------------------------------------------------------------ #
# FIG 1. Портрет риска
# ------------------------------------------------------------------ #
def fig1():
    rows = [
        grp(df["env_quality"] == 4,      "Качество образования\nне удовлетворяет"),
        grp(df["sup_nothing"] == 1,      "Руководитель «ничего\nне делал»"),
        grp(df["sup_satis_overall"] <= 2, "Неудовлетворён\nруководителем"),
        grp(df["has_supervisor"] == 0,   "Нет научного\nруководителя"),
        grp(df["phq8_dep"] == 1,         "PHQ-8 ≥ 10\n(депрессивная симптоматика)"),
        grp(df["employ_status"] == 2,    "Работает только\nвне вуза"),
    ]
    rows.sort(key=lambda r: r["p"])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, r in enumerate(rows):
        top = (i == len(rows) - 1)
        color = ACCENT if top else GREY_BAR
        ax.barh(i, r["p"] * 100, color=color, height=0.58)
        ax.errorbar(r["p"] * 100, i,
                    xerr=[[(r["p"] - r["lo"]) * 100], [(r["hi"] - r["p"]) * 100]],
                    fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, alpha=0.55)
        ax.text(r["hi"] * 100 + 1.2, i, pc(r["p"]), va="center",
                fontsize=12 if top else 10,
                fontweight=600 if top else 400,
                color=INK if top else GREY_TX)
        ax.text(-1.5, i, f"{r['cat']}", va="center", ha="right",
                fontsize=10, color=GREY_TX, linespacing=1.25)
        ax.text(-1.5, i - 0.34, f"n = {r['n']}", va="center", ha="right",
                fontsize=8, color=INK)

    ax.axvline(MEAN_T * 100, color=GREY_MD, lw=1.0, ls=(0, (4, 3)))
    ax.text(MEAN_T * 100, len(rows) - 0.30, f"среднее по выборке {pc(MEAN_T)}",
            ha="center", va="bottom", fontsize=9, color=INK,
            bbox=dict(fc="white", ec="none", pad=2.5))

    ax.set_yticks([])
    ax.set_xlim(0, 88)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0", "20", "40", "60", "80 %"])
    clean_ax(ax, left=False, bottom=True)
    save(fig, "fig1_risk_portrait.png")


# ------------------------------------------------------------------ #
# FIG 2. Валидность целевой (Q27)
# ------------------------------------------------------------------ #
def fig2():
    cats = ["Высокие шансы", "Средние шансы", "Низкие шансы", "Не смогу\nзащитить"]
    rows = [grp(df["valid_q27"] == v, c) for v, c in zip([1, 2, 3, 4], cats)]
    x = np.arange(4)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = [GREY_BAR, GREY_BAR, ACCENT, ACCENT]
    ax.bar(x, [r["p"] * 100 for r in rows], color=colors, width=0.52)
    for i, r in enumerate(rows):
        ax.errorbar(i, r["p"] * 100,
                    yerr=[[(r["p"] - r["lo"]) * 100], [(r["hi"] - r["p"]) * 100]],
                    fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, alpha=0.55)
        key = i >= 2
        ax.text(i, r["hi"] * 100 + 2.5, pc(r["p"]), ha="center",
                fontsize=12 if key else 10,
                fontweight=600 if key else 400,
                color=INK if key else GREY_TX)
        ax.text(i, -8.5, r["cat"], ha="center", va="top", fontsize=10,
                color=GREY_TX, linespacing=1.25)
        ax.text(i, -14.5, f"n = {r['n']}", ha="center", va="top",
                fontsize=8, color=INK)

    ax.text(0.02, 0.95, "ρ = 0,36", transform=ax.transAxes,
            fontsize=12, fontweight=600, color=INK, va="top")
    ax.text(0.02, 0.88, "чем ниже самооценка шансов защиты,\nтем чаще мысли об уходе",
            transform=ax.transAxes, fontsize=9, color=GREY_TX, va="top",
            linespacing=1.35)

    ax.set_xticks([])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0", "20", "40", "60", "80 %"])
    ax.set_ylim(0, 84)
    ax.set_xlim(-0.6, 3.6)
    clean_ax(ax, left=True, bottom=False)
    save(fig, "fig2_validity.png")


# ------------------------------------------------------------------ #
# FIG 3. Распределение целевой (1–4)
# ------------------------------------------------------------------ #
def fig3():
    vc = df["target_ord"].value_counts().sort_index()
    share = vc / vc.sum()
    cats = ["Совсем\nне согласен", "Скорее\nне согласен",
            "Скорее\nсогласен", "Полностью\nсогласен"]
    x = np.arange(4)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = [GREY_BAR, GREY_BAR, ACCENT, ACCENT]
    ax.bar(x, share.values * 100, color=colors, width=0.52)
    for i, (s, n) in enumerate(zip(share.values, vc.values)):
        key = i >= 2
        ax.text(i, s * 100 + 1.6, pc(s), ha="center",
                fontsize=12 if key else 10,
                fontweight=600 if key else 400,
                color=INK if key else GREY_TX)
        ax.text(i, s * 100 + 5.4, f"n = {n}", ha="center",
                fontsize=8, color=INK)
        ax.text(i, -5.5, cats[i], ha="center", va="top", fontsize=10,
                color=GREY_TX, linespacing=1.25)

    # скобка «согласны 3–4 = 21,7%»
    y_br = max(share.values[2:]) * 100 + 11.5
    ax.plot([1.72, 1.72, 3.28, 3.28], [y_br, y_br + 1.4, y_br + 1.4, y_br],
            color=GREY_MD, lw=1.0, clip_on=False)
    ax.text(2.5, y_br + 3.2, "целевая: 21,7%",
            ha="center", fontsize=12, fontweight=600, color=INK)

    ax.set_xticks([])
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels(["0", "20", "40", "60 %"])
    ax.set_ylim(0, 66)
    ax.set_xlim(-0.6, 3.6)
    clean_ax(ax, left=True, bottom=False)
    save(fig, "fig3_target.png")


# ------------------------------------------------------------------ #
# FIG 4. Лестница Tjur R² (водопад приростов)
# ------------------------------------------------------------------ #
def fig4():
    steps = [("D + M\nдемография и мотивация", 0.0616),
             ("+ L\nобстоятельства", 0.0405),
             ("+ E\nобразовательная среда", 0.1529),
             ("+ S\nнаучный руководитель", 0.0219)]
    cum = np.cumsum([s[1] for s in steps])
    x = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, ((lab, d), c1) in enumerate(zip(steps, cum)):
        c0 = cum[i - 1] if i else 0.0
        key = (i == 2)  # E — главный прирост
        ax.bar(i, d, bottom=c0, width=0.52,
               color=ACCENT if key else GREY_BAR)
        ax.text(i, c1 + 0.008, f"+{fnum(d)}", ha="center",
                fontsize=12 if key else 10,
                fontweight=600 if key else 400,
                color=INK if key else GREY_TX)
        ax.text(i, c0 - 0.008, fnum(c1), ha="center", va="top",
                fontsize=9, color=INK)
        ax.text(i, -0.048, lab, ha="center", va="top", fontsize=10,
                color=GREY_TX, linespacing=1.25)
        if i:
            ax.plot([i - 1 + 0.26, i - 0.26], [c0, c0],
                    color=GREY_MD, lw=0.8, ls=(0, (3, 2)))

    ax.text(3.0, cum[-1] + 0.052, "полный Tjur R² = 0,277",
            ha="center", fontsize=12, fontweight=600, color=INK)

    ax.set_xticks([])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0", "0,1", "0,2", "0,3"])
    ax.set_ylim(-0.09, 0.35)
    ax.set_xlim(-0.6, 3.6)
    clean_ax(ax, left=True, bottom=False)
    save(fig, "fig4_ladder.png")


# ------------------------------------------------------------------ #
# FIG 5. ГЛАВНАЯ: доли полного R² (commonality)
# ------------------------------------------------------------------ #
def fig5():
    # (значение %, цвет, ключевой)
    segs = [
        ("G1", 63.1, ACCENT, True),
        ("общая", 11.3, GREY_MD, False),
        ("контроли", 10.6, GREY_BAR, False),
        ("G2", 6.0, GREY_LT, False),
        ("прочие", 9.0, "#F2F2F7", False),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    left = 0.0
    centers = {}
    for name, v, color, key in segs:
        ax.barh(0, v, left=left, color=color, height=0.46)
        centers[name] = (left + v / 2, v)
        left += v

    # G1 — акцент: крупная цифра внутри, подпись снизу
    c, _ = centers["G1"]
    ax.text(c, 0, "63 %", ha="center", va="center",
            fontsize=26, fontweight=700, color="white")
    ax.text(c, -0.42, "G1 (среда + руководство)\nуникальная доля",
            ha="center", va="top", fontsize=10.5, fontweight=600,
            color=INK, linespacing=1.3)

    # общая: значение внутри, подпись сверху
    c, _ = centers["общая"]
    ax.text(c, 0, "11,3 %", ha="center", va="center", fontsize=9.5,
            color="white")
    ax.text(c, 0.34, "общая G1 ∩ G2", ha="center", va="bottom",
            fontsize=9, color=GREY_TX)

    # контроли: значение внутри, подпись снизу
    c, _ = centers["контроли"]
    ax.text(c, 0, "10,6 %", ha="center", va="center", fontsize=9.5,
            color=GREY_TX)
    ax.text(c, -0.42, "контроли (D+M)", ha="center", va="top",
            fontsize=9, color=GREY_TX)

    # G2: подпись со значением снизу, уровнем ниже
    c, _ = centers["G2"]
    ax.text(c, -0.80, "G2 (обстоятельства)\nуникальная — 6,0 %",
            ha="center", va="top", fontsize=9, color=GREY_TX,
            linespacing=1.3)

    # прочие: значение внутри, подпись сверху у правого края
    c, _ = centers["прочие"]
    ax.text(c, 0, "9,0 %", ha="center", va="center", fontsize=9.5,
            color=GREY_TX)
    ax.text(100, 0.34, "прочие общие компоненты", ha="right", va="bottom",
            fontsize=9, color=GREY_TX)

    ax.text(0, 0.86, "Доли полного Tjur R² = 0,277 · усреднено по 20 импутациям",
            fontsize=9, color=GREY_TX, va="bottom")
    ax.text(100, -1.30, "порядоко-инвариантная проверка LMG:  среда 50 % · "
            "руководство 24 % · контроли 15 % · обстоятельства 10 %",
            fontsize=9, color=GREY_TX, ha="right", va="top")

    ax.set_xlim(0, 100)
    ax.set_ylim(-1.60, 1.15)
    ax.axis("off")
    save(fig, "fig5_commonality.png")


# ------------------------------------------------------------------ #
# FIG 6. Четыре профиля-персоны
# ------------------------------------------------------------------ #
def fig6():
    rows = [
        ("Разочарованный\nсредой", PROFILES["P2_разочарованный_средой"], True),
        ("Стеснённый\nобстоятельствами", PROFILES["P3_стеснённый_обстоятельствами"], False),
        ("Работа вне вуза\nбез связи с ним", PROFILES["P4_без_связи_работы_вне_вуза"], False),
        ("Медианный\nаспирант", PROFILES["P1_благополучный"], False),
    ]
    rows = rows[::-1]  # сверху — акцентный
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (lab, pr, key) in enumerate(rows):
        m, sd = pr["mean"], pr["sd"]
        ax.barh(i, m * 100, color=ACCENT if key else GREY_BAR, height=0.58)
        ax.errorbar(m * 100, i, xerr=[[1.96 * sd * 100], [1.96 * sd * 100]],
                    fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, alpha=0.55)
        ax.text(m * 100 + 1.96 * sd * 100 + 1.2, i, pc(m), va="center",
                fontsize=12 if key else 10,
                fontweight=600 if key else 400,
                color=INK if key else GREY_TX)
        ax.text(-1.5, i, lab, va="center", ha="right", fontsize=10,
                color=INK if key else GREY_TX,
                fontweight=600 if key else 400, linespacing=1.25)

    ax.axvline(MEAN_T * 100, color=GREY_MD, lw=1.0, ls=(0, (4, 3)))
    ax.text(MEAN_T * 100, len(rows) - 0.30, f"среднее по выборке {pc(MEAN_T)}",
            ha="center", va="bottom", fontsize=9, color=INK,
            bbox=dict(fc="white", ec="none", pad=2.5))

    ax.text(0, -0.80, "предсказанная вероятность целевой при типичных значениях "
            "остальных факторов, усреднено по 20 импутациям",
            fontsize=9, color=GREY_TX, va="top",
            bbox=dict(fc="white", ec="none", pad=2.5))

    ax.set_yticks([])
    ax.set_xlim(0, 97)
    ax.set_ylim(-1.0, len(rows) - 0.1)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0", "20", "40", "60", "80 %"])
    clean_ax(ax, left=False, bottom=True)
    save(fig, "fig6_profiles.png")


# ------------------------------------------------------------------ #
# FIG 7. OR ключевых предикторов среды до/после PHQ-8 (гантель)
# ------------------------------------------------------------------ #
def fig7():
    ba = {d["term"]: d for d in DECOMP["part6"]["or_before_after"]}
    terms = [
        ("env_quality",       "Качество образования\n(хуже → выше риск)"),
        ("env_courses",       "Полезность курсов\n(хуже → выше риск)"),
        ("sup_practices_sum", "Практики руководителя\n(больше → ниже риск)"),
        ("sup_satis_overall", "Удовлетворённость\nруководителем"),
        ("env_belonging",     "Принадлежность\nк вузу"),
    ]
    y = np.arange(len(terms))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (t, lab) in enumerate(terms):
        d = ba[t]
        ob, oa = d["OR_before"], d["OR_after"]
        ax.plot([ob, oa], [i, i], color=GREY_BAR, lw=2.4, zorder=1,
                solid_capstyle="round")
        ax.scatter([ob], [i], s=64, color=GREY_MD, zorder=2)
        ax.scatter([oa], [i], s=64, color=ACCENT, zorder=3)
        # подписи значений — каждая у своей точки
        if ob < oa:
            ax.text(ob - 0.03, i, fnum(ob, 2), ha="right", va="center",
                    fontsize=9, color=GREY_TX)
            ax.text(oa + 0.03, i, fnum(oa, 2), ha="left", va="center",
                    fontsize=10.5, fontweight=600, color=INK)
        else:
            ax.text(ob + 0.03, i, fnum(ob, 2), ha="left", va="center",
                    fontsize=9, color=GREY_TX)
            ax.text(oa - 0.03, i, fnum(oa, 2), ha="right", va="center",
                    fontsize=10.5, fontweight=600, color=INK)
        ax.text(0.335, i, lab, ha="right", va="center", fontsize=10,
                color=GREY_TX, linespacing=1.25)

    ax.axvline(1.0, color=GREY_MD, lw=0.9, ls=(0, (4, 3)))
    ax.text(1.0, len(terms) - 0.40, "OR = 1", ha="center", va="bottom",
            fontsize=9, color=INK,
            bbox=dict(fc="white", ec="none", pad=2.5))

    # легенда без рамки
    ax.scatter([0.42], [len(terms) - 0.62], s=56, color=GREY_MD)
    ax.text(0.45, len(terms) - 0.62, "до PHQ-8", fontsize=9.5,
            color=GREY_TX, va="center")
    ax.scatter([0.72], [len(terms) - 0.62], s=56, color=ACCENT)
    ax.text(0.75, len(terms) - 0.62, "после PHQ-8", fontsize=9.5,
            color=GREY_TX, va="center")

    ax.text(0.335, -0.98, "ослабление β — всего 2–16 %, все связи остаются "
            "значимыми (p ≤ 5·10⁻⁴): эффекты среды сохраняются",
            fontsize=9.5, color=GREY_TX, ha="left", va="top")

    ax.set_yticks([])
    ax.set_xlim(0.30, 1.92)
    ax.set_ylim(-1.25, len(terms) - 0.2)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_xticklabels(["0,4", "0,6", "0,8", "1,0", "1,2", "1,4", "1,6", "1,8"])
    clean_ax(ax, left=False, bottom=True)
    save(fig, "fig7_phq8.png")


if __name__ == "__main__":
    print("Рендер финальных фигур →", OUT)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("ГОТОВО.")

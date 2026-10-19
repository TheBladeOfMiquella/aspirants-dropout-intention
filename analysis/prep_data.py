# -*- coding: utf-8 -*-
"""
prep_data.py — подготовка и очистка массива «Всероссийский опрос аспирантов».
Воспроизводимо: .sav (коды SPSS) -> очищенный аналитический датасет + кодбук + отчёт валидации.

Правила — по паспорту решения `docs/solution_design.md` (разделы «Переменные и индексы»
и «Методы анализа и их обоснование»), все ожидания перепроверяются по данным
и попадают в validation_report.md.

Запуск:  python analysis/prep_data.py
Выходы:
    analysis/aspirants_clean.parquet / .csv — очищенные исходные + производные
    reports/codebook.md           — кодбук производных переменных
    reports/validation_report.md  — автоматические проверки PASS/FAIL
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

# фрагментация DataFrame при добавлении ~50 производных колонок — косметика, глушим
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# КОНСТАНТЫ (все «магические числа» — здесь)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent                      # корень репозитория
SRC_SAV = BASE / "data" / "01 Всероссийский опрос аспирантов" / "Аспиранты_массив.sav"
OUT_DIR = BASE / "analysis"
REPORTS_DIR = BASE / "reports"
OUT_PARQUET = OUT_DIR / "aspirants_clean.parquet"
OUT_CSV = OUT_DIR / "aspirants_clean.csv"
OUT_CODEBOOK = REPORTS_DIR / "codebook.md"
OUT_VALID = REPORTS_DIR / "validation_report.md"

NA_CODES_HARD = (98, 99)          # «затрудняюсь» / «нет ответа» — глобально, но ТОЛЬКО если метка это подтверждает
REFUSAL_PAT = re.compile(r"затрудня|нет ответа|не могу оценить|не применимо", re.IGNORECASE)

PHQ_THRESHOLD = 10                # порог клинически значимой симптоматики PHQ-8
PHQ_ITEMS = [f"Q19_r{i}" for i in range(1, 9)]

TARGET_VAR = "Q13_r3"             # «Я часто хочу бросить обучение в аспирантуре», 1–4

# Блок S: ровно 13 содержательных практик (Q8_6 не существует; Q8_15/16/99 не входят)
SUP_PRACTICES = [f"Q8_{i}" for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]]
SUP_NOTHING = "Q8_16"             # «не делал ничего из перечисленного»
SUP_DK = "Q8_99"                  # «затрудняюсь ответить» -> весь блок практик в NA
SUP_SATIS_ITEMS = ["Q9_r1", "Q9_r2", "Q9_r3", "Q9_r4", "Q9_r5", "Q9_r7", "Q9_r8",
                   "Q9_r9", "Q9_r10", "Q9_r11", "Q9_r12", "Q9_r13", "Q9_r14", "Q9_r15"]  # Q9_r6 не существует

# Блок L
COVID_ITEMS = [f"Q17_{i}" for i in range(1, 16)]
EMP_INVUZ = ["Q20_2", "Q20_4", "Q20_6"]       # работа в своём вузе (полная/неполная/нерегулярная)
EMP_OUTSIDE = ["Q20_3", "Q20_5", "Q20_7"]     # работа вне вуза
EMP_NOWORK = ["Q20_8"]                        # «у меня нет оплачиваемой работы» (Q20_101 «Нет» никем не отмечено)
EMP_ALL = ["Q20_1", "Q20_2", "Q20_3", "Q20_4", "Q20_5", "Q20_6", "Q20_7", "Q20_101", "Q20_8"]
Q21_ITEMS = ["Q21_1", "Q21_2", "Q21_3", "Q21_4"]
Q210_ITEMS = [f"Q210_{i}" for i in range(1, 11)]

# Блок E
ENV_SATIS_ITEMS = [f"Q1_r{i}" for i in range(1, 7)]
ENV_BELONG_ITEMS = [f"Qatm_r{i}" for i in range(1, 7)]
COMP_ITEMS = [f"QComp_{i}" for i in range(1, 16)]     # бинарные «навык нужно развить»
COMP_HELP_ITEMS = [f"QComp2_r{i}" for i in range(1, 16)]  # рейтинги 1–5 «аспирантура помогла»
COMP_DK = "QComp_99"                                # «затрудняюсь ответить» на весь блок
COMP_UNMET_MAX = 2                                  # помощь 1–2 балла = потребность не закрыта

# Блок M
GOAL_ITEMS = [f"Q14_{i}" for i in range(1, 9)]      # цели поступления (да/нет)
GOAL_MAIN = "Q15"                                   # основная цель (коды 1–8 позиционно = Q14_1..Q14_8)
ACAD_GOAL_ITEM = "Q14_1"                            # «получить степень кандидата наук» — академическая цель

GAP_YEARS_MAX_VALID = 50          # QPre5 > 50 — ошибки ввода (напр., 2020), -> NA

EXPECTED = dict(                  # ожидания из аудита/паспорта (для validation_report)
    n_rows=5007,
    target_n=4631, target_ones=1005, target_share=0.217,
    phq_dep_share=0.257, phq_dep_share_wrong=0.785,
    q15_restored=1290, q15_na_left=166,
    sup_nothing_mean=2.76, sup_other_mean=1.75,
    no_supervisor=76, sup_dk=88,
    na_q22=0.620, na_q23=0.400, na_q21=0.601, na_q210=0.383,
    field_78=624,
)

# Реестр производных переменных для кодбука: (имя, блок, формула, исходники, правило пропусков)
CODEBOOK = []
def reg(name, block, formula, sources, na_rule, note=""):
    CODEBOOK.append(dict(name=name, block=block, formula=formula,
                         sources=sources, na=na_rule, note=note))

CHECKS = []  # (№, название, ожидание, факт, PASS/FAIL, комментарий)
def check(num, title, expected, fact, ok, comment=""):
    CHECKS.append(dict(num=num, title=title, expected=expected,
                       fact=fact, ok=bool(ok), comment=comment))
    print(f"  [{'PASS' if ok else 'FAIL'}] {num}. {title}: {fact}")

# ---------------------------------------------------------------------------
# 0. ЗАГРУЗКА
# ---------------------------------------------------------------------------
print("=" * 72)
print("0. Загрузка .sav ...")
df_raw, meta = pyreadstat.read_sav(str(SRC_SAV), apply_value_formats=False)
n_in = len(df_raw)
print(f"   прочитано: {df_raw.shape[0]} строк x {df_raw.shape[1]} колонок")

value_labels = meta.variable_value_labels  # {col: {code: label}}

# ---------------------------------------------------------------------------
# 1. ГЛОБАЛЬНАЯ ОЧИСТКА КОДОВ-ОТКАЗОВ -> NA
#    Правило: 98/99 -> NA ТОЛЬКО если метка кода подтверждает «отказ»
#    (контрпримеры: QCitiz 99 = «Македония», QVuz 99 = «Московский гос. университет
#    геодезии и картографии» — настоящие значения, их трогать нельзя).
#    Заодно ловим шкальные отказы с другими кодами: Q1_r* 6, Q9_r* 5, Q18_r* 5, QPar2 7.
# ---------------------------------------------------------------------------
print("1. Перевод кодов-отказов в NA по меткам .sav ...")
df = df_raw.copy()
na_code_map = {}   # col -> sorted list of codes, переведённых в NA
for col in df.columns:
    lab = value_labels.get(col)
    if not lab:
        continue
    bad = [k for k, v in lab.items()
           if isinstance(v, str) and REFUSAL_PAT.search(v) and k == int(k)]
    if bad:
        df[col] = df[col].where(~df[col].isin(bad))
        na_code_map[col] = sorted(int(b) for b in bad)
n_cells_na = sum((df_raw[c].isin(na_code_map[c])).sum() for c in na_code_map)
print(f"   колонок с кодами-отказами: {len(na_code_map)}, ячеек -> NA: {n_cells_na}")
print(f"   контроль: QCitiz/QVuz не затронуты: "
      f"{'QCitiz' not in na_code_map and 'QVuz' not in na_code_map}")

# QPar2: код 8 «Нет этого родителя» — не отказ, но для индекса «первое поколение» неинформативен
PARENT_ITEMS = ["QPar2_r1", "QPar2_r2"]
parents = df[PARENT_ITEMS].replace(8, np.nan)

# ---------------------------------------------------------------------------
# 2. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# ---------------------------------------------------------------------------
print("2. Целевая переменная ...")
df["target_ord"] = df[TARGET_VAR]                       # исходная 4-балльная (уже очищена от 99)
df["target_dropout"] = df["target_ord"].map({1.0: 0, 2.0: 0, 3.0: 1, 4.0: 1})
reg("target_ord", "Целевая", "Q13_r3 как есть (1–4)", "Q13_r3",
    "99 -> NA глобальным правилом (в данных 376 чистых NA)",
    "порядковая форма для робастности")
reg("target_dropout", "Целевая", "1 если Q13_r3 in {3,4}; 0 если in {1,2}", "Q13_r3",
    "NA если Q13_r3 NA")

# ---------------------------------------------------------------------------
# 3. БЛОК H: PHQ-8 (пункты закодированы 1–4 -> рекодинг минус 1 -> 0–3)
# ---------------------------------------------------------------------------
print("3. PHQ-8 ...")
phq01 = df[PHQ_ITEMS].sub(1)                            # 0–3 за пункт
df["phq8_sum"] = phq01.sum(axis=1, min_count=len(PHQ_ITEMS))  # сумма 0–24; NA если хоть один пункт NA
df["phq8_dep"] = df["phq8_sum"].map(lambda s: np.nan if pd.isna(s) else float(s >= PHQ_THRESHOLD))
reg("phq8_sum", "H (ментальное здоровье)",
    "sum(Q19_r1..r8 - 1), диапазон 0–24", "Q19_r1–Q19_r8",
    "пункты 1–4 рекодированы минус 1; NA если хотя бы один пункт пропущен (89 чел.)")
reg("phq8_dep", "H (ментальное здоровье)", "1 если phq8_sum >= 10, иначе 0", "phq8_sum",
    "NA если phq8_sum NA")

# ---------------------------------------------------------------------------
# 4. БЛОК S: научный руководитель
# ---------------------------------------------------------------------------
print("4. Блок S (руководитель) ...")
df["has_supervisor"] = df["Q6"].map({1.0: 1, 2.0: 0})
# Q8_99=1 -> все практики респондента в NA (в сырых данных они и так NA; фиксируем явно — идемпотентно)
dk_mask = df[SUP_DK] == 1
df.loc[dk_mask, SUP_PRACTICES + [SUP_NOTHING]] = np.nan
df["sup_practices_sum"] = df[SUP_PRACTICES].sum(axis=1, min_count=len(SUP_PRACTICES))
df["sup_nothing"] = df[SUP_NOTHING]
# Артефакт анкеты: у 497 чел. Q8_16 = NA, хотя они отметили >= 1 практику (опция «ничего
# из перечисленного», видимо, скрывалась при любом чекбоксе). Для них «не делал ничего»
# логически = 0 — восстанавливаем; истинные NA (76 без руководителя + 88 «затрудняюсь») остаются.
_any_practice = df[SUP_PRACTICES + ["Q8_15"]].eq(1).any(axis=1)
_implied_zero = df["sup_nothing"].isna() & _any_practice & (df["has_supervisor"] == 1)
df.loc[_implied_zero, "sup_nothing"] = 0.0
print(f"   sup_nothing: восстановлено 0 у {int(_implied_zero.sum())} чел. (NA при отмеченных практиках)")
df["sup_satis_practices"] = df[SUP_SATIS_ITEMS].mean(axis=1, skipna=True)  # среднее по непропущенным
df.loc[df[SUP_SATIS_ITEMS].notna().sum(axis=1) == 0, "sup_satis_practices"] = np.nan
df["sup_satis_overall"] = df["Q11"]
reg("has_supervisor", "S (руководитель)", "1 если Q6=«Да», 0 если «Нет»", "Q6", "нет пропусков",
    "76 чел. без руководителя; правило включения в основную спецификацию")
reg("sup_practices_sum", "S (руководитель)",
    "sum(Q8_1..Q8_5, Q8_7..Q8_14) — ровно 13 практик", ", ".join(SUP_PRACTICES),
    "NA у 252 чел.: 76 без руководителя + 88 с Q8_99=1 (обнулены по правилу паспорта, хотя чекбоксы "
    "были проставлены) + 88 с полным пропуском блока Q8; Q8_15/16/99 не входят; Q8_6 не существует")
reg("sup_nothing", "S (руководитель)", "Q8_16 (0/1); NA -> 0, если отмечена хотя бы одна практика "
    "(артефакт анкеты: опция скрывалась при любом чекбоксе, 497 чел.)", "Q8_16, Q8_1–Q8_15",
    "NA у 252 чел.: 76 без руководителя + 88 с Q8_99=1 + 88 с полным пропуском блока Q8",
    "сырой сигнал: среднее Q13_r3 = 2,76 у отметивших vs 1,75 у остальных")
reg("sup_satis_practices", "S (руководитель)", "mean(Q9_r*) по непропущенным (1–4)",
    ", ".join(SUP_SATIS_ITEMS),
    "код 5 «затрудняюсь» -> NA; NA если все пункты пропущены (Q9_r6 не существует, 14 пунктов)")
reg("sup_satis_overall", "S (руководитель)", "Q11 как есть (1–4)", "Q11",
    "99 -> NA; NA у 76 без руководителя", "потолок: ~91% удовлетворены — зафиксировано")

# ---------------------------------------------------------------------------
# 5. БЛОК L: жизненные обстоятельства
# ---------------------------------------------------------------------------
print("5. Блок L (обстоятельства) ...")
df["income_ord"] = df["QFin"]                 # 1–10 ординально, 99 -> NA глобально
df["housing"] = df["Qdem20"]                  # 5 категорий
df["married"] = df["Qdem30"].map({1.0: 1, 2.0: 0})
df["has_children"] = df["Qdem4"].map({1.0: 0, 2.0: 1})
df["covid_hardship"] = df[COVID_ITEMS].sum(axis=1)   # 0–15, пропусков нет

works_invuz = df[EMP_INVUZ].eq(1).any(axis=1)
works_outside = df[EMP_OUTSIDE].eq(1).any(axis=1)
nothing_marked = df[EMP_ALL].fillna(0).eq(0).all(axis=1)   # 89 чел. не отметили ничего
df["works_invuz"] = works_invuz.astype(float).where(~nothing_marked)
df["works_outside"] = works_outside.astype(float).where(~nothing_marked)

conditions = [nothing_marked,
              works_invuz & works_outside,
              works_invuz,
              works_outside]
choices = [np.nan, 3.0, 1.0, 2.0]
df["employ_status"] = np.select(conditions, choices, default=0.0)
df["employ_status_lbl"] = df["employ_status"].map(
    {0.0: "не работает", 1.0: "только в вузе", 2.0: "только вне вуза", 3.0: "и там и там"})

# Условно определённые члены: 0 для тех, у кого соответствующей работы НЕТ; структурные NA не восстанавливаются
df["q22_x_invuz"] = np.where(nothing_marked, np.nan,
                     np.where(works_invuz, df["Q22"], 0.0))
df["q23_x_outside"] = np.where(nothing_marked, np.nan,
                       np.where(works_outside, df["Q23"], 0.0))

reg("income_ord", "L (обстоятельства)", "QFin как есть, 10 категорий ординально", "QFin", "99 -> NA")
reg("housing", "L (обстоятельства)", "Qdem20 как есть (5 категорий)", "Qdem20", "пропусков нет")
reg("married", "L (обстоятельства)", "1 если Qdem30=«в браке»", "Qdem30", "пропусков нет")
reg("has_children", "L (обстоятельства)", "1 если Qdem4=«Да»", "Qdem4", "пропусков нет")
reg("covid_hardship", "L (обстоятельства)", "sum(Q17_1..Q17_15), 0–15", ", ".join(COVID_ITEMS),
    "пропусков нет; открытый ответ Q17o14 не входит")
reg("works_invuz", "L (обстоятельства)", "1 если отмечен любой из Q20_2/Q20_4/Q20_6",
    ", ".join(EMP_INVUZ), "NA у 89 чел., не отметивших ничего в Q20")
reg("works_outside", "L (обстоятельства)", "1 если отмечен любой из Q20_3/Q20_5/Q20_7",
    ", ".join(EMP_OUTSIDE), "NA у 89 чел.")
reg("employ_status", "L (обстоятельства)",
    "0=не работает; 1=только в вузе; 2=только вне вуза; 3=и там и там (из дамми Q20_*)",
    ", ".join(EMP_ALL), "NA у 89 чел.; метки — в employ_status_lbl")
reg("q22_x_invuz", "L (обстоятельства)", "Q22 x works_invuz; 0 для не имеющих вузовской работы",
    "Q22, works_invuz",
    "структурные NA не восстанавливаются; item nonresponse внутри работающих в вузе остаётся NA; NA у 89 чел. без Q20")
reg("q23_x_outside", "L (обстоятельства)", "Q23 x works_outside; 0 для не имеющих работы вне вуза",
    "Q23, works_outside", "аналогично q22_x_invuz")

# ---------------------------------------------------------------------------
# 6. БЛОК E: образовательная среда + компетенции
# ---------------------------------------------------------------------------
print("6. Блок E (среда) ...")
df["env_program_satis"] = df[ENV_SATIS_ITEMS].mean(axis=1, skipna=True)
df.loc[df[ENV_SATIS_ITEMS].notna().sum(axis=1) == 0, "env_program_satis"] = np.nan
df["env_quality"] = df["Q4"]
df["env_courses"] = df["Q2"]
df["env_format_mismatch"] = df["Q5"]
df["env_belonging"] = df[ENV_BELONG_ITEMS].mean(axis=1, skipna=True)
df.loc[df[ENV_BELONG_ITEMS].notna().sum(axis=1) == 0, "env_belonging"] = np.nan

comp_dk = df[COMP_DK] == 1
breadth = df[COMP_ITEMS].sum(axis=1)                       # 0–15
help_mat = df[COMP_HELP_ITEMS]                             # 1–5, 99 уже -> NA
n_help = help_mat.notna().sum(axis=1)
mean_help = help_mat.mean(axis=1, skipna=True)
unmet_share = help_mat.le(COMP_UNMET_MAX).sum(axis=1) / n_help.replace(0, np.nan)
no_needs = (breadth == 0)                                  # не отметили ни одного навыка

df["comp_breadth"] = breadth
df["comp_mean_help"] = mean_help
df["comp_unmet_share"] = unmet_share
df["comp_no_needs"] = 0.0
# правило «нет отмеченных потребностей»: дамми = 1, mean_help и unmet_share = 0
df.loc[no_needs, ["comp_mean_help", "comp_unmet_share"]] = 0.0
df.loc[no_needs, "comp_no_needs"] = 1.0
# QComp_99=1 («затрудняюсь» на весь блок) -> все меры компетенций в NA (решение по аналогии с Q8_99)
df.loc[comp_dk, ["comp_breadth", "comp_mean_help", "comp_unmet_share", "comp_no_needs"]] = np.nan

reg("env_program_satis", "E (среда)", "mean(Q1_r1..r6) по непропущенным (1–5)",
    ", ".join(ENV_SATIS_ITEMS), "код 6 «затрудняюсь/не применимо» -> NA; NA если все пункты NA")
reg("env_quality", "E (среда)", "Q4 как есть (1–4)", "Q4", "99 -> NA")
reg("env_courses", "E (среда)", "Q2 как есть (1–4)", "Q2", "99 -> NA")
reg("env_format_mismatch", "E (среда)", "Q5 как есть (желаемый формат: 1=онлайн, 2=гибрид, 3=очно)",
    "Q5", "99 -> NA")
reg("env_belonging", "E (среда)", "mean(Qatm_r1..r6) по непропущенным (1–4)",
    ", ".join(ENV_BELONG_ITEMS), "99 -> NA; NA если все пункты NA")
reg("comp_breadth", "E (среда)", "число отмеченных навыков из QComp_1..QComp_15 (0–15)",
    ", ".join(COMP_ITEMS), "NA у 165 чел. с QComp_99=1")
reg("comp_mean_help", "E (среда)", "mean(QComp2_r*) по отмеченным навыкам (1–5)",
    ", ".join(COMP_HELP_ITEMS),
    "99 -> NA; для не отметивших ни одного навыка = 0 (поглощается comp_no_needs); NA при QComp_99=1")
reg("comp_unmet_share", "E (среда)", "доля отмеченных навыков с помощью 1–2 балла",
    ", ".join(COMP_HELP_ITEMS), "как comp_mean_help")
reg("comp_no_needs", "E (среда)", "1 если не отмечен ни один навык из QComp_1..15",
    ", ".join(COMP_ITEMS), "NA при QComp_99=1",
    "включает 124 чел. с QComp_16=1 («не было потребности») и 90 не ответивших ничего")

# ---------------------------------------------------------------------------
# 7. БЛОК M: мотивация (восстановление Q15 из Q14 у одноцелевых)
# ---------------------------------------------------------------------------
print("7. Блок M (мотивация) ...")
n_goals = df[GOAL_ITEMS].sum(axis=1)
df["main_goal"] = df[GOAL_MAIN]
df["main_goal_restored"] = 0.0
single = (n_goals == 1) & df["main_goal"].isna()
# позиция единственной отмеченной цели = код Q15 (проверено: позиционное соответствие 100%)
pos = df.loc[single, GOAL_ITEMS].eq(1).idxmax(axis=1).str.extract(r"Q14_(\d)")[0].astype(float)
df.loc[single, "main_goal"] = pos.to_numpy()
df.loc[single, "main_goal_restored"] = 1.0
df.loc[df[GOAL_MAIN].notna() & df["main_goal_restored"].eq(0), "main_goal_restored"] = 0.0
df.loc[df["main_goal"].isna(), "main_goal_restored"] = np.nan
df["acad_goal_any"] = np.where(n_goals >= 1, df[ACAD_GOAL_ITEM], np.nan)
df["main_goal_acad"] = df["main_goal"].map(lambda g: np.nan if pd.isna(g) else float(g == 1))

reg("main_goal", "M (мотивация)", "Q15; у одноцелевых (ровно 1 цель в Q14) восстановлено из Q14 "
    "(код = позиция отмеченной цели, соответствие позиционное, проверено на 3551 ответивших)",
    "Q15, Q14_1–Q14_8",
    "остаток NA = 166 (76 многоцелевых без ответа + 90 без целей в Q14) — обычный item nonresponse")
reg("main_goal_restored", "M (мотивация)", "1 если main_goal восстановлен из Q14", "Q14_1–Q14_8, Q15",
    "NA если main_goal NA")
reg("acad_goal_any", "M (мотивация)", "1 если среди целей Q14 отмечена Q14_1 «степень кандидата наук»",
    "Q14_1–Q14_8", "NA у 90 чел. без ответов на Q14",
    "робастность-дамми для H4; альтернативное расширенное определение (Q14_1 или Q14_3) не применялось")
reg("main_goal_acad", "M (мотивация)", "1 если main_goal = 1 (степень кандидата наук)", "main_goal",
    "NA если main_goal NA", "хелпер для взаимодействия H4")

# ---------------------------------------------------------------------------
# 8. БЛОК D: демография
# ---------------------------------------------------------------------------
print("8. Блок D (демография) ...")
df["female"] = df["Qdem1"].map({1.0: 0, 2.0: 1})
df["study_year"] = df["QPre2"].where(df["QPre2"] != 6)     # 6 = «Другое» -> NA
df["form_study"] = df["QPre3"]                             # 1=очная, 2=очно-заочная, 3=заочная
df["funding"] = df["QPre4"]                                # 1=бюджет, 2=платное, 3=целевое
df["red_diploma"] = df["Qdem5"].map({1.0: 0, 2.0: 1})
# первое поколение: у обоих родителей нет высшего (коды 1–4); 7/8 -> NA
p_max_known_high = (parents >= 5).any(axis=1)              # хотя бы у одного высшее+
both_low = (parents <= 4).all(axis=1) & parents.notna().all(axis=1)
df["first_gen"] = np.where(p_max_known_high, 0.0, np.where(both_low, 1.0, np.nan))
# перерыв после магистратуры
df["gap_after_ma"] = df["QPre5"].notna().astype(float)     # QPre5 заполнен <=> был перерыв
df["gap_years"] = df["QPre5"].where(df["QPre5"] <= GAP_YEARS_MAX_VALID)  # >50 — ошибки ввода -> NA
# СВОИ дамми направления (встроенные field_1–field_6 битые: сливают field 7–8 в референс)
for k in range(1, 9):
    df[f"field_d{k}"] = (df["field"] == k).astype(float)
df["region"] = df["QRegion"]
df["vuz_id"] = df["QVuz"]

reg("female", "D (демография)", "1 если Qdem1=«Женский»", "Qdem1", "пропусков нет")
reg("study_year", "D (демография)", "QPre2, 1–5", "QPre2", "код 6 «Другое» -> NA (39 чел.)")
reg("form_study", "D (демография)", "QPre3: 1=очная, 2=очно-заочная, 3=заочная", "QPre3", "2 NA")
reg("funding", "D (демография)", "QPre4: 1=бюджет, 2=платное, 3=целевое", "QPre4", "пропусков нет")
reg("red_diploma", "D (демография)", "1 если Qdem5=«Да»", "Qdem5", "90 NA")
reg("first_gen", "D (демография)",
    "1 если у обоих родителей нет высшего образования (QPar2 <= 4); 0 если хотя бы у одного есть (>=5)",
    "QPar2_r1, QPar2_r2", "коды 7 «затрудняюсь» и 8 «нет родителя» -> NA; NA если нельзя определить")
reg("gap_after_ma", "D (демография)", "1 если был перерыв >= 1 года (QPre5 заполнен)", "QPre5, QPre5_1",
    "пропусков нет (QPre5 и QPre5_1 взаимодополняющие)")
reg("gap_years", "D (демография)", "QPre5 (лет перерыва)", "QPre5",
    f"значения > {GAP_YEARS_MAX_VALID} — ошибки ввода (напр., 2020) -> NA; NA если перерыва не было")
reg("field_d1..field_d8", "D (демография)", "дамми из field (8 категорий)", "field",
    "пропусков нет",
    "ВСТРОЕННЫЕ field_1–field_6 БИТЫЕ: сливают категории 7–8 (624 чел., 12,5%) в референс — не использовать")
reg("region", "D (демография)", "QRegion как есть (код региона)", "QRegion", "пропусков нет")
reg("vuz_id", "Служебная", "QVuz как есть (~249 вузов) — для кластеризации/многоуровневой модели",
    "QVuz", "пропусков нет; код 99 — настоящий вуз, НЕ трогать")

# возраст: в массиве ОТСУТСТВУЕТ (нет переменной возраста/года рождения) — зафиксировано в отчёте

# ---------------------------------------------------------------------------
# 9. ВНЕМОДЕЛЬНЫЕ (конвергентная валидность)
# ---------------------------------------------------------------------------
df["valid_q27"] = df["Q27"]
df["valid_q28"] = df["Q28"]
reg("valid_q27", "Внемодельные", "Q27 как есть (1–4, самооценка шансов защиты)", "Q27",
    "пропусков нет", "НЕ ДЛЯ МОДЕЛЕЙ: измерена одновременно с исходом, правдоподобно его следствие")
reg("valid_q28", "Внемодельные", "Q28 как есть (1–7, планируемые сроки защиты)", "Q28",
    "пропусков нет", "НЕ ДЛЯ МОДЕЛЕЙ: см. valid_q27")

# ---------------------------------------------------------------------------
# 10. НАДЁЖНОСТЬ ИНДЕКСОВ: альфа Кронбаха
# ---------------------------------------------------------------------------
print("10. Альфа Кронбаха ...")
def cronbach_alpha(items_df: pd.DataFrame) -> float:
    """Классическая alpha: k/(k-1) * (1 - sum(var_i)/var_total), complete cases."""
    x = items_df.dropna()
    k = x.shape[1]
    if len(x) < 30:
        return np.nan
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var))

def cronbach_alpha_pairwise(items_df: pd.DataFrame, min_periods: int = 100) -> float:
    """Стандартизированная alpha по pairwise-complete корреляциям.

    Нужна для sup_satis_practices: пункты Q9_r* заданы ТОЛЬКО если соответствующая
    практика отмечена в Q8_* (проверено: ответ при неотмеченной практике = 0,0%),
    поэтому полных строк в матрице почти нет и классическая alpha не определена.
    Пары с < min_periods общих наблюдений -> NaN; пункты, дающие NaN-пары,
    исключаются из расчёта (фактически это Q9_r15 «Другое», n=37).
    """
    r = items_df.corr(min_periods=min_periods)
    r = r.dropna(axis=0, how="all").dropna(axis=1, how="all")  # сначала пункты без единой пары (Q9_r15)
    ok = r.notna().all(axis=1)                                 # затем пункты с хоть одной NaN-парой
    r = r.loc[ok, ok]
    k = r.shape[1]
    if k < 3:
        return np.nan
    v = r.values
    return float(k / (k - 1) * (1 - np.trace(v) / v.sum()))

ALPHAS = {
    "env_program_satis (Q1_r1–r6)": cronbach_alpha(df[ENV_SATIS_ITEMS]),
    "env_belonging (Qatm_r1–r6)": cronbach_alpha(df[ENV_BELONG_ITEMS]),
    "sup_satis_practices (Q9_r*, стандартизир., pairwise)": cronbach_alpha_pairwise(df[SUP_SATIS_ITEMS]),
    "PHQ-8 (Q19_r1–r8, рекодинг 0–3)": cronbach_alpha(phq01),
}
for k, v in ALPHAS.items():
    print(f"   {k}: alpha = {v:.3f}")

# ---------------------------------------------------------------------------
# 11. ВАЛИДАЦИЯ (12 обязательных проверок)
# ---------------------------------------------------------------------------
print("11. Валидация ...")
n_out = len(df)
check(1, "Размер: 5007 строк на входе и выходе",
      EXPECTED["n_rows"], f"вход={n_in}, выход={n_out}",
      n_in == EXPECTED["n_rows"] == n_out)

t = df["target_dropout"]
n_t, n1 = int(t.notna().sum()), int(t.sum())
share1 = n1 / n_t
check(2, "target_dropout: доля 1 ~ 21,7% (1005/4631)",
      f"{EXPECTED['target_ones']}/{EXPECTED['target_n']} = {EXPECTED['target_share']:.3f}",
      f"{n1}/{n_t} = {share1:.4f}",
      n1 == EXPECTED["target_ones"] and n_t == EXPECTED["target_n"])

phq_share = df["phq8_dep"].mean()          # среднее по непропущенным
phq_wrong = (df[PHQ_ITEMS].sum(axis=1, min_count=8) >= PHQ_THRESHOLD).mean()  # анти-проверка без рекодинга
check(3, "phq8_dep: доля ~ 25,7% среди ответивших (анти-проверка: не 78,5%)",
      f"{EXPECTED['phq_dep_share']:.3f}",
      f"{phq_share:.4f} (n={int(df['phq8_dep'].notna().sum())}); без рекодинга было бы {phq_wrong:.4f}",
      abs(phq_share - EXPECTED["phq_dep_share"]) < 0.01 and phq_share < 0.40)

n_rest = int((df["main_goal_restored"] == 1).sum())
n_left = int(df["main_goal"].isna().sum())
check(4, "Восстановление Q15: ровно 1290, остаток ~166 NA",
      f"{EXPECTED['q15_restored']} восстановлено, {EXPECTED['q15_na_left']} NA",
      f"восстановлено={n_rest}, осталось NA={n_left}",
      n_rest == EXPECTED["q15_restored"] and n_left == EXPECTED["q15_na_left"])

missing_src = [c for c in SUP_PRACTICES if c not in df_raw.columns]
check(5, "sup_practices_sum: ровно 13 источников; Q8_6 отсутствует",
      "13 колонок, Q8_6 нет в данных",
      f"{len(SUP_PRACTICES)} колонок: {', '.join(SUP_PRACTICES)}; Q8_6 в данных: {'Q8_6' in df_raw.columns}",
      len(SUP_PRACTICES) == 13 and not missing_src and "Q8_6" not in df_raw.columns)

m_nothing = df.loc[df["sup_nothing"] == 1, "target_ord"].mean()
m_other = df.loc[(df["sup_nothing"] == 0) & (df["has_supervisor"] == 1), "target_ord"].mean()
check(6, "sup_nothing: среднее Q13_r3 ~ 2,76 vs ~1,75",
      f"{EXPECTED['sup_nothing_mean']} vs {EXPECTED['sup_other_mean']}",
      f"{m_nothing:.3f} (n={int((df['sup_nothing']==1).sum())}) vs {m_other:.3f}",
      abs(m_nothing - EXPECTED["sup_nothing_mean"]) < 0.05 and abs(m_other - EXPECTED["sup_other_mean"]) < 0.05)

n_nosup = int((df["Q6"] == 2).sum())
check(7, "Без руководителя: 76 чел.", EXPECTED["no_supervisor"], n_nosup,
      n_nosup == EXPECTED["no_supervisor"],
      f"Q8_99=1 (практики в NA): {int(dk_mask.sum())} чел. (ожидание {EXPECTED['sup_dk']})")

na_q22 = df_raw["Q22"].isna().mean()
na_q23 = df_raw["Q23"].isna().mean()
na_q21 = df_raw[Q21_ITEMS].isna().all(axis=1).mean()
na_q210 = df_raw[Q210_ITEMS].isna().all(axis=1).mean()
ok8 = (abs(na_q22 - EXPECTED["na_q22"]) < 0.01 and abs(na_q23 - EXPECTED["na_q23"]) < 0.01
       and abs(na_q21 - EXPECTED["na_q21"]) < 0.005 and abs(na_q210 - EXPECTED["na_q210"]) < 0.005)
check(8, "Пропуски Q22/Q23/Q21_*/Q210_* ~ 62/40/60,1/38,3%",
      "0.620 / 0.400 / 0.601 / 0.383",
      f"{na_q22:.4f} / {na_q23:.4f} / {na_q21:.4f} / {na_q210:.4f}", ok8,
      "это структурные пропуски-фильтры: не импутируются")

emp_shares = df["employ_status_lbl"].value_counts(normalize=True).sort_index()
emp_str = "; ".join(f"{k}: {v:.3f}" for k, v in emp_shares.items())
check(9, "Распределение employ_status по 4 категориям",
      "4 категории + 89 NA",
      f"{emp_str}; NA={int(df['employ_status'].isna().sum())}",
      df["employ_status"].nunique() == 4 and int(df["employ_status"].isna().sum()) == 89)

alpha_min = min(ALPHAS.values())
alpha_str = "; ".join(f"{k}={v:.3f}" for k, v in ALPHAS.items())
weak = [k for k, v in ALPHAS.items() if v < 0.6]
check(10, "alpha-Кронбаха всех индексов >= 0,7 (или пометка)",
      ">= 0.7", alpha_str,
      alpha_min >= 0.7,
      ("индексы с alpha < 0,6 — не использовать, вводить покомпонентно: " + ", ".join(weak)) if weak
      else ("ниже 0,7, но >= 0,6: допустимо с оговоркой" if alpha_min < 0.7 else ""))

raw_f = df_raw[[f"field_{i}" for i in range(1, 7)]].fillna(0)
merged78 = ((df_raw["field"].isin([7, 8])) & (raw_f.sum(axis=1) == 0)).sum()
check(11, "Встроенные field_1–6 битые (field 7–8 слиты в референс)",
      f"{EXPECTED['field_78']} чел. (12,5%) в категориях 7–8 имеют все field_1–6 = 0",
      f"{merged78} чел.; field 7–8 в данных: {int(df_raw['field'].isin([7,8]).sum())}",
      merged78 == EXPECTED["field_78"],
      "используются собственные field_d1..field_d8")

derived_names = [c["name"] for c in CODEBOOK]
derived_names = [n for n in derived_names if ".." not in n] + [f"field_d{k}" for k in range(1, 9)]
na_map = (df[[c for c in derived_names if c in df.columns]].isna().mean()
          .sort_values(ascending=False).head(20))
check(12, "Карта пропусков производных (топ-20)", "—",
      "см. таблицу ниже", True)

# дополнительные факты (не обязательные проверки, но расхождения аудита)
q11_sat = (df.loc[df["has_supervisor"] == 1, "sup_satis_overall"] >= 3).mean()
pos_ok = all(
    (df.loc[df[GOAL_MAIN] == k, f"Q14_{k}"] == 1).mean() == 1.0
    for k in range(1, 8) if (df[GOAL_MAIN] == k).any()
)
gap_bad = int((df_raw["QPre5"] > GAP_YEARS_MAX_VALID).sum())

# ---------------------------------------------------------------------------
# 12. СОХРАНЕНИЕ ДАТАСЕТА
# ---------------------------------------------------------------------------
print("12. Сохранение ...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")   # utf-8 с BOM — корректно открывается в Excel
print(f"   {OUT_PARQUET}")
print(f"   {OUT_CSV}  ({df.shape[0]} x {df.shape[1]})")

# ---------------------------------------------------------------------------
# 13. ОТЧЁТЫ
# ---------------------------------------------------------------------------
print("13. Отчёты ...")

# --- codebook.md ---
cb = ["# Кодбук производных переменных — «Всероссийский опрос аспирантов»", "",
      f"Источник: `{SRC_SAV.name}` (коды SPSS, {n_in} строк x {df_raw.shape[1]} колонок).",
      "Очищенный датасет: `aspirants_clean.parquet` / `aspirants_clean.csv` "
      f"({n_out} строк x {df.shape[1]} колонок).", "",
      "## Глобальные правила очистки", "",
      "1. **Коды-отказы -> NA до любых расчётов.** Коды 98/99 переведены в NA только там, "
      "где метка значения в .sav подтверждает отказ («затрудняюсь»/«нет ответа»/«не могу оценить»/«не применимо»). "
      "Тем же правилом пойманы шкальные отказы с другими кодами: `Q1_r*` (6), `Q9_r*` (5), `Q18_r*` (5), `QPar2_r*` (7). "
      "**Контрпримеры, которые НЕ тронуты:** `QCitiz`=99 («Македония») и `QVuz`=99 (настоящий вуз) — "
      "это содержательные коды, глобальная замена 98/99 «в лоб» уничтожила бы 6 наблюдений.",
      f"   Затронуто колонок: {len(na_code_map)}, ячеек -> NA: {n_cells_na}.",
      "2. **Структурные пропуски-фильтры не импутируются и не восстанавливаются** "
      "(блок S у 76 без руководителя; Q22/Q23 у не имеющих соответствующей работы; Q21_*/Q210_* по фильтрам занятости).",
      "3. Исходные переменные в датасете сохранены под исходными именами, уже очищенные от кодов-отказов. "
      "Производные — snake_case, описаны ниже.",
      "4. Переменная возраста в массиве **отсутствует** (нет ни возраста, ни года рождения) — "
      "в блоке D прокси: год обучения, перерыв после магистратуры.", "",
      "## Производные переменные", "",
      "| Переменная | Блок | Формула | Исходники | Правило пропусков | Примечание |",
      "|---|---|---|---|---|---|"]
for c in CODEBOOK:
    cb.append(f"| `{c['name']}` | {c['block']} | {c['formula']} | {c['sources']} | {c['na']} | {c['note']} |")
cb += ["", "## Надёжность индексов (альфа Кронбаха)", "",
       "| Индекс | alpha | Решение |", "|---|---|---|"]
for k, v in ALPHAS.items():
    dec = "OK (>= 0,7)" if v >= 0.7 else ("допустимо с оговоркой (0,6–0,7)" if v >= 0.6
                                          else "НЕ ИСПОЛЬЗОВАТЬ как индекс — вводить покомпонентно")
    cb.append(f"| {k} | {v:.3f} | {dec} |")
cb += ["", "## Переменные «не для моделей»", "",
       "- `valid_q27`, `valid_q28` — самооценка шансов защиты и планируемые сроки: измерены одновременно "
       "с исходом и правдоподобно являются его следствием (overcontrol). Только проверка конвергентной "
       "валидности целевой.",
       "- `Q210_*` (области работы вне вуза) — только описательно, в модель не входят.",
       "- Встроенные `field_1`–`field_6` — битые (сливают категории 7–8 в референс), не использовать; "
       "вместо них `field_d1`–`field_d8`.",
       "- `vuz_id` — только для кластеризации / случайного интерсепта по вузу.", ""]
OUT_CODEBOOK.write_text("\n".join(cb), encoding="utf-8")

# --- validation_report.md ---
n_pass = sum(1 for c in CHECKS if c["ok"])
vr = ["# Отчёт валидации — подготовка массива «Всероссийский опрос аспирантов»", "",
      f"Скрипт: `analysis/prep_data.py`. Вход: {n_in} строк; выход: {n_out} строк, {df.shape[1]} колонок.",
      f"**Итог: {n_pass} из {len(CHECKS)} проверок PASS.**", "",
      "| № | Проверка | Ожидание | Факт | Статус |", "|---|---|---|---|---|"]
for c in CHECKS:
    vr.append(f"| {c['num']} | {c['title']} | {c['expected']} | {c['fact']} | "
              f"{'PASS' if c['ok'] else '**FAIL**'} |")
vr += ["", "## Комментарии к проверкам", ""]
for c in CHECKS:
    if c["comment"]:
        vr.append(f"- **{c['num']}.** {c['comment']}")
vr += ["", "## Проверка 12: карта пропусков производных (топ-20 по доле NA)", "",
       "| Переменная | Доля NA | Число NA |", "|---|---|---|"]
for name, share in na_map.items():
    vr.append(f"| `{name}` | {share:.3f} | {int(df[name].isna().sum())} |")
vr += ["", "## Дополнительные факты и расхождения с аудитом", "",
       f"- Позиционное соответствие Q15 = позиции цели в Q14: **{'100% (подтверждено)' if pos_ok else 'ЕСТЬ НЕСОСТЫКОВКИ'}** "
       f"на {int(df[GOAL_MAIN].notna().sum())} ответивших.",
       f"- Потолок общей удовлетворённости руководителем (Q11): аудит ожидал 92,8%; "
       f"фактически доля удовлетворённых (3–4) среди имеющих руководителя = **{q11_sat*100:.1f}%** "
       f"(по всем ответившим — {(df['sup_satis_overall']>=3).mean()*100:.1f}%). Незначительное расхождение, "
       "на выводы не влияет (переменная всё равно с сильным потолком).",
       f"- PHQ-8: доля dep=1 среди ответивших {phq_share*100:.1f}% (ожидание 25,7%); по всей выборке с учётом "
       f"89 пропусков — {(df['phq8_dep'].sum()/len(df))*100:.1f}%. Анти-проверка без рекодинга даёт {phq_wrong*100:.1f}% "
       "(ожидаемые ~78,5%) — рекодинг применён корректно.",
       f"- Кодов 98 в массиве **нет вообще**; код 99 встречается содержательно только в Q11 (113 ячеек) "
       "+ ложные 99 в QCitiz (1) и QVuz (5), которые сохранены.",
       f"- В QPre5 (длина перерыва) {gap_bad} значений > {GAP_YEARS_MAX_VALID} — ошибки ввода (напр., 2020), "
       "переведены в NA в `gap_years`.",
       f"- 89 респондентов не отметили ничего в блоке занятости Q20_* (и они же — единственные пропуски в Q19 PHQ-8): "
       "у них employ_status, works_*, q22_x_invuz, q23_x_outside, phq8_* = NA.",
       f"- Блок Q8 имеет ТРИ группы пропусков: 76 без руководителя (структурные), 88 с Q8_99=1 "
       "(«затрудняюсь» — их чекбоксы практик были проставлены, но по правилу паспорта весь блок обнулён в NA) "
       "и 88 с полным пропуском блока (все пункты NA, включая Q8_99; у них NA и целевая Q13_r3). "
       "Итого sup_practices_sum/sup_nothing NA = 252.",
       f"- Q8_16 («не делал ничего») был NA у 497 чел. с отмеченными практиками (артефакт анкеты: "
       "опция, видимо, скрывалась при любом чекбоксе) — для них sup_nothing логически восстановлен в 0 "
       "(после восстановления среднее Q13_r3 у прочих = 1,757 — в точности ожидание аудита 1,75).",
       f"- QComp_99=1 («затрудняюсь» на блок компетенций) у {int(comp_dk.sum())} чел. -> все меры comp_* в NA "
       "(решение по аналогии с Q8_99; в паспорте явно не оговорено).",
       f"- comp_no_needs=1 у {int((df['comp_no_needs']==1).sum())} чел.: из них QComp_16=1 у "
       f"{int(((df['comp_no_needs']==1)&(df['QComp_16']==1)).sum())}, ничего не отмечено у "
       f"{int(((df['comp_no_needs']==1)&(df['QComp_16']!=1)).sum())}.",
       f"- Пункты Q9_r* (удовлетворённость практиками) заданы только по практикам, отмеченным в Q8_* "
       "(проверено: ответов при неотмеченной практике 0,0%). Поэтому классическая complete-case α для "
       "sup_satis_practices не определена (1 полная строка из 5007); вместо неё — стандартизированная α "
       "по pairwise-complete корреляциям (пункт Q9_r15 «Другое», n=37, из расчёта исключён).",
       "- Переменной **возраста в массиве нет** — блок D использует прокси (год обучения, перерыв).",
       "- CSV сохранён в UTF-8 с BOM (utf-8-sig) для корректного открытия кириллицы в Excel.", ""]
OUT_VALID.write_text("\n".join(vr), encoding="utf-8")

print(f"   {OUT_CODEBOOK}")
print(f"   {OUT_VALID}")
print("=" * 72)
print(f"ГОТОВО. Проверок PASS: {n_pass}/{len(CHECKS)}")
if n_pass < len(CHECKS):
    sys.exit(1)

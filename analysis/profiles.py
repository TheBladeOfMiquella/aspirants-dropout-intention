# -*- coding: utf-8 -*-
"""Типичные профили аспирантов: предсказанные вероятности (пулинг по 20 импутациям)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from models_hier import build_design, TARGET

imp = pd.read_parquet(BASE / "mice_imputations.parquet")

# медианный «типичный» аспирант — берём медианы/моды по датасету m0
g0 = imp[imp["m"] == 0].drop(columns=["m"]).reset_index(drop=True)
X0 = build_design(g0, ["D","M","L","E","S"])
base = X0.median()

PROFILES = {
    "P1_благополучный": {},   # всё по медиане
    "P2_разочарованный_средой": {
        "env_belonging": 1.0, "env_quality": 4.0, "env_program_satis": 1.5,
        "sup_satis_overall": 2.0, "sup_practices_sum": 2.0,
    },
    "P3_стеснённый_обстоятельствами": {
        "income_ord": 2.0, "covid_hardship": 6.0,
    },
    "P4_без_связи_работы_вне_вуза": {
        "q23_x_outside": 1.0, "income_ord": 3.0, "covid_hardship": 4.0,
    },
}

preds = {k: [] for k in PROFILES}
for m, g in imp.groupby("m"):
    d = g.drop(columns=["m"]).reset_index(drop=True)
    X = build_design(d, ["D","M","L","E","S"])
    y = d[TARGET].astype(float)
    model = sm.Logit(y, X).fit(disp=0, maxiter=300)
    for name, mods in PROFILES.items():
        row = X.median().copy()
        for k, v in mods.items():
            if k in row.index:
                row[k] = v
        # дамми статуса занятости для P4: работает вне вуза
        if name == "P4_без_связи_работы_вне_вуза":
            for c in row.index:
                if c.startswith("employ_status_"):
                    row[c] = 1.0 if c.endswith("2.0") or c.endswith("_2") else 0.0
        p = float(model.predict(pd.DataFrame([row]))[0])
        preds[name].append(p)

out = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in preds.items()}
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(BASE / "profiles.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

#!/usr/bin/env python3
"""Creates the practice datasets in data/ (seeded -> identical every run).   python tools/generate_data.py

students.csv : 600 students, 11 columns, ~3% blanks (np.genfromtxt turns them into nan)
sensor.csv   : 10,000 one-minute readings, 5 columns, blanks + -999 sentinels + spikes + one sensor outage
houses.csv   : 400 houses for the linear-regression project
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2024)


def write_csv(name, header, rows):
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")


# ---------------------------------------------------------------- students
n = 600
age = rng.integers(17, 26, n)
gender = rng.choice(["M", "F"], n)
height = np.where(gender == "M", rng.normal(174, 7, n), rng.normal(161, 6.5, n)).round(1)
weight = (height - 100 + rng.normal(0, 8, n)).round(1)
study = np.clip(rng.gamma(4, 1.1, n), 0.5, 14).round(1)
attend = np.clip(rng.normal(82, 11, n), 35, 100).round(0).astype(int)
ability = rng.normal(0, 1, n)
def score(base, w_study, w_att, noise):
    s = base + 9 * ability + w_study * study + w_att * (attend - 80) + rng.normal(0, noise, n)
    return np.clip(s, 0, 100).round(0).astype(int)
math = score(48, 1.6, 0.20, 6)
physics = score(45, 1.3, 0.18, 7)
chem = score(47, 1.2, 0.15, 7)
english = score(55, 0.9, 0.10, 8)
sid = np.arange(1001, 1001 + n)
cols = [sid, age, gender, height, weight, math, physics, chem, english, attend, study]
header = ["student_id", "age", "gender", "height_cm", "weight_kg", "math", "physics", "chemistry", "english", "attendance_pct", "study_hours"]
rows = [list(r) for r in zip(*[c.tolist() for c in cols])]
for j in range(3, 11):                                   # blanks in numeric columns
    for i in rng.choice(n, int(n * 0.03), replace=False):
        rows[i][j] = None
write_csv("students.csv", header, rows)

# ---------------------------------------------------------------- sensor
m = 10_000
t = np.arange(m) * 60                                     # seconds, one reading per minute
day = 2 * np.pi * t / 86400
temp = 22 + 4 * np.sin(day - 1.5) + rng.normal(0, 0.25, m)
hum = 55 - 9 * np.sin(day - 1.5) + rng.normal(0, 1.2, m)
pres = 1013 + 2.5 * np.sin(day / 3) + np.cumsum(rng.normal(0, 0.03, m))
vib = np.abs(rng.normal(0.4, 0.08, m))
spikes = rng.choice(m, 25, replace=False)
temp[spikes] += rng.choice([-1, 1], 25) * rng.uniform(7, 12, 25)
vib[rng.choice(m, 12, replace=False)] += rng.uniform(1.5, 3, 12)
temp[6000:6040] = np.nan                                  # a 40-minute outage
rows = []
data = [t, temp.round(2), hum.round(2), pres.round(2), vib.round(3)]
arr = np.column_stack(data).astype(object)
for k in range(1, 5):
    idx = rng.choice(m, 150, replace=False)
    arr[idx, k] = None                                    # blank readings
idx = rng.choice(m, 40, replace=False)
arr[idx, 2] = -999                                        # sentinel for "sensor error"
arr[:, 0] = arr[:, 0].astype(int)
sens_rows = [[None if (isinstance(v, float) and np.isnan(v)) else v for v in r] for r in arr.tolist()]
write_csv("sensor.csv", ["t_sec", "temperature_c", "humidity_pct", "pressure_hpa", "vibration_g"], sens_rows)

# ---------------------------------------------------------------- houses
h = 400
area = rng.integers(500, 3600, h)
bed = np.clip(np.round(area / 900 + rng.normal(0, 0.7, h)), 1, 6).astype(int)
hage = rng.integers(0, 51, h)
price = (30 + 0.09 * area + 12 * bed - 0.8 * hage + rng.normal(0, 15, h)).round(1)
write_csv("houses.csv", ["area_sqft", "bedrooms", "age_years", "price_k"], zip(area.tolist(), bed.tolist(), hage.tolist(), price.tolist()))

meta = {
    "students": {"file": "data/students.csv", "rows": n, "desc": "Exam results and body measurements of 600 students (gender is coded M=0, F=1 in the numeric array). About 3% of the numeric cells are blank -> nan.",
                 "columns": {"student_id": "ID 1001-1600", "age": "years", "gender": "text M/F in the csv; 0/1 in `students`", "height_cm": "", "weight_kg": "", "math": "score 0-100", "physics": "score 0-100", "chemistry": "score 0-100", "english": "score 0-100", "attendance_pct": "0-100", "study_hours": "hours per day"}},
    "sensor": {"file": "data/sensor.csv", "rows": m, "desc": "10,000 one-minute readings from an environment sensor (about 7 days). Contains blanks (nan), -999 error codes in humidity, temperature spikes, vibration bursts and a 40-minute outage.",
               "columns": {"t_sec": "seconds since start", "temperature_c": "deg C", "humidity_pct": "% (-999 = sensor error)", "pressure_hpa": "hPa", "vibration_g": "g"}},
    "houses": {"file": "data/houses.csv", "rows": h, "desc": "400 house sales for the regression project: price (in thousands) depends on area, bedrooms and age plus noise.",
               "columns": {"area_sqft": "living area", "bedrooms": "count", "age_years": "age of the house", "price_k": "sale price in thousands (target)"}},
}
(OUT / "datasets.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", ", ".join(p.name for p in sorted(OUT.iterdir())))

# ── APPENDIX B: Feature Engineering ─────────────────────────────────────────
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler

era  = pd.read_csv("ERA5Land_KabulBasin_1990_2022.csv", parse_dates=["date"])
wapda= pd.read_csv("WAPDA_Nowshera_1982_2022.csv",     parse_dates=["date"])
df   = era.merge(wapda, on="date").sort_values("date").reset_index(drop=True)

# Discharge lags
for lag in [1,2,3,4,5,7]:
    df[f"Q_lag{lag}"] = df["Q"].shift(lag)

# Precipitation lags and rolling sums
for lag in [0,1,2,3,5,7]:
    df[f"P_lag{lag}" if lag>0 else "P_t"] = df["P_mm"].shift(lag)
df["P_3day_sum"]  = df["P_mm"].rolling(3).sum()
df["P_7day_sum"]  = df["P_mm"].rolling(7).sum()

# Temperature lags
for lag in [0,1,2]:
    df[f"T_lag{lag}" if lag>0 else "T_t"] = df["T_C"].shift(lag)

# Snow lags
for lag in [0,7,14]:
    df[f"snow_lag{lag}"] = df["snow_mm"].shift(lag)

# Seasonal encodings
df["month_sin"]    = np.sin(2*np.pi*df["date"].dt.month/12)
df["month_cos"]    = np.cos(2*np.pi*df["date"].dt.month/12)
df["is_monsoon"]   = df["date"].dt.month.isin([7,8,9]).astype(int)
df["is_snowmelt"]  = df["date"].dt.month.isin([4,5,6]).astype(int)

# Targets
for n in [1,2,3]: df[f"Q_t{n}"] = df["Q"].shift(-n)

df.dropna(inplace=True)

FEATURES = [f"Q_lag{l}" for l in [1,2,3,4,5,7]] + \
           ["P_t"]+[f"P_lag{l}" for l in [1,2,3,5,7]] + \
           ["P_3day_sum","P_7day_sum"] + \
           ["T_t","T_lag1","T_lag2"] + \
           ["snow_lag0","snow_lag7","snow_lag14"] + \
           ["month_sin","month_cos","is_monsoon","is_snowmelt"]
TARGETS = ["Q_t1","Q_t2","Q_t3"]

train = df[df["date"]<"2016-01-01"]
val   = df[(df["date"]>="2016-01-01")&(df["date"]<"2020-01-01")]
test  = df[df["date"]>="2020-01-01"]

scaler = MinMaxScaler()
X_tr = scaler.fit_transform(train[FEATURES+TARGETS])
X_va = scaler.transform(val[FEATURES+TARGETS])
X_te = scaler.transform(test[FEATURES+TARGETS])

for name, arr in [("train",X_tr),("val",X_va),("test",X_te)]:
    pd.DataFrame(arr, columns=FEATURES+TARGETS).to_csv(f"scaled_{name}.csv",index=False)
print("Feature engineering complete.")

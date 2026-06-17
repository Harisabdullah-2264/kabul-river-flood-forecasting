# ── APPENDIX C: XGBoost Training and SHAP Analysis ──────────────────────────
import pandas as pd, numpy as np, shap, joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

def nse(obs,sim): return 1-np.sum((obs-sim)**2)/np.sum((obs-np.mean(obs))**2)
def kge(obs,sim):
    r=np.corrcoef(obs,sim)[0,1]; a=np.std(sim)/np.std(obs)
    b=np.mean(sim)/np.mean(obs); return 1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
def rmse(obs,sim): return np.sqrt(np.mean((obs-sim)**2))

tr=pd.read_csv("scaled_train.csv"); va=pd.read_csv("scaled_val.csv")
te=pd.read_csv("scaled_test.csv")
FEATURES=[c for c in tr.columns if not c.startswith("Q_t")]

for n,tgt in enumerate(["Q_t1","Q_t2","Q_t3"],1):
    print(f"\n=== {n}-day lead ===")
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
                         early_stopping_rounds=50, random_state=42, n_jobs=-1)
    model.fit(tr[FEATURES],tr[tgt],eval_set=[(va[FEATURES],va[tgt])],verbose=50)
    pred=model.predict(te[FEATURES])
    obs =te[tgt].values
    print(f"  NSE={nse(obs,pred):.3f}  KGE={kge(obs,pred):.3f}  RMSE={rmse(obs,pred):.1f}")
    joblib.dump(model,f"xgb_{n}day.pkl")

# SHAP analysis on 1-day model
model1 = joblib.load("xgb_1day.pkl")
explainer = shap.TreeExplainer(model1)
shap_vals = explainer.shap_values(te[FEATURES])
shap_df   = pd.DataFrame(shap_vals, columns=FEATURES)
shap_df.to_csv("shap_values_1day.csv",index=False)

# Seasonal split
te_dates = pd.read_csv("scaled_test.csv")  # re-use to attach dates if available
for season,months in [("monsoon",[7,8,9]),("snowmelt",[4,5,6]),("dry",list(range(1,13)))][:2]:
    mask = te["is_monsoon"]==1 if season=="monsoon" else te["is_snowmelt"]==1
    sv = explainer.shap_values(te[FEATURES][mask])
    seas_df = pd.DataFrame(np.abs(sv).mean(axis=0),index=FEATURES).sort_values(0,ascending=False)
    print(f"\n{season.upper()} top-10:"); print(seas_df.head(10))
print("SHAP analysis complete.")

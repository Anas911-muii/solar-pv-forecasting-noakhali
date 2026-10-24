"""
Multi-horizon solar PV power forecasting for Noakhali, Bangladesh.

Pipeline: NASA POWER hourly data -> pvlib PV simulation (1 kWp) -> clear-sky-ratio
target -> smart persistence baseline, XGBoost, LSTM -> hold-out test,
time-series cross-validation (XGBoost) and seasonal analysis.

Run in Google Colab or locally:  python solar_forecasting_noakhali.py
Outputs (CSV files) are written to the current folder.
"""
import numpy as np
import pandas as pd
import requests
import pvlib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

LAT, LON = 22.87, 91.10          # Noakhali
HORIZONS = [1, 3, 6, 24]         # forecast horizons (hours)
RUN_LSTM = True                  # set False to skip the (slow) LSTM part

FEATS = ["kt", "kt_day", "kt_l3", "kt_l24", "kt_m24", "Temp", "RH", "Wind",
         "Pressure", "dP3", "dRH3", "hour_t", "month", "pvclear_t"]
SEASONS = {"Dry": [11, 12, 1, 2], "Pre-monsoon": [3, 4, 5],
           "Monsoon": [6, 7, 8, 9], "Post-monsoon": [10]}


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# ---------------------------------------------------------------- 1. data
def load_data():
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M,PS",
        "community": "RE", "latitude": LAT, "longitude": LON,
        "time-standard": "UTC", "format": "JSON",
    }

    def get(start, end):
        params["start"], params["end"] = start, end
        r = requests.get(url, params=params, timeout=300).json()
        x = pd.DataFrame(r["properties"]["parameter"])
        x.index = pd.to_datetime(x.index, format="%Y%m%d%H")
        return x

    df = pd.concat([get("20150101", "20191231"), get("20200101", "20241231")])
    df.index = df.index + pd.Timedelta(hours=6)            # UTC -> UTC+6
    df = df.replace(-999, np.nan).interpolate(limit=3)
    df.columns = ["GHI", "GHI_clear", "Temp", "RH", "Wind", "Pressure"]
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["GHI_lag1"] = df["GHI"].shift(1)    # only drops the first row (keeps sample count)
    return df.dropna()


# ------------------------------------------------------ 2. PV simulation
def pv_from_ghi(df, ghi):
    """1 kWp, tilt 23 deg, south-facing; Erbs -> Faiman -> PVWatts. Returns kW/kWp."""
    loc = pvlib.location.Location(LAT, LON, tz="Asia/Dhaka")
    mid = df.index.tz_localize("Asia/Dhaka") + pd.Timedelta(minutes=30)
    sp = loc.get_solarposition(mid)
    g = pd.Series(ghi.values, index=mid)
    er = pvlib.irradiance.erbs(g, sp["zenith"], mid)
    poa = pvlib.irradiance.get_total_irradiance(
        23, 180, sp["zenith"], sp["azimuth"], dni=er["dni"], ghi=g, dhi=er["dhi"],
        dni_extra=pvlib.irradiance.get_extra_radiation(mid))
    tcell = pvlib.temperature.faiman(poa["poa_global"], df["Temp"].values, df["Wind"].values)
    pdc = pvlib.pvsystem.pvwatts_dc(poa["poa_global"], tcell, 1000, -0.004)
    return (pdc * 0.86 / 1000).fillna(0).values


def build_features(df):
    df = df.copy()
    df["PV_kW"] = pv_from_ghi(df, df["GHI"])
    df["PV_clear"] = pv_from_ghi(df, df["GHI_clear"])
    df["kt"] = (df["GHI"] / df["GHI_clear"].replace(0, np.nan)).fillna(0).clip(0, 1.2)
    df["kt_day"] = df["kt"].where(df["GHI_clear"] > 10).ffill().fillna(0)   # last daylight kt
    df["kt_l3"] = df["kt_day"].shift(3)
    df["kt_l24"] = df["kt_day"].shift(24)
    df["kt_m24"] = df["kt_day"].rolling(24).mean()
    df["dP3"] = df["Pressure"] - df["Pressure"].shift(3)
    df["dRH3"] = df["RH"] - df["RH"].shift(3)
    return df


def make_horizon(d0, h):
    d = d0.copy()
    d["target"] = d["PV_kW"].shift(-h)
    d["pvclear_t"] = d["PV_clear"].shift(-h)
    d["hour_t"] = (d["hour"] + h) % 24
    return d.dropna()


# ----------------------------------------------------------- 3. XGBoost
def fit_xgb(tr):
    """Predict the ratio PV / PV_clear on daylight samples."""
    m = tr["pvclear_t"] > 0.02
    ratio = (tr["target"] / tr["pvclear_t"]).clip(0, 1.5)
    model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    return model.fit(tr.loc[m, FEATS], ratio[m])


def predict_xgb(model, te):
    return np.where(te["pvclear_t"] > 0.02, model.predict(te[FEATS]) * te["pvclear_t"], 0)


def holdout_xgb(d0):
    rows = []
    for h in HORIZONS:
        d = make_horizon(d0, h)
        s = int(len(d) * 0.8)                       # chronological 80/20 split
        tr, te = d.iloc[:s], d.iloc[s:]
        p = predict_xgb(fit_xgb(tr), te)
        y = te["target"].values
        day = y > 0.01                              # daytime samples only
        smart = (te["kt_day"] * te["pvclear_t"]).values     # smart persistence
        naive = te["PV_kW"].values                          # plain persistence
        r_s, r_x, r_n = rmse(y[day], smart[day]), rmse(y[day], p[day]), rmse(y[day], naive[day])
        rows.append([h, round(r_n, 4), round(r_s, 4), round(r_x, 4),
                     round((1 - r_x / r_s) * 100, 1), round(r2_score(y[day], p[day]), 3)])
    return pd.DataFrame(rows, columns=["Horizon_h", "Persistence", "SmartPersist",
                                       "XGB", "XGB_skill_%", "XGB_R2"])


def cv_xgb(d0):
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    for h in HORIZONS:
        d = make_horizon(d0, h).reset_index(drop=True)
        xs, ss = [], []
        for tr_i, te_i in tscv.split(d):
            tr, te = d.iloc[tr_i], d.iloc[te_i]
            p = predict_xgb(fit_xgb(tr), te)
            y = te["target"].values
            day = y > 0.01
            smart = (te["kt_day"] * te["pvclear_t"]).values
            xs.append(rmse(y[day], p[day]))
            ss.append(rmse(y[day], smart[day]))
        xs, ss = np.array(xs), np.array(ss)
        rows.append([h, round(ss.mean(), 4), round(ss.std(), 4), round(xs.mean(), 4),
                     round(xs.std(), 4), round((1 - xs / ss).mean() * 100, 1)])
    return pd.DataFrame(rows, columns=["Horizon_h", "Smart_mean", "Smart_std",
                                       "XGB_mean", "XGB_std", "Skill_%"])


def season_xgb(d0):
    rows = []
    for h in HORIZONS:
        d = make_horizon(d0, h)
        s = int(len(d) * 0.8)
        tr, te = d.iloc[:s], d.iloc[s:]
        p = predict_xgb(fit_xgb(tr), te)
        y = te["target"].values
        smart = (te["kt_day"] * te["pvclear_t"]).values
        mo = te["month"].values
        for name, months in SEASONS.items():
            k = (y > 0.01) & np.isin(mo, months)
            if k.sum():
                a, b = rmse(y[k], smart[k]), rmse(y[k], p[k])
                rows.append([name, h, int(k.sum()), round(a, 4), round(b, 4),
                             round((1 - b / a) * 100, 1)])
    return pd.DataFrame(rows, columns=["Season", "Horizon_h", "N", "Smart", "XGB", "Skill_%"])


# ------------------------------------------------------------- 4. LSTM
def lstm_holdout(df):
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    tf.random.set_seed(42)

    df = df.copy()
    df["hs"], df["hc"] = np.sin(2 * np.pi * df["hour"] / 24), np.cos(2 * np.pi * df["hour"] / 24)
    df["ms"], df["mc"] = np.sin(2 * np.pi * df["month"] / 12), np.cos(2 * np.pi * df["month"] / 12)
    cols = ["kt", "kt_day", "Temp", "RH", "Wind", "Pressure", "PV_clear", "hs", "hc", "ms", "mc"]
    L, F, n = 24, len(cols), len(df)
    A = df[cols].values.astype("float32")
    W = np.lib.stride_tricks.sliding_window_view(A, (L, F))[:, 0]
    pv, pvc = df["PV_kW"].values, df["PV_clear"].values
    hr, ktd = df["hour"].values, df["kt_day"].values

    rows = []
    for h in HORIZONS:
        t = np.arange(L - 1, n - h)
        X = W[:len(t)]
        y, pc = pv[t + h], pvc[t + h]
        ht = (hr[t] + h) % 24
        S = np.c_[pc, np.sin(2 * np.pi * ht / 24), np.cos(2 * np.pi * ht / 24)].astype("float32")
        ratio = np.clip(y / np.where(pc > 0.02, pc, np.nan), 0, 1.5)

        s = int(len(t) * 0.8)
        mu = X[:s].reshape(-1, F).mean(0)
        sd = X[:s].reshape(-1, F).std(0) + 1e-6
        Xn = ((X - mu) / sd).astype("float32")
        tm = (pc[:s] > 0.02) & ~np.isnan(ratio[:s])

        inp, st = layers.Input((L, F)), layers.Input((3,))
        z = layers.concatenate([layers.LSTM(64)(inp), st])
        z = layers.Dense(32, activation="relu")(z)
        model = Model([inp, st], layers.Dense(1)(z))
        model.compile(optimizer="adam", loss="mse")
        model.fit([Xn[:s][tm], S[:s][tm]], ratio[:s][tm], epochs=12, batch_size=256,
                  validation_split=0.1, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

        r_pred = model.predict([Xn[s:], S[s:]], verbose=0).ravel()
        p = np.where(pc[s:] > 0.02, np.clip(r_pred, 0, 1.5) * pc[s:], 0)
        yt = y[s:]
        day = yt > 0.01
        smart = ktd[t[s:]] * pc[s:]
        r_l, r_s = rmse(yt[day], p[day]), rmse(yt[day], smart[day])
        rows.append([h, round(r_l, 4), round((1 - r_l / r_s) * 100, 1), round(r2_score(yt[day], p[day]), 3)])
        print("LSTM done, horizon", h)
    return pd.DataFrame(rows, columns=["Horizon_h", "LSTM", "LSTM_skill_%", "LSTM_R2"])


# --------------------------------------------------------------- main
if __name__ == "__main__":
    df = load_data()
    df = build_features(df)
    print("samples:", len(df))

    holdout = holdout_xgb(df)
    if RUN_LSTM:
        holdout = holdout.merge(lstm_holdout(df), on="Horizon_h")
    print(holdout)
    holdout.to_csv("holdout_results.csv", index=False)

    cv = cv_xgb(df)
    print(cv)
    cv.to_csv("cv_results.csv", index=False)

    seasons = season_xgb(df)
    print(seasons.to_string(index=False))
    seasons.to_csv("season_results.csv", index=False)

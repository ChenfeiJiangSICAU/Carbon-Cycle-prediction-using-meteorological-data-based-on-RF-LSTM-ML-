import xarray as xr
import numpy as np
import pandas as pd

# ds: 你的多时刻 Dataset（concat 后）
# ----------------------------------
import glob
files = glob.glob("../Data/*.nc")
valid_files = []

#Check files one by one
for f in files:
    try:
        xr.open_dataset(f)
        valid_files.append(f)
    except Exception as e:
        print(f"Skip the bad file: {f}")
        print("Reason:", e)

print(f"Valid: {len(valid_files)}")

#Read_Combine_Select city

ds = xr.open_mfdataset(
    valid_files,
    combine="by_coords",
    parallel=True,
    chunks="auto" #读取是否分块
)

ds_city = ds.sel(
    latitude=30.0,
    longitude=103.0,
    method="nearest",
)

#Select Pressure Layer
levels = [1000]
ds_sel = ds_city.sel(pressure_level=levels)
df = ds_sel.to_dataframe().reset_index()

columns_to_drop = ['number','ciwc','cswc','cc','clwc']
df.drop(columns_to_drop, axis=1, inplace=True)


# 时间特征（关键）
df["hour"] = df["valid_time"].dt.hour
df["doy"]  = df["valid_time"].dt.dayofyear

# 周期编码（避免离散问题）
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

df["doy_sin"]  = np.sin(2*np.pi*df["doy"]/365)
df["doy_cos"]  = np.cos(2*np.pi*df["doy"]/365)

# 风速
df["wind_speed"] = np.sqrt(df["u"]**2 + df["v"]**2)


# 温度（假设单位 K → 转 °C）
T = df["t"] - 273.15
RH = df["r"] / 100.0  # 归一化

# -------------------------
# GPP（光合作用）
# -------------------------
Topt = 25
sigma = 10

light = np.maximum(0, np.sin(2*np.pi*df["hour"]/24))

GPP = (
    10
    * light
    * np.exp(-((T - Topt)/sigma)**2)
    * RH
)

# -------------------------
# Reco（呼吸）
# -------------------------
Reco = 2 * np.exp(0.08 * T)

# -------------------------
# 空间异质性（避免过拟合公式）
# -------------------------
lat_factor = 1 - np.abs(df["latitude"]) / 90
GPP  *= lat_factor
Reco *= (1 + 0.3 * (1 - lat_factor))

# -------------------------
# NEE
# -------------------------
NEE = Reco - GPP

# 加噪声（关键）
NEE += np.random.normal(0, 0.5, size=len(NEE))

df["NEE_sim"] = NEE


df = df.sort_values("valid_time")

for lag in [1, 2]:
    df[f"t_lag{lag}"]  = df["t"].shift(lag)
    df[f"r_lag{lag}"]  = df["r"].shift(lag)
    df[f"wind_lag{lag}"] = df["wind_speed"].shift(lag)

df = df.dropna()

print('Preprocess successfully done')
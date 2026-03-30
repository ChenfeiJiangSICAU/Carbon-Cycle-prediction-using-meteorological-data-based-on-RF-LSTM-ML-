from Preprocessing import df


features = [
    "t", "r", "u", "v", "wind_speed",
    "latitude", "longitude",
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
    "t_lag1", "t_lag2",
    "r_lag1", "r_lag2",
    "wind_lag1", "wind_lag2"
]

X = df[features]
y = df["NEE_sim"]


train_idx = df["valid_time"].dt.year <= 2024
test_idx = df["valid_time"].dt.year == 2025


X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print('Split successfully done')


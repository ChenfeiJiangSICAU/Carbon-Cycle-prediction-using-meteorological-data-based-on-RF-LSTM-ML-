from xgboost import XGBRegressor
from Dataset import X_train,y_train,X_test,y_test
import numpy as np


model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist"
)

model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
model.save_model("xgb_model.json")

print('Trained Successfully')
print("RMSE:", rmse)
print("R2:", r2)
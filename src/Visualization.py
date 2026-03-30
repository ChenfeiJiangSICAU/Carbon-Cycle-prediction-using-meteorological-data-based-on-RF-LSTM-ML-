import matplotlib.pyplot as plt
from Preprocessing import df
from Dataset import test_idx,y_test
from Model import y_pred


plt.show()
#%%
import matplotlib.dates as mdates

plt.figure(figsize=(16,4))

plt.plot(df.loc[test_idx, "valid_time"], y_test, label="True")
plt.plot(df.loc[test_idx, "valid_time"], y_pred, label="Pred")

# 设置主刻度每 1 天显示一次
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xticks(rotation=45)
plt.legend()
plt.title("Time Series Comparison")
plt.show()




df_test = df.loc[test_idx].copy()
df_test["pred"] = y_pred

group = df_test.groupby("hour").mean(numeric_only=True)

plt.figure()
plt.plot(group.index, group["NEE_sim"], label="True")
plt.plot(group.index, group["pred"], label="Pred")

plt.xlabel("Hour")
plt.title("Diurnal Cycle")
plt.legend()
plt.show()


from Model import model
from xgboost import plot_importance

plot_importance(model)
plt.show()



error = y_pred - y_test

plt.figure()
plt.hist(error, bins=100)
plt.title("Error Distribution")
plt.show()
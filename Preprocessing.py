import xarray as xr
import glob
files = glob.glob("your file path")
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

#Read_Combine_Select
import xarray as xr

ds = xr.open_mfdataset(
    valid_files,
    combine="by_coords",
    parallel=False,
    chunks=None
)

ds_city = ds.sel(
    latitude=30.0,
    longitude=103.0,
    method="nearest"
)

#Select Pressure Layer
levels = [1000, 850, 500]

ds_sel = ds_city.sel(pressure_level=levels)


#Turn to Dataframe
#Note: Reset index here is used to turn '(valid_time, pressure_level)' these two indexes into normal columns in df,preparing for the following Pivot conversion
df = ds_sel.to_dataframe().reset_index()

#Data structure conversion,Pivot
df_wide = df.pivot_table(
    index="valid_time",
    columns="pressure_level",
    values=["t", "r", "u", "v"]
)

#Column index flattening

df_wide.columns = [
    f"{var}_{int(level)}"
    for var, level in df_wide.columns
]

#Reset index
df_wide = df_wide.reset_index()

#Make labels {delete}
import numpy as np

df_wide["hour"] = df_wide["valid_time"].dt.hour

#Note: Here we created artificial data labels to fill in the blanks. In actual engineering, you need to replace these with your own data labels
df_wide["Y"] = (
    0.3 * df_wide["t_850"]

    -0.2 * df_wide["r_850"]

    +0.05 * (df_wide["u_850"]**2 + df_wide["v_850"]**2)**0.5

    +3 * np.maximum(0, np.sin(2*np.pi*df_wide["hour"]/24))

    +2 * np.sin(2*np.pi*df_wide["valid_time"].dt.dayofyear/365)

    + np.random.normal(0, 0.5, len(df_wide))
)

#Save
df_wide.to_parquet("202305.parquet")
print('Successfully Saved')
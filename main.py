
import pandas as pd

df = pd.read_parquet("20260218_144523_sales_data.parquet", engine="fastparquet")
print(df.head())

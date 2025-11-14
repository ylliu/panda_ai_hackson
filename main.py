from feture_data import FutureData

future_data = FutureData()
list = future_data.get_master_future()
df = future_data.download_main_contract_minute_data(list)
print(df.head(5))
df.to_csv("LC_20230731_20251030.csv", index=False)

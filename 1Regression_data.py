import pandas as pd
import quandl

quandl.ApiConfig.api_key = "VAK3ZhAMmzFCCMBXXQyp"   # api set
df = quandl.get("WIKI/GOOGL")
pd.set_option('display.width', None)    # fix pandas ...
# print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())


import pandas as pd
import quandl, math

quandl.ApiConfig.api_key = "VAK3ZhAMmzFCCMBXXQyp"
df = quandl.get("WIKI/GOOGL")
pd.set_option('display.width', None)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)     # spare
forecast_out = int(math.ceil(0.01*len(df)))     # prodict
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head())

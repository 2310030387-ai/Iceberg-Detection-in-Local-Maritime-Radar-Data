import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df[['band_1','band_2','inc_angle']].values
y = train_df['is_iceberg'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_test = scaler.transform(test_df[['band_1','band_2','inc_angle']].values)

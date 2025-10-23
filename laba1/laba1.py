import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = pd.read_csv('test.csv')

print(data.head(10))

missing_values = data.isnull().sum()
print(missing_values)

for col in ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']:
    mode_val = data[col].mode()[0]
    data[col].fillna(mode_val, inplace=True)

for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']:
    median_val = data[col].median()
    data[col].fillna(median_val, inplace=True)

data = pd.get_dummies(data, columns=['HomePlanet', 'Destination'], drop_first=True)

data['Age'] = scaler.fit_transform(data[['Age']])

missing_values = data.isnull().sum()
print(missing_values)
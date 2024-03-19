import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
# Étape 1: Lire les fichiers .txt
# Adaptez les noms de fichiers et les chemins selon vos données
file_paths = ['lap1.txt','lap2.txt']
""" for folder_path in folders_path:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file)) """
            
list_of_dataframes = []
for file_path in file_paths:
    df_temp = pd.read_csv(file_path, sep=",", low_memory=False)  # Adaptez le séparateur si nécessaire
    list_of_dataframes.append(df_temp)

# Étape 2: Fusionner les DataFrames
df_merged = pd.concat(list_of_dataframes, ignore_index=True)
df_merged.columns = [col.strip() for col in df_merged.columns]  # Clean column names
print(df_merged.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1083336 entries, 0 to 1083335
Data columns (total 17 columns):
 #   Column                 Non-Null Count    Dtype  
---  ------                 --------------    -----  
 0   #                      1040547 non-null  float64
 1   Time in Sec            1040547 non-null  float64
 2   Time in usec           1040547 non-null  float64
 3   Bytes sent             1040547 non-null  float64
 4   Sender latitude        1040547 non-null  object
 5   Sender longitude       1040547 non-null  object
 6   Sender Speed(km/h)     1040547 non-null  float64
 7   Sender Altitude(m)     1040547 non-null  float64
 8   Receiver latitude      1040545 non-null  object
 9   Receiver Longitude     1040545 non-null  object
 10  Receiver Speed         1040545 non-null  float64
 11  Receiver Altitude(m)   1040545 non-null  float64
 12  Packet Received (Y/N)  1040247 non-null  object
 13  Bytes Received         1040247 non-null  float64
 14  Signal Strength        1040247 non-null  float64
 15  Noise Strength         1040247 non-null  float64
 16  Direction: Northbound  42789 non-null    object
dtypes: float64(11), object(6)
memory usage: 140.5+ MB
None
"""
# Objective : Predict the 'Bytes Sent' column using GRU model
 # Étape 3: Prétraitement des données
print(df_merged.head())
print(df_merged.shape)
""" def convert_coordinate(coord):
    if isinstance(coord, str):
        direction = coord[-1]
        if direction in ['S', 'W', 'E', 'N']:
            coord = -float(coord[:-1])
    return coord

df_merged['Sender latitude'] = df_merged['Sender latitude'].apply(convert_coordinate)
df_merged['Sender longitude'] = df_merged['Sender longitude'].apply(convert_coordinate)
df_merged['Receiver latitude'] = df_merged['Receiver latitude'].apply(convert_coordinate)
df_merged['Receiver Longitude'] = df_merged['Receiver Longitude'].apply(convert_coordinate) """
# Supprimer les colonnes inutiles de latitude et longitude
df_merged = df_merged.drop(['Noise Strength','Sender latitude', 'Sender longitude', 'Receiver latitude', 'Receiver Longitude', ], axis=1)

df_merged['Packet Received (Y/N)'] = df_merged['Packet Received (Y/N)'].map({'Y': 1, 'N': 0})
#df_merged['Direction: Northbound'] = df_merged['Direction: Northbound'].map({'N': 1, 'S': 0})
# Normaliser les données
scaler = MinMaxScaler()
df_merged = pd.DataFrame(scaler.fit_transform(df_merged), columns=df_merged.columns)
# Étape 4: Préparer les données pour le modèle
# Séparer les données en entrées et sorties
X = df_merged.drop('Bytes sent', axis=1)
y = df_merged['Bytes sent']
# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convertir les DataFrames en tableaux numpy
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
print('df_merged : \n',df_merged.head())
# Convert y_test to a numpy array
y_test = y_test.values
# Étape 5: Créer et entraîner le modèle
# Créer le modèle
model = Sequential()
model.add(GRU(50, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(GRU(50, return_sequences=True))
model.add(GRU(50))
model.add(Dense(1))

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Adapter le modèle
model.fit(X_train, y_train, epochs=10, batch_size=64)
# Créer le modèle
model = Sequential()
model.add(GRU(50, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(GRU(50, return_sequences=True))
model.add(GRU(50))
model.add(Dense(1))
# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
# Adapter le modèle
model.fit(X_train, y_train, epochs=10, batch_size=64)
# Étape 6: Évaluer le modèle
# Prédire les valeurs de test
y_pred = model.predict(X_test)
# Calculer l'erreur quadratique moyenne
mse = np.mean((y_pred - y_test)**2)
#mse = np.mean(np.square(np.subtract(y_test, y_pred)))
print(f'Mean Squared Error: {mse}')
# plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(14,7))
plt.plot(y_test, color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import random
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import joblib

# Cargar el dataset generado
df = pd.read_excel('nuevo_dataset_tld_semi(d).xlsx')

# Parámetros del espectro a interpolar
num_puntos_interp = 50
energias_interp = np.linspace(df['E'].min(), df['E'].max(), num_puntos_interp)

# Agrupar por espectro completo (cada conjunto de parámetros)
grupos = df.groupby(['A', 'E0', 'C', 'Eg', 'eps_inf', 'd_film', 'gamma', 'wp'])
X = []
y = []

for (A, E0, C, Eg, eps_inf, d_film, gamma, wp), group in grupos:
    group_sorted = group.sort_values('E')
    if group_sorted['E'].nunique() < 2:
        continue  # No se puede interpolar con menos de 2 puntos
    interp_psi = interp1d(group_sorted['E'], group_sorted['psi (deg)'], kind='linear', fill_value="extrapolate")
    interp_delta = interp1d(group_sorted['E'], group_sorted['delta (deg)'], kind='linear', fill_value="extrapolate")
    espectro = np.concatenate([interp_psi(energias_interp), interp_delta(energias_interp)])
    X.append(espectro)
    y.append([A, C, Eg, E0, eps_inf, d_film, gamma, wp])

X = np.array(X)
y = np.array(y)

# Normalización
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Crear etiquetas discretas para StratifiedKFold (basado en binning de E0)
etiquetas_estrato = pd.qcut(y[:, 3], q=10, labels=False)

# Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []
histories = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, etiquetas_estrato)):
    print(f"\nEntrenando fold {fold + 1}/{n_splits}")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

    # Modelo
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(8)  # A, C, Eg, E0, eps_inf, gamma, wp
    ])

    model.compile(optimizer='adam', loss='mse')
    # Callback de EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Entrenamiento
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    histories.append(history)
    # Evaluación
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

# Comparativa final
y_true_full = np.vstack(all_y_true)
y_pred_full = np.vstack(all_y_pred)


mae_final = mean_absolute_error(y_true_full, y_pred_full, multioutput='raw_values')
print("\nMAE final por parámetro:", mae_final)
rmse_final = np.sqrt(mean_squared_error(y_true_full, y_pred_full, multioutput='raw_values'))
print("\nRMSE final por parámetro:", rmse_final)

# Guardar modelo final
model.save('nuevo_modelo_tld_10000.h5')
joblib.dump(scaler_X, 'nuevo_scaler_X_nn_tld.pkl')
joblib.dump(scaler_y, 'nuevo_scaler_y_nn_tld.pkl')


theta_i = 70


# Crear DataFrame con columnas intercaladas real/predicho
columnas = ['A', 'C', 'Eg', 'E0', 'eps_inf', 'd_film', 'gamma', 'wp']
data = {}

for i, nombre in enumerate(columnas):
    data[f'{nombre}_real'] = y_true_full[:, i]
    data[f'{nombre}_pred'] = y_pred_full[:, i]

df_resultados = pd.DataFrame(data)

# Guardar como archivo Excel
df_resultados.to_excel('predicciones_nn_tld.xlsx', index=False)
print("Archivo Excel guardado como 'predicciones_nn_tld.xlsx'")

# MAE plot
parametros = ['A', 'C', 'Eg', 'E0', 'eps_inf', 'd_film', 'gamma', 'wp']
plt.figure(figsize=(8, 5))
plt.bar(parametros, mae_final)
plt.title('MAE por parámetro')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# RMSE plot
plt.figure(figsize=(8, 5))
plt.bar(parametros, rmse_final)
plt.title('RMSE por parámetro')
plt.ylabel('RMSE')
plt.xlabel('Parámetros')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Loss plot promedio
min_epochs = min(len(h.history['loss']) for h in histories)
avg_train = np.mean([h.history['loss'][:min_epochs] for h in histories], axis=0)
avg_val = np.mean([h.history['val_loss'][:min_epochs] for h in histories], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(avg_train, label='Train Loss Promedio', lw=2)
plt.plot(avg_val, label='Val Loss Promedio', lw=2, linestyle='--')
plt.title('Curvas de Loss Promedio')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from scipy.interpolate import interp1d
from drude import calcular_psi_delta_semi_tld
import matplotlib.pyplot as plt
import joblib


# --- Callback personalizado para mostrar progreso global ---
class GlobalProgressCallback(Callback):
    def __init__(self, fold_idx, total_folds, total_epochs):
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        self.total_epochs = total_epochs
        self.epochs_completed = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_completed = epoch + 1
        total_epochs_done = self.fold_idx * self.total_epochs + self.epochs_completed
        total_epochs_all = self.total_folds * self.total_epochs
        percent = 100 * total_epochs_done / total_epochs_all
        print(f"\rProgreso total entrenamiento: {percent:.1f}% (Fold {self.fold_idx+1}, Época {epoch+1}/{self.total_epochs})", end='')

# --- Cargar dataset ---
df = pd.read_excel('nuevo_dataset_tld_semi(d).xlsx')

num_puntos_interp = 50
energias_interp = np.linspace(df['E'].min(), df['E'].max(), num_puntos_interp)

grupos = df.groupby(['A', 'E0', 'C', 'Eg', 'eps_inf', 'd_film', 'gamma', 'wp'])
X, y = [], []
precomputed_spectra = {}

for (A, E0, C, Eg, eps_inf, d_film, gamma, wp), group in grupos:
    group_sorted = group.sort_values('E')
    if group_sorted['E'].nunique() < 2:
        continue
    interp_psi = interp1d(group_sorted['E'], group_sorted['psi (deg)'], kind='linear', fill_value="extrapolate")
    interp_delta = interp1d(group_sorted['E'], group_sorted['delta (deg)'], kind='linear', fill_value="extrapolate")
    espectro = np.concatenate([interp_psi(energias_interp), interp_delta(energias_interp)])
    X.append(espectro)
    params = (A, C, Eg, E0, eps_inf, d_film, gamma, wp)
    y.append(params)
    # Usa la función calcular_psi_delta_semi para generar espectros teóricos y almacenar en precomputed_spectra (acelerar el cálculo de la pérdida física).
    df_model = calcular_psi_delta_semi_tld(A, E0, C, Eg, eps_inf, d_film, gamma, wp, theta_0=70, E_values=energias_interp)
    precomputed_spectra[params] = np.concatenate([
        df_model['psi (deg)'].values,
        df_model['delta (deg)'].values
    ])

X, y = np.array(X), np.array(y)

# --- Escalado ---
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_scaled, y_scaled = scaler_X.fit_transform(X), scaler_y.fit_transform(y)

etiquetas_estrato = pd.qcut(y[:, 3], q=10, labels=False)

# --- Scheduler de LR y función de pérdida PINN ---
lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)

def pinn_loss(X_true_full):
    def loss(y_true, y_pred):
        loss_param = tf.reduce_mean(tf.square(y_true - y_pred))
        # Reconstruye el espectro teórico (ψ, Δ) a partir de los parámetros predichos
        def reconstruct_spectra(y_pred_batch):
            outputs = []
            for row in y_pred_batch:
                key = tuple(np.round(row, decimals=6))  # clave redondeada para evitar problemas de precisión
                if key in precomputed_spectra:
                    outputs.append(precomputed_spectra[key])
                else:
                    outputs.append(np.zeros_like(X_true_full[0]))  # en caso de error
            return tf.convert_to_tensor(outputs, dtype=tf.float32)

        reconstructed = tf.numpy_function(reconstruct_spectra, [y_pred], tf.float32)
        reconstructed.set_shape([None, X_true_full.shape[1]])
        X_true_tensor = tf.convert_to_tensor(X_true_full, dtype=tf.float32)
        X_true_batch = X_true_tensor[:tf.shape(y_pred)[0]]
        loss_phys = tf.reduce_mean(tf.square(X_true_batch - reconstructed))
        return loss_param + loss_phys

    return loss

# --- Entrenamiento ---
total_folds = 5
skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)  
all_y_true, all_y_pred, histories = [], [], []

total_epochs = 200  
batch_size = 64     

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, etiquetas_estrato)):
    print(f"\nFold {fold+1}/{total_folds}")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

    input_layer = Input(shape=(X_train.shape[1],))
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    output_layer = layers.Dense(8)(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=pinn_loss(X_train))

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    progress = GlobalProgressCallback(fold_idx=fold, total_folds=total_folds, total_epochs=total_epochs)

    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs=total_epochs, batch_size=batch_size,
                        callbacks=[early_stop, progress], verbose=0)

    print()  # salto de línea después del progreso

    histories.append(history)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    all_y_true.append(y_true)
    all_y_pred.append(y_pred)

# --- MAE y gráfica ---
y_true_full = np.vstack(all_y_true)
y_pred_full = np.vstack(all_y_pred)
mae_final = mean_absolute_error(y_true_full, y_pred_full, multioutput='raw_values')
print("\nMAE final por parámetro:", mae_final)
rmse_final = np.sqrt(mean_squared_error(y_true_full, y_pred_full, multioutput='raw_values'))
print("\nRMSE final por parámetro:", rmse_final)

model.save('nuevo_pinn_modelo_tld_semi.h5')
joblib.dump(scaler_X, 'scaler_X_tld.pkl')
joblib.dump(scaler_y, 'scaler_y_tld.pkl')

columnas = ['A', 'C', 'Eg', 'E0', 'eps_inf', 'd_film', 'gamma', 'wp']
data = {f'{nombre}_real': y_true_full[:, i] for i, nombre in enumerate(columnas)}
data.update({f'{nombre}_pred': y_pred_full[:, i] for i, nombre in enumerate(columnas)})
df_resultados = pd.DataFrame(data)
df_resultados.to_excel('predicciones_pinn_tld_semi.xlsx', index=False)
print("Archivo Excel guardado como 'predicciones_pinn_tld_semi.xlsx'")


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

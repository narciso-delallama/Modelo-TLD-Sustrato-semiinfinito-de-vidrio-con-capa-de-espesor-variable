import numpy as np
import pandas as pd
from drude import calcular_psi_delta_semi_tld
from tqdm import tqdm



# === GENERAR LOS 100 ESPECTROS ===
n_spectra = 20000
# Valores de energía para el dataset
dataset = []

for spectrum_id in range(n_spectra):
    while True:
        A = np.random.uniform(50, 350)
        E0 = np.random.uniform(1.0, 5.0)
        C = np.random.uniform(0.5, 5.0)
        Eg = np.random.uniform(1.0, 5.0)
        eps_inf = np.random.uniform(1.0, 3.0)
        d_film = np.random.uniform(1, 250)
        gamma = np.random.uniform(0.1, 4)
        wp = np.random.uniform(10.0, 20.0)

        # --- FILTROS DE SEGURIDAD ---
        if E0 <= C / np.sqrt(2):
            continue  # Evita raíz negativa en gamma
        if 4 * E0**2 <= C**2:
            continue  # Evita raíz negativa en alpha
        if E0 == 0 or C == 0:
            continue  # Evita divisiones por cero
        if E0 <= C / np.sqrt(2) or E0 < 1.0:
            continue
        break  # Si pasa todos los filtros, salimos del while

    # Calcular espectro
    df_temp = calcular_psi_delta_semi_tld(A, E0, C, Eg, eps_inf, d_film, gamma, wp, 70)
    df_temp['A'] = A
    df_temp['E0'] = E0
    df_temp['C'] = C
    df_temp['Eg'] = Eg
    df_temp['eps_inf'] = eps_inf
    df_temp['d_film'] = d_film
    df_temp['gamma'] = gamma
    df_temp['wp'] = wp
    dataset.append(df_temp)

# Unir todo en un único DataFrame
df_final = pd.concat(dataset, ignore_index=True)

# Reordenar columnas
column_order = ['A', 'C', 'E0', 'Eg', 'eps_inf', 'd_film', 'gamma', 'wp', 'E', 'psi (deg)', 'delta (deg)']
df_final = df_final[column_order]

# Guardar a CSV (opcional)
df_final.to_excel("nuevo_dataset_tld_semi(d).xlsx", index=False)
df_final.to_csv("_nuevo_dataset_tld_semi(d).csv", index=False)
print("Dataset generado y guardado exitosamente.")


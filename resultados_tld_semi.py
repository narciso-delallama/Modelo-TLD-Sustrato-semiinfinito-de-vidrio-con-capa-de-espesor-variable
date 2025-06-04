import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from drude import calcular_psi_delta_semi_tld, tl_drude_eps1, tl_drude_eps2
import elli.kkr.kkr as kkr
from sklearn.metrics import mean_squared_error

# Cargar datos
df_nn = pd.read_excel('predicciones_nn_tld.xlsx')
df_pinn = pd.read_excel('predicciones_pinn_tld_semi.xlsx')

# Elegir índices aleatorios
np.random.seed(43)
num_indices = 3
indices = np.random.choice(len(df_pinn), num_indices, replace=False)

theta_i = 70
E = np.linspace(0.5, 6.5, 100)

# --- Parámetros visuales ---
label_fontsize = 15
title_fontsize = 22
legend_fontsize = 15
tick_size = 13
bar_width = 0.35
line_width = 2
grid_visible = True

# ---------- GRÁFICAS PSI / DELTA ----------
fig, axs = plt.subplots(num_indices, 2, figsize=(10, 3.5 * num_indices))
if num_indices == 1: axs = np.expand_dims(axs, axis=0)

for row_idx, idx in enumerate(indices):
    fila_nn = df_nn.iloc[idx]
    fila_pinn = df_pinn.iloc[idx]

    df_real = calcular_psi_delta_semi_tld(fila_nn['A_real'], fila_nn['E0_real'], fila_nn['C_real'],
                                          fila_nn['Eg_real'], fila_nn['eps_inf_real'], fila_nn['d_film_real'],
                                          fila_nn['gamma_real'], fila_nn['wp_real'], theta_i)
    df_pred_nn = calcular_psi_delta_semi_tld(fila_nn['A_pred'], fila_nn['E0_pred'], fila_nn['C_pred'],
                                             fila_nn['Eg_pred'], fila_nn['eps_inf_pred'], fila_nn['d_film_pred'],
                                             fila_nn['gamma_pred'], fila_nn['wp_pred'], theta_i)
    df_pred_pinn = calcular_psi_delta_semi_tld(fila_pinn['A_pred'], fila_pinn['E0_pred'], fila_pinn['C_pred'],
                                               fila_pinn['Eg_pred'], fila_pinn['eps_inf_pred'], fila_pinn['d_film_pred'],
                                               fila_pinn['gamma_pred'], fila_pinn['wp_pred'], theta_i)

    axs[row_idx, 0].plot(df_real['E'], df_real['psi (deg)'], label='Psi Real', color='blue')
    axs[row_idx, 0].plot(df_pred_nn['E'], df_pred_nn['psi (deg)'], '--', label='Psi NN', color='orange')
    axs[row_idx, 0].plot(df_pred_pinn['E'], df_pred_pinn['psi (deg)'], '--', label='Psi PINN', color='green')
    axs[row_idx, 0].set_ylabel(r'$\Psi$ (°)', fontsize=label_fontsize)
    axs[row_idx, 0].legend(fontsize=legend_fontsize)
    axs[row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 0].grid(True)

    axs[row_idx, 1].plot(df_real['E'], df_real['delta (deg)'], label='Delta Real', color='blue')
    axs[row_idx, 1].plot(df_pred_nn['E'], df_pred_nn['delta (deg)'], '--', label='Delta NN', color='orange')
    axs[row_idx, 1].plot(df_pred_pinn['E'], df_pred_pinn['delta (deg)'], '--', label='Delta PINN', color='green')
    axs[row_idx, 1].set_ylabel(r'$\Delta$ (°)', fontsize=label_fontsize)
    axs[row_idx, 1].legend(fontsize=legend_fontsize)
    axs[row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 1].grid(True)

plt.tight_layout()
plt.show()


# ---------- RMSE por Parámetro ----------
param_grupo1 = ['E0', 'C', 'Eg', 'eps_inf', 'gamma', 'wp']
param_grupo2 = ['A', 'd_film']
etiquetas1 = [r'$E_0$', r'$C$', r'$E_g$', r'$\varepsilon_\infty$', r'$\Gamma$/eV', r'$\omega_p$/Hz']
etiquetas2 = [r'$A$', r'$d_{film}$']

fig, axs = plt.subplots(num_indices, 2, figsize=(14, 4 * num_indices))
if num_indices == 1: axs = np.expand_dims(axs, axis=0)

for row_idx, idx in enumerate(indices):
    fila_nn = df_nn.iloc[idx]
    fila_pinn = df_pinn.iloc[idx]

    rmse_nn_1, rmse_pinn_1 = [], []
    rmse_nn_2, rmse_pinn_2 = [], []

    for param in param_grupo1:
        rmse_nn_1.append(np.sqrt(mean_squared_error([fila_nn[f'{param}_real']], [fila_nn[f'{param}_pred']])))
        rmse_pinn_1.append(np.sqrt(mean_squared_error([fila_pinn[f'{param}_real']], [fila_pinn[f'{param}_pred']])))

    for param in param_grupo2:
        rmse_nn_2.append(np.sqrt(mean_squared_error([fila_nn[f'{param}_real']], [fila_nn[f'{param}_pred']])))
        rmse_pinn_2.append(np.sqrt(mean_squared_error([fila_pinn[f'{param}_real']], [fila_pinn[f'{param}_pred']])))

    x1 = np.arange(len(param_grupo1))
    x2 = np.arange(len(param_grupo2))

    axs[row_idx, 0].bar(x1 - bar_width/2, rmse_nn_1, bar_width, label='NN', color='lightblue', linewidth=line_width)
    axs[row_idx, 0].bar(x1 + bar_width/2, rmse_pinn_1, bar_width, label='PINN', color='lightgreen', linewidth=line_width)
    axs[row_idx, 0].set_xticks(x1)
    axs[row_idx, 0].set_xticklabels(etiquetas1, fontsize=tick_size)
    axs[row_idx, 0].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 0].legend()
    axs[row_idx, 0].grid(grid_visible)

    axs[row_idx, 1].bar(x2 - bar_width/2, rmse_nn_2, bar_width, label='NN', color='lightblue', linewidth=line_width)
    axs[row_idx, 1].bar(x2 + bar_width/2, rmse_pinn_2, bar_width, label='PINN', color='lightgreen', linewidth=line_width)
    axs[row_idx, 1].set_xticks(x2)
    axs[row_idx, 1].set_xticklabels(etiquetas2, fontsize=tick_size)
    axs[row_idx, 1].tick_params(axis='both', labelsize=tick_size)
    axs[row_idx, 1].legend()
    axs[row_idx, 1].grid(grid_visible)

plt.tight_layout()
plt.show()

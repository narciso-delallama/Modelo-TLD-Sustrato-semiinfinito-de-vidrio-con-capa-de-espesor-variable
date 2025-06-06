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
label_fontsize = 20    # tamaño etiquetas ejes
title_fontsize = 22    # tamaño título
legend_fontsize = 15   # tamaño leyenda
tick_size = 20         # tamaño de los números en ejes

# ---------- GRÁFICAS PSI / DELTA ----------
fig, axs = plt.subplots(nrows=num_indices * 2, ncols=2, figsize=(13, 6.25 * num_indices))
axs = np.array(axs).reshape(num_indices * 2, 2)

for i, idx in enumerate(indices):
    fila_nn = df_nn.iloc[idx]
    fila_pinn = df_pinn.iloc[idx]

    # ----------- Espectros Psi / Delta (fila 2i) -----------
    df_real = calcular_psi_delta_semi_tld(
        fila_nn['A_real'], fila_nn['E0_real'], fila_nn['C_real'], fila_nn['Eg_real'],
        fila_nn['eps_inf_real'], fila_nn['d_film_real'], fila_nn['gamma_real'],
        fila_nn['wp_real'], theta_i
    )
    df_pred_nn = calcular_psi_delta_semi_tld(
        fila_nn['A_pred'], fila_nn['E0_pred'], fila_nn['C_pred'], fila_nn['Eg_pred'],
        fila_nn['eps_inf_pred'], fila_nn['d_film_pred'], fila_nn['gamma_pred'],
        fila_nn['wp_pred'], theta_i
    )
    df_pred_pinn = calcular_psi_delta_semi_tld(
        fila_pinn['A_pred'], fila_pinn['E0_pred'], fila_pinn['C_pred'], fila_pinn['Eg_pred'],
        fila_pinn['eps_inf_pred'], fila_pinn['d_film_pred'], fila_pinn['gamma_pred'],
        fila_pinn['wp_pred'], theta_i
    )

    axs[2*i, 0].plot(df_real['E'], df_real['psi (deg)'], label='Psi Real', color='blue')
    axs[2*i, 0].plot(df_pred_nn['E'], df_pred_nn['psi (deg)'], '--', label='Psi NN', color='orange')
    axs[2*i, 0].plot(df_pred_pinn['E'], df_pred_pinn['psi (deg)'], '--', label='Psi PINN', color='green')
    axs[2*i, 0].set_ylabel(r'$\Psi$ (°)', fontsize=label_fontsize)
    axs[2*i, 0].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[2*i, 0].tick_params(axis='both', labelsize=tick_size)
    axs[2*i, 0].legend(fontsize=legend_fontsize)
    axs[2*i, 0].grid(True)

    axs[2*i, 1].plot(df_real['E'], df_real['delta (deg)'], label='Delta Real', color='blue')
    axs[2*i, 1].plot(df_pred_nn['E'], df_pred_nn['delta (deg)'], '--', label='Delta NN', color='orange')
    axs[2*i, 1].plot(df_pred_pinn['E'], df_pred_pinn['delta (deg)'], '--', label='Delta PINN', color='green')
    axs[2*i, 1].set_ylabel(r'$\Delta$ (°)', fontsize=label_fontsize)
    axs[2*i, 1].set_xlabel('Energía (eV)', fontsize=label_fontsize)
    axs[2*i, 1].tick_params(axis='both', labelsize=tick_size)
    axs[2*i, 1].legend(fontsize=legend_fontsize)
    axs[2*i, 1].grid(True)

    # ----------- RMSE por Parámetro (fila 2i+1) -----------
    param_grupo1 = ['E0', 'C', 'Eg', 'eps_inf', 'gamma', 'wp']
    param_grupo2 = ['A', 'd_film']
    etiquetas1 = [r'$E_0$/eV', r'$C$/eV', r'$E_g$/eV', r'$\varepsilon_\infty$', r'$\Gamma$/eV', r'$\omega_p$/Hz']
    etiquetas2 = [r'$A$', r'$d_{film}$/nm']

    rmse_nn_1 = [np.sqrt(mean_squared_error([fila_nn[f'{p}_real']], [fila_nn[f'{p}_pred']])) for p in param_grupo1]
    rmse_pinn_1 = [np.sqrt(mean_squared_error([fila_pinn[f'{p}_real']], [fila_pinn[f'{p}_pred']])) for p in param_grupo1]
    rmse_nn_2 = [np.sqrt(mean_squared_error([fila_nn[f'{p}_real']], [fila_nn[f'{p}_pred']])) for p in param_grupo2]
    rmse_pinn_2 = [np.sqrt(mean_squared_error([fila_pinn[f'{p}_real']], [fila_pinn[f'{p}_pred']])) for p in param_grupo2]

    x1 = np.arange(len(param_grupo1))
    x2 = np.arange(len(param_grupo2))

    axs[2*i+1, 0].bar(x1 - 0.15, rmse_nn_1, 0.3, label='NN', color='lightblue')
    axs[2*i+1, 0].bar(x1 + 0.15, rmse_pinn_1, 0.3, label='PINN', color='lightgreen')
    for j, p in enumerate(param_grupo1):
        valor_real = fila_nn[f'{p}_real']
        altura = max(rmse_nn_1[j], rmse_pinn_1[j]) + 0.01
        axs[2*i+1, 0].text(x1[j], altura, f'{valor_real:.2f}', 
                        ha='center', va='bottom', fontsize=15)

    axs[2*i+1, 0].set_xticks(x1)
    axs[2*i+1, 0].set_xticklabels(etiquetas1, fontsize=tick_size)
    axs[2*i+1, 0].set_ylabel('RMSE', fontsize=label_fontsize)
    axs[2*i+1, 0].tick_params(axis='both', labelsize=tick_size)
    axs[2*i+1, 0].legend(fontsize=legend_fontsize)
    axs[2*i+1, 0].grid(True)

    axs[2*i+1, 1].bar(x2 - 0.15, rmse_nn_2, 0.3, label='NN', color='lightblue')
    axs[2*i+1, 1].bar(x2 + 0.15, rmse_pinn_2, 0.3, label='PINN', color='lightgreen')
    for j, p in enumerate(param_grupo2):
        valor_real = fila_nn[f'{p}_real']
        altura = max(rmse_nn_2[j], rmse_pinn_2[j]) + 0.01
        axs[2*i+1, 1].text(x2[j], altura, f'{valor_real:.2f}', 
                        ha='center', va='bottom', fontsize=15)

    axs[2*i+1, 1].set_xticks(x2)
    axs[2*i+1, 1].set_xticklabels(etiquetas2, fontsize=tick_size)
    axs[2*i+1, 1].set_ylabel('RMSE', fontsize=label_fontsize)
    axs[2*i+1, 1].tick_params(axis='both', labelsize=tick_size)
    axs[2*i+1, 1].legend(fontsize=legend_fontsize)
    axs[2*i+1, 1].grid(True)

plt.tight_layout(h_pad=3.0)
plt.savefig('tld_spectros_rmse.png', dpi=300, bbox_inches='tight')
plt.show()

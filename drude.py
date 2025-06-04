import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Tauc_Lorentz import tauc_lorentz_eps1, tauc_lorentz_eps2
from tmm_core import ellips

def drude_eps1(eps_inf, gamma, wp, E):
    """
    Calcula la parte real de la permitividad compleja usando el modelo de Drude.
    
    Parámetros:
    eps_inf : float
        Permitividad en el infinito.
    gamma : float
        Ancho de la banda de absorción.
    wp : float
        Frecuencia plasmónica.
    E : array_like
        Energía en eV.
    
    Retorna:
    eps1 : array_like
        Parte real de la permitividad compleja.
    """
    eps1 = eps_inf - (wp**2 / (E**2 + gamma**2))
    return eps1

def tl_drude_eps1(eps_inf, gamma, wp, E, A, E0, C, Eg):
    """
    Calcula la parte real de la permitividad compleja usando el modelo de Drude.
    
    Parámetros:
    eps_inf : float
        Permitividad en el infinito.
    gamma : float
        Ancho de la banda de absorción.
    wp : float
        Frecuencia plasmónica.
    E : array_like
        Energía en eV.
    
    Retorna:
    eps1 : array_like
        Parte real de la permitividad compleja.
    """
    eps1 = tauc_lorentz_eps1(E, A, E0, C, Eg) + eps_inf - (wp**2 / (E**2 + gamma**2))
    return eps1

def drude_eps2(gamma, wp, E):   
    """
    Calcula la parte imaginaria de la permitividad compleja usando el modelo de Drude.
    
    Parámetros:
    gamma : float
        Ancho de la banda de absorción.
    wp : float
        Frecuencia plasmónica.
    E : array_like
        Energía en eV.
    
    Retorna:
    eps2 : array_like
        Parte imaginaria de la permitividad compleja.
    """
    eps2 = (wp**2 * gamma) / (E * (E**2 + gamma**2))
    return eps2

def tl_drude_eps2(gamma, wp, E, A, E0, C, Eg):
    """
    Calcula la parte imaginaria de la permitividad compleja usando el modelo de Drude.
    
    Parámetros:
    gamma : float
        Ancho de la banda de absorción.
    wp : float
        Frecuencia plasmónica.
    E : array_like
        Energía en eV.
    
    Retorna:
    eps2 : array_like
        Parte imaginaria de la permitividad compleja.
    """
    eps2 = tauc_lorentz_eps2(E, A, E0, C, Eg) + (wp**2 * gamma) / (E * (E**2 + gamma**2))
    return eps2

def tld_calculo_n_k(eps1, eps2):
    n = np.sqrt((eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)
    k = np.sqrt((-eps1 + np.sqrt(eps1**2 + eps2**2)) / 2)
    return n, k

def tld_generar_nyk(A, E0, C, Eg, eps_inf, gamma, wp, Emin=0.5, Emax=6.5, points=1000):
    E = np.linspace(Emin, Emax, points)
    eps2 = tl_drude_eps2(gamma, wp, E, A, E0, C, Eg)
    eps1 = tl_drude_eps1(eps_inf, gamma, wp, E, A, E0, C, Eg)
    n, k = tld_calculo_n_k(eps1, eps2)
    return E, n, k, eps1, eps2

def calcular_psi_delta_semi_tld(A, E0, C, Eg, eps_inf, d_film, gamma, wp, theta_0, E_values=np.linspace(0.5, 6.5, 50)):
    th_0 = np.radians(theta_0)
    n_air = 1.0
    n_substrate = 1.5
    inf = float('inf')

    if E_values is None:
        E_values = np.linspace(0.5, 6.5, 50)

    if theta_0 is None:
        theta_0 = 70

    eps2 = tl_drude_eps2(gamma, wp, E_values, A, E0, C, Eg)
    eps1 = tl_drude_eps1(eps_inf, gamma, wp, E_values, A, E0, C, Eg)
    n_vals, k_vals = tld_calculo_n_k(eps1, eps2)

    resultados = []

    for i in range(len(E_values)):
        n_complex = n_vals[i] + 1j * k_vals[i]
        n_list = [n_air, n_complex, n_substrate]
        d_list = [inf, d_film, inf]
        lam_vac = 1239.84193 / E_values[i]  # Convertir eV a nm

        try:
            result = ellips(n_list, d_list, th_0, lam_vac)
            psi = np.degrees(result['psi'])
            delta = np.degrees(-result['Delta']+np.pi)
        except Exception:
            psi = np.nan
            delta = np.nan
        
        resultados.append({
            'E': E_values[i],
            'n': n_vals[i],
            'k': k_vals[i],
            'psi (deg)': psi,
            'delta (deg)': delta
        })
    return pd.DataFrame(resultados)





#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calibración completa con propagación de errores

• Ajusta n picos Lorentz a cada espectro.
• Calcula suma S y su σ_S por propagación (gradᵀ·pcov·grad).
• Ajuste lineal S = m·cal + b con σ_m y σ_b.
• Guarda diccionario {'slope', 'intercept', 'slope_err',
                      'intercept_err', 'r2'} en un .pkl.
Autor: ChatGPT (o3) — 2025-06-09
"""

import os, re, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- Modelo Lorentziano ---------- #
def lorentzian(x, amp, cen, wid):
    return (2*amp/np.pi) * (wid / (4*(x-cen)**2 + wid**2))

def multiple_lorentzians(x, *p):
    n = len(p)//3
    out = np.zeros_like(x)
    for i in range(n):
        a, c, w = p[3*i:3*i+3]
        out += lorentzian(x, a, c, w)
    return out

# ---------- Lectura de espectro ---------- #
HEADER_LINES = 19
X_MIN, X_MAX = 650.0, 710.0   # rango a conservar

def read_spectrum(path):
    wl, od = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f.readlines()[HEADER_LINES:]:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.replace(',', '.').split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if X_MIN <= x <= X_MAX:
                wl.append(x)
                od.append(y)
    return np.array(wl), np.array(od)

# ---------- Valor de calibración del nombre ---------- #
def extract_cal_value(fname):
    m = re.search(r'#(\d+\.?\d*)#', fname)
    if m:
        return float(m.group(1))
    m2 = re.search(r'(\d+\.?\d*)', fname)
    return float(m2.group(1)) if m2 else None

# ---------- Ajuste de Lorentzianas y error de la suma ---------- #
def fit_lorentzians(wl, od, n_peaks=5, maxfev=500000):
    # estimaciones iniciales sencillas
    amp0 = od.max()/n_peaks
    c0s  = np.linspace(wl.min(), wl.max(), n_peaks+2)[1:-1]
    w0   = (wl.max()-wl.min())/(4*n_peaks)
    p0 = np.ravel([[amp0, c, w0] for c in c0s])
    popt, pcov = curve_fit(multiple_lorentzians, wl, od,
                           p0=p0, maxfev=maxfev)
    return popt, pcov

def lorentz_sum_and_error(wl, popt, pcov, eps=1e-6):
    """
    Devuelve (S, σ_S) donde S = Σ_i L(wl_i; popt)
    """
    y_hat = multiple_lorentzians(wl, *popt)
    S = y_hat.sum()

    # gradiente numérico de S respecto a cada parámetro
    grad = np.zeros_like(popt)
    for i in range(len(popt)):
        dp = np.zeros_like(popt); dp[i] = eps
        S_plus  = multiple_lorentzians(wl, *(popt+dp)).sum()
        S_minus = multiple_lorentzians(wl, *(popt-dp)).sum()
        grad[i] = (S_plus - S_minus)/(2*eps)

    var_S = grad @ pcov @ grad
    sigma_S = np.sqrt(var_S) if var_S > 0 else np.nan
    return S, sigma_S

# ---------- Calibración global ---------- #
def process_folder(folder, n_peaks=2, sigma_y=0.01):
    spectra = []
    sums, sums_err, cals = [], [], []

    plt.figure(figsize=(12,7))
    for fp in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        wl, od = read_spectrum(fp)
        if wl.size == 0:
            continue
        cal = extract_cal_value(os.path.basename(fp))
        if cal is None:
            continue

        popt, pcov = fit_lorentzians(wl, od, n_peaks)
        S, S_err   = lorentz_sum_and_error(wl, popt, pcov)

        spectra.append((wl, od, fp))
        cals.append(cal)
        sums.append(S)
        sums_err.append(S_err)

        plt.plot(wl, od, alpha=.6,
                 label=f"{os.path.basename(fp)}  (cal={cal})")

    plt.axvline(663, color='k', ls='--')
    plt.title("Espectros de calibración")
    plt.xlabel("λ (nm)"); plt.ylabel("OD")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.show()

    return (np.array(cals),
            np.array(sums),
            np.array(sums_err))

# ---------- Regresión lineal con covarianza ---------- #
def linear_regression(x, y):
    (m, b), cov = np.polyfit(x, y, 1, cov=True)
    m_err, b_err = np.sqrt(np.diag(cov))

    y_pred = m*x + b
    r2 = 1 - np.sum((y - y_pred)**2)/np.sum((y - y.mean())**2)

    # --- Configuración estética científica con LaTeX ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # --- Gráfico ---
    xs = np.linspace(x.min(), x.max(), 200)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=120)  # proporción 3:2

    # Datos con marcador de círculo blanco y borde negro
    ax.errorbar(
        x, y, yerr=None, fmt='o',
        mfc='white', mec='black', color='black',
        markersize=6, capsize=3,
        label=r'Datos'
    )

    # Recta de ajuste
    ax.plot(
        xs, m*xs + b,
        color='red', lw=1.2,
        label=r'Ajuste lineal'
    )

    # Etiquetas de ejes
    ax.set_xlabel(r'Dosis (Gy)')
    ax.set_ylabel(r'Suma de Lorentzianas')

    # Ticks y rejilla
    ax.set_xticks(np.arange(0, 22.1, 2.0))
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.5)

    # Anotación con parámetros de la recta
    texto = (
        rf"$y = ({m:.4f}\pm{m_err:.4f})\,x + ({b:.4f}\pm{b_err:.4f})$" "\n"
        rf"$R^2 = {r2:.4f}$"
    )
    ax.annotate(
        texto,
        xy=(0.98, 0.02), xycoords='axes fraction',
        ha='right', va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.6)
    )

    # Leyenda
    ax.legend(frameon=False, fontsize=9, loc='upper left')

    fig.tight_layout()
    plt.show()

    # --- Resultados en consola ---
    print("\n--- Recta de calibración ---")
    print(f"m = {m:.6f} ± {m_err:.6f}")
    print(f"b = {b:.6f} ± {b_err:.6f}")
    print(f"R² = {r2:.6f}")


    return m, b, m_err, b_err, r2

# ---------- Guardar modelo ---------- #
def save_model(path, cal_dict):
    with open(path, 'wb') as f:
        pickle.dump(cal_dict, f)
    print(f"\nModelo guardado en: {path}")

# ---------- MAIN ---------- #
def main():
    data_dir = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Lorentz\2025_03_18_radiocromic_ocean_espectrometro\suavizados"      # <-- adapta tu carpeta
    out_file = "calibration_model.pkl"

    cals, sums, sums_err = process_folder(data_dir, n_peaks=2)

    m, b, m_err, b_err, r2 = linear_regression(cals, sums)

    model = {
        'slope'      : m,
        'intercept'  : b,
        'slope_err'  : m_err,
        'intercept_err': b_err,
        'r2'         : r2
    }
    save_model(out_file, model)

    # -------- propagación ejemplo dosis ----------
    #   dosis = (S - b)/m
    #   σ_dosis^2 = (∂/∂m)^2 σ_m^2 + (∂/∂b)^2 σ_b^2 + (∂/∂S)^2 σ_S^2
    #   donde ∂d/∂m = -(S-b)/m² ; ∂d/∂b = -1/m ; ∂d/∂S = 1/m

    def dosis_y_error(S, S_err):
        d = (S - b)/m
        dd_dm   = -(S - b)/(m**2)
        dd_db   = -1/m
        dd_dS   =  1/m
        var = (dd_dm*m_err)**2 + (dd_db*b_err)**2 + (dd_dS*S_err)**2
        return d, np.sqrt(var)
    # Ejemplo con el primer punto
    d_demo, d_err_demo = dosis_y_error(sums[0], sums_err[0])
    print(f"\nEjemplo dosis primer espectro: {d_demo:.4f} ± {d_err_demo:.4f}")

if __name__ == "__main__":
    main()

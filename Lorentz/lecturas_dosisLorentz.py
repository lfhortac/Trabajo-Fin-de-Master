import os, glob, pickle, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────
#  Configuración global
# ──────────────────────────────────────────────────────────
x_min, x_max   = 650.0, 710.0   # rango útil
n_peaks        = 5              # nº de picos Lorentz a ajustar
eps_grad       = 1e-6           # paso para gradiente numérico (σS)
cal_model_file = "calibration_model.pkl"
out_csv        = "dose_results.csv"

# ──────────────────────────────────────────────────────────
#  Funciones de modelo
# ──────────────────────────────────────────────────────────
def lorentzian(x, amp, cen, wid):
    return (2*amp/np.pi) * (wid / (4*(x-cen)**2 + wid**2))

def multiple_lorentzians(x, *p):
    res = np.zeros_like(x)
    for i in range(len(p)//3):
        a, c, w = p[3*i:3*i+3]
        res += lorentzian(x, a, c, w)
    return res

# ──────────────────────────────────────────────────────────
#  Lectura de espectro
# ──────────────────────────────────────────────────────────
def read_spectrum(fp, skip=19):
    wl, od = [], []
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f.readlines()[skip:]:
            ln = ln.strip()
            if not ln or ln.startswith('#'):     # línea vacía o comentario
                continue
            parts = ln.replace(',', '.').split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if x_min <= x <= x_max:
                wl.append(x); od.append(y)
    return np.array(wl), np.array(od)

# ──────────────────────────────────────────────────────────
#  Ajuste de Lorentzianas y σS
# ──────────────────────────────────────────────────────────
def init_guess(wl, od, n):
    a0   = od.max()/n
    c0   = np.linspace(wl.min(), wl.max(), n+2)[1:-1]
    w0   = (wl.max()-wl.min())/(4*n)
    return np.ravel([[a0, c, w0] for c in c0])

def fit_lorentzians(wl, od, n):
    p0 = init_guess(wl, od, n)
    popt, pcov = curve_fit(multiple_lorentzians, wl, od, p0=p0, maxfev=500000)
    return popt, pcov

def lorentz_sum_and_error(wl, popt, pcov):
    """
    Devuelve S = Σ modelo y σS propagado (gradᵀ pcov grad).
    """
    S = multiple_lorentzians(wl, *popt).sum()

    # gradiente numérico de S respecto a cada parámetro
    grad = np.zeros_like(popt)
    for i in range(len(popt)):
        dp = np.zeros_like(popt); dp[i] = eps_grad
        S_plus  = multiple_lorentzians(wl, *(popt+dp)).sum()
        S_minus = multiple_lorentzians(wl, *(popt-dp)).sum()
        grad[i] = (S_plus - S_minus)/(2*eps_grad)

    var_S = grad @ pcov @ grad
    sigma_S = np.sqrt(var_S) if var_S > 0 else np.nan
    return S, sigma_S

# ──────────────────────────────────────────────────────────
#  Carga del modelo de calibración
# ──────────────────────────────────────────────────────────
def load_calibration_model(path=cal_model_file):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    print(f"Modelo → {path}")
    for k in ("slope", "slope_err", "intercept", "intercept_err", "r2"):
        print(f"  {k}: {d[k]}")
    return d

# ──────────────────────────────────────────────────────────
#  Cálculo de dosis + σdosis
# ──────────────────────────────────────────────────────────
def dose_and_error(S, sigma_S, model):
    m, b   = model['slope'], model['intercept']
    sm, sb = model['slope_err'], model['intercept_err']

    dose = (S - b)/m

    dd_dm = -(S - b)/m**2
    dd_db = -1/m
    dd_dS =  1/m

    sigma_dose = np.sqrt(
        (dd_dm*sm)**2 +
        (dd_db*sb)**2 +
        (dd_dS*sigma_S)**2
    )
    return dose, sigma_dose

# ──────────────────────────────────────────────────────────
#  Procesamiento de una carpeta
# ──────────────────────────────────────────────────────────
def process_folder(folder, model):
    txts = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not txts:
        print("⚠  No se encontraron .txt en", folder); return []

    results = []
    plt.figure(figsize=(12,8))

    for fp in txts:
        fn = os.path.basename(fp)
        wl, od = read_spectrum(fp)
        if wl.size == 0:
            continue

        popt, pcov = fit_lorentzians(wl, od, n_peaks)
        S, sigma_S = lorentz_sum_and_error(wl, popt, pcov)
        dose, sigma_dose = dose_and_error(S, sigma_S, model)

        results.append({
            "filename"       : fn,
            "S"              : S,
            "sigma_S"        : sigma_S,
            "dose"           : dose,
            "sigma_dose"     : sigma_dose
        })

        # trazado
        plt.plot(wl, od, 'o', ms=3, alpha=.6,
                 label=f"{fn}  →  dosis {dose:.3f}±{sigma_dose:.3f}")
        plt.plot(wl, multiple_lorentzians(wl, *popt), '-', alpha=.8)

    plt.xlabel("λ (nm)"); plt.ylabel("OD")
    plt.title("Espectros y ajustes")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=7)
    plt.tight_layout(); plt.show()
    return results

# ──────────────────────────────────────────────────────────
#  Guardar CSV
# ──────────────────────────────────────────────────────────
def save_results(res, csv_file=out_csv):
    if not res:
        print("⚠  Nada que guardar."); return
    pd.DataFrame(res).to_csv(csv_file, index=False)
    print("Resultados →", csv_file)

# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────
def main():
    data_dir = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Datos19mayo\RC6\OD_resultados"
    if not os.path.isdir(data_dir):
        print("⚠  Carpeta no encontrada:", data_dir); return

    model = load_calibration_model()
    res   = process_folder(data_dir, model)
    save_results(res)

    # resumen breve en consola
    for r in res:
        print(f"{r['filename']}:  {r['dose']:.3f} ± {r['sigma_dose']:.3f}")

if __name__ == "__main__":
    main()

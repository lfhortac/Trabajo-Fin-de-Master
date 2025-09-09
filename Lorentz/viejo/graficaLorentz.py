import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import joblib

# --- Parámetros globales ---
ROI_MIN = 650.0   # límite inferior de la región de interés (nm)
ROI_MAX = 710.0   # límite superior de la región de interés (nm)

def read_spectrophotometer_file(filepath):
    """Lee todo el espectro, filtrando líneas no numéricas."""
    wl, od = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()[351:]
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        try:
            w, d = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        wl.append(w); od.append(d)
    return np.array(wl), np.array(od)

def lorentz(x, I, xc, w):
    return (2 * I / np.pi) * (w / (4*(x-xc)**2 + w**2))

def multiple_lorentzians(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        y += lorentz(x, *params[i:i+3])
    return y

def fit_lorentzians(x, y, n_peaks=2):
    """Ajusta sólo al array x,y proporcionados (ya filtrados)."""
    # estimación inicial
    I0 = np.max(y) / n_peaks
    span = x.max() - x.min()
    guess = []
    for k in range(n_peaks):
        guess += [I0,
                  x.min() + (k+1)*(span/(n_peaks+1)),
                  span/(n_peaks*4)]
    try:
        popt, pcov = curve_fit(multiple_lorentzians, x, y, p0=guess, maxfev=50000)
        return popt, pcov
    except Exception as e:
        print(f"  Error en ajuste de curvas: {e}")
        return None, None

def calculate_dose(total_od, model):
    slope, intercept = model.coef_[0], model.intercept_
    dose = (total_od - intercept) / slope
    return dose, 0.0

def process_dose_files(data_directory, model, n_peaks=5):
    txts = glob.glob(os.path.join(data_directory, "*.txt"))
    if not txts:
        print("No hay archivos .txt.")
        return []

    plt.figure(figsize=(12, 8))
    results = []

    for fp in txts:
        fn = os.path.basename(fp)
        print(f"Procesando {fn}…")
        wl, od = read_spectrophotometer_file(fp)
        if wl.size == 0:
            print("  → espectro vacío.")
            continue

        # 1) Graficar TODO el espectro
        plt.plot(wl, od, 'k-', alpha=0.3, label=f"{fn} (espectro completo)")

        # 2) Enmascarar la región de interés
        mask_roi = (wl >= ROI_MIN) & (wl <= ROI_MAX)
        x_roi = wl[mask_roi]
        y_roi = od[mask_roi]
        if x_roi.size < 5:
            print("  → muy pocos puntos en Región de Interés.")
            continue

        # 3) Ajustar sólo en esa región
        params, _ = fit_lorentzians(x_roi, y_roi, n_peaks=n_peaks)
        if params is None:
            continue

        # 4) Reconstruir ajuste en la ROI
        fit_roi = multiple_lorentzians(x_roi, *params)
        # y cada pico
        components = [lorentz(x_roi, *params[i:i+3]) 
                      for i in range(0, len(params), 3)]
        total_sum = fit_roi.sum()

        # 5) Calcular dosis
        dose, std = calculate_dose(total_sum, model)
        results.append({'file': fn, 'dose': dose, 'std': std})

        # 6) Graficar ajuste (sólo en ROI) y componentes
        plt.plot(x_roi, fit_roi, 'r-', lw=2, label=f"{fn} ajuste ROI")
        for idx, comp in enumerate(components, start=1):
            plt.plot(x_roi, comp, '--', lw=1.5,
                     label=f"{fn} pico {idx}")

    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Densidad óptica (OD)")
    plt.title("Espectros completos y ajustes Lorentzianos en ROI")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return results

if __name__ == "__main__":
    modelo = joblib.load("calibration_model.pkl")
    carpeta = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Datos19mayo\solo1"
    process_dose_files(carpeta, modelo, n_peaks=5)

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def read_calibration_params(filepath):
    """
    Lee un txt con líneas de la forma
      a = <valor> ± <incertidumbre>
      b = ...
      c = ...
    y devuelve floats: (a, sigma_a, b, sigma_b, c, sigma_c).
    """
    params = {}
    pattern = re.compile(r'([abc])\s*=\s*([+-]?[0-9.]+(?:e[+-]?\d+)?)\s*±\s*([0-9.]+(?:e[+-]?\d+)?)')
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                key = m.group(1)
                val = float(m.group(2))
                err = float(m.group(3))
                params[key] = (val, err)
    a, sa = params['a']
    b, sb = params['b']
    c, sc = params['c']
    return a, sa, b, sb, c, sc

def read_spectrophotometer_file(filepath, header_lines=19):
    wavelengths, optical_densities = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        raw_lines = file.readlines()[header_lines:]
    for line in raw_lines:
        parts = line.strip().replace(',', '.').split()
        if len(parts) < 2:
            continue
        try:
            wl = float(parts[0])
            od = float(parts[1])
        except ValueError:
            continue
        wavelengths.append(wl)
        optical_densities.append(od)
    return np.array(wavelengths), np.array(optical_densities)

def find_counts_at_wavelength(wl, counts, target=663.0, tol=0.0005):
    """
    Devuelve (media, desviación típica) de counts en torno a target±tol.
    Si sólo hay un punto, σ=0 en lugar de nan.
    """
    mask = np.abs(wl - target) <= tol
    if not np.any(mask):
        return None, None
    vals = counts[mask]
    mean = np.mean(vals)
    if vals.size > 1:
        std = np.std(vals, ddof=1)
    else:
        std = 0.0
    return mean, std

def invert_exp_with_cov(y, popt, pcov, sigma_y):
    a, b, c = popt
    if y <= c:
        return None, None

    # x invertido
    x = np.log((y - c) / a) / b

    # derivadas parciales
    dx_da = -1.0 / (a * b)
    dx_db = -np.log((y - c) / a) / (b**2)
    dx_dc = -1.0 / (b * (y - c))

    # vector Jacobiano sólo para [a,b,c]
    J = np.array([dx_da, dx_db, dx_dc])

    # contribución de la incertidumbre en y (independiente)
    #var_y = (1.0 / (b * (y - c)) * sigma_y)**2
    # contribución de la incertidumbre en los parámetros
    # propagación completa para [a,b,c]
    var_params = J @ pcov @ J.T

    # var total
    sigma_x = np.sqrt(var_params)
    return x, sigma_x

def extract_dose_from_filename(filename):
    # Busca #dosis# o la primera cifra
    m = re.search(r'#(\d+\.?\d*)#', filename)
    if m:
        return float(m.group(1))
    m2 = re.search(r'(\d+\.?\d*)', filename)
    return float(m2.group(1)) if m2 else None



if __name__ == '__main__':
    # --- Ajusta esta ruta a tus datos ---
    data_dir = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\LongitudOnda\2025_03_18_radiocromic_ocean_espectrometro\suavizados'
    params_file = Path(__file__).resolve().parent / 'calibration_params.txt'

    # 1) Leer parámetros de calibración
    a, sa, b, sb, c, sc = read_calibration_params(params_file)
    pcov = np.array([
        [sa**2, 0, 0],
        [0, sb**2, 0],
        [0, 0, sc**2]
    ])
    print(f"Parámetros cargados:\n"
          f"  a = {a:.6e} ± {sa:.6e}\n"
          f"  b = {b:.6e} ± {sb:.6e}\n"
          f"  c = {c:.6e} ± {sc:.6e}\n")

    # 2) Procesar cada espectro y calcular dosis ± incertidumbre
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        fp = os.path.join(data_dir, fname)
        wl, od = read_spectrophotometer_file(fp)
        y_mean, y_std = find_counts_at_wavelength(wl, od)
        if y_mean is None:
            print(f"{fname}: no hay datos cerca de 663 nm → SKIP")
            continue

        # 3) Extraer dosis del nombre del archivo
        x_val, sigma_x_val = invert_exp_with_cov(y_mean, (a, b, c), pcov, y_std)
        print(f"Dosis = {x_val:.3f} ± {sigma_x_val:.3f}")
        if x_val is None:
            print(f"{fname}: OD@663={y_mean:.3f} ≤ c={c:.3f} → dosis indefinida")
        else:
            print(f"{fname}:\n"
                  f"  OD@663 = {y_mean:.3f} ± 0.5 \n"
                  f"  dosis  = {x_val:.3f} ± {sigma_x_val:.3f}\n")

    # (Opcional) gráfico de todos los espectros
    spectra = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        wl, od = read_spectrophotometer_file(os.path.join(data_dir, fname))
        if wl.size:
            spectra.append((fname, wl, od))
    if spectra:
        plt.figure(figsize=(9, 6), dpi=100)
        for fname, wl, od in spectra:
            plt.plot(wl, od, label=f"{extract_dose_from_filename(fname)} Gy", alpha=1.0)
        plt.axvline(663, linestyle='--', label='663 nm')
        plt.xlim(475, 800)
        plt.grid( which='both', linestyle='--', alpha=0.5)
        plt.xticks(np.arange(475, 801, 25))
        plt.yticks(np.arange(-0.2, 3.1, 0.2))
        plt.ylim(-0.2, 3.0)
        plt.gca().set_aspect('auto', adjustable='box')
        #plt.title('Espectros OD')
        plt.legend(fontsize=11, ncol=2)
    
        plt.tight_layout()
        plt.xlabel('Longitud de onda (nm)')
        plt.ylabel('Densidad óptica')
        plt.show()
    else:
        print("No se cargó ningún espectro válido.")


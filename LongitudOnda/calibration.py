import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_spectrophotometer_file(filepath, header_lines=19):
    wavelengths, optical_densities = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[header_lines:]
    for line in lines:
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

def extract_dose_from_filename(filename):
    # Busca #dosis# o la primera cifra
    m = re.search(r'#(\d+\.?\d*)#', filename)
    if m:
        return float(m.group(1))
    m2 = re.search(r'(\d+\.?\d*)', filename)
    return float(m2.group(1)) if m2 else None

def find_counts_at_wavelength(wavelengths, counts, target=663.0, tol=0.0005):
    mask = np.abs(wavelengths - target) <= tol
    return np.mean(counts[mask]) if np.any(mask) else None

def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

def process_calibration_data(data_dir, params_output):
    doses = []
    counts663 = []
    
    # 1) Recorre todos los archivos .txt
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.txt'):
            continue
        fullpath = os.path.join(data_dir, fname)
        
        # 2) Lee espectro y extrae OD a 663 nm
        wl, od = read_spectrophotometer_file(fullpath)
        c663 = find_counts_at_wavelength(wl, od)
        if c663 is None:
            print(f"  ⚠️ No encontré 663 nm en {fname}")
            continue
        
        # 3) Extrae dosis del nombre
        dose = extract_dose_from_filename(fname)
        if dose is None:
            print(f"  ⚠️ No pude extraer dosis de {fname}")
            continue
        
        doses.append(dose)
        counts663.append(c663)
    
    # Pasa a arrays y ordena por dosis
    doses = np.array(doses)
    counts663 = np.array(counts663)
    idx = np.argsort(doses)
    doses = doses[idx]
    counts663 = counts663[idx]
    
    # 4) Ajuste exponencial con covarianza
    # ------------------------------------------------------------------
    # Si no tienes errores en tus mediciones, define uno homogéneo:
    # Este es el valor que te funciona para obtener errores decentes
    sigma_od_medidos = np.full_like(counts663, 0.0001)
    print(sigma_od_medidos)
    # Buena semilla (seed) para el ajuste: [amplitud, pendiente inicial, offset]
    p0 = [counts663.max() - counts663.min(), 0.1, counts663.min()]

    try:
        popt, pcov = curve_fit(
            exponential_model,
            doses,
            counts663,
            sigma=sigma_od_medidos,
            p0=p0,
            maxfev=10000,
            absolute_sigma=True
        )
    except RuntimeError as e:
        print(f"  ❌ ERROR: El ajuste no pudo converger. Razón: {e}")
        return # Salir de la función si el ajuste falla

    # 5) Calcular parámetros, errores y métricas de calidad
    # ------------------------------------------------------------------
    a, b, c = popt
    # Los errores de los parámetros vienen de la matriz de covarianza
    sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(pcov))

    # Calcular R² (coeficiente de determinación)
    residuals = counts663 - exponential_model(doses, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((counts663 - np.mean(counts663))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Calcular el Error Estándar del Residuo (SER)
    n_puntos = len(doses)
    n_parametros = len(popt)
    dof = n_puntos - n_parametros # Grados de libertad
    ser = np.sqrt(ss_res / dof) if dof > 0 else 0

    # 6) Guarda los resultados en el archivo de texto
    # ------------------------------------------------------------------
    with open(params_output, 'w', encoding='utf-8') as f:
        f.write(f"a = {a:.6e} ± {sigma_a:.6e}\n")
        f.write(f"b = {b:.6e} ± {sigma_b:.6e}\n")
        f.write(f"c = {c:.6e} ± {sigma_c:.6e}\n")
        f.write(f"R2 = {r2:.8f}\n")
        f.write(f"SER = {ser:.6e}\n") # Error Estándar del Residuo
    print(f"\n✅ Parámetros guardados en '{params_output}'")

    # 7) Gráfica final (sin cambios en esta parte)
    # ------------------------------------------------------------------
        # 7) Gráfica final con estilo científico y barras de error
    # ------------------------------------------------------------------
        # 7) Gráfica final con estilo científico y barras de error en los datos
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "text.usetex": True,       # usa LaTeX para texto
        "font.family": "serif",    # tipografía científica
        "axes.spines.top": False,  # oculta marco superior
        "axes.spines.right": False # oculta marco derecho
    })

    fig, ax = plt.subplots(figsize=(6.6, 4.4))  # proporción 3:2

    # Datos experimentales con barras de error
    ax.errorbar(
        doses, counts663, yerr=sigma_od_medidos,
        fmt='o', mfc='white', mec='red', color='red',
        ecolor='black', elinewidth=1, capsize=3, markersize=6,
        label=r'Datos medidos'
    )

    # Curva de ajuste (sin barras adicionales)
    x_fit = np.linspace(min(doses), max(doses), 300)
    y_fit = exponential_model(x_fit, *popt)
    ax.plot(
        x_fit, y_fit, '-', lw=1.5, color='black',
        label=fr'Ajuste exponencial ($R^2={r2:.4f}$)'
    )

    # Etiquetas y estética
    ax.set_xlabel(r'Dosis (Gy)')
    ax.set_ylabel(r'OD a 663\,nm')
    ax.set_xlim(0, max(doses) * 1.05)
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.5)

    ax.legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    data_dir = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\LongitudOnda\2025_03_18_radiocromic_ocean_espectrometro\suavizados'
    # Carpeta del script
    script_dir = Path(__file__).resolve().parent
    params_file = script_dir / 'calibration_params.txt'
    
    try:
        process_calibration_data(data_dir, params_file)
    except Exception as e:
        print("ERROR EN CALIBRACIÓN:", e)

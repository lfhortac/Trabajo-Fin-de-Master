import os, re
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def extract_dose(filename):
    """
    Devuelve la primera dosis numérica que encuentre en `filename`.
    - Si está entre #...#, la usa.
    - Si no, usa el primer número que aparezca.
    - Si no hay ningún número, devuelve None.
    """
    m = re.search(r'#\s*([\d.]+)\s*#', filename)   # #15#  ó  #2.5#
    if m:
        return float(m.group(1))
         # primer número libre
       # buscamos al menos un dígito, opcional parte decimal:
    m2 = re.search(r'(\d+(?:\.\d+)?)', filename) 
    return float(m2.group(1)) if m2 else None



def read_OD663_from_folder(folder, header=19, target=663, tol=0.5):
    doses, ods = [], []
    for fn in os.listdir(folder):
        if not fn.lower().endswith('.txt'):
            continue

        dose = extract_dose(fn)
        if dose is None:
            print(f"⚠️  No se pudo extraer la dosis de '{fn}', se omite.")
            continue

        # lee columnas (ajusta si tu formato es distinto)
        wl, od = np.loadtxt(Path(folder)/fn,
                            skiprows=header, usecols=[0,1], unpack=True,
                            delimiter=None, dtype=float)

        mask = np.abs(wl - target) <= tol
        if not mask.any():
            print(f"⚠️  '{fn}' no tiene puntos en {target} ± {tol} nm, se omite.")
            continue

        doses.append(dose)
        ods.append(np.mean(od[mask]))

    return np.array(doses), np.array(ods)

def linear(x, m, n):
    return m*x + n

if __name__ == "__main__":
    data_dir  = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\LongitudOnda\2025_03_18_radiocromic_ocean_espectrometro\suavizados"               # <-- pon tu carpeta
    script_dir = Path(__file__).resolve().parent
    out_params = script_dir / "linear_params.txt"
    out_cov    = script_dir / "linear_cov.txt"

    x, y = read_OD663_from_folder(data_dir)
    popt, pcov  = curve_fit(linear, x, y)        # m, n  (+ covarianza)
    m, n = popt
    r2 = 1 - np.sum((y - linear(x, *popt))**2) / np.sum((y - np.mean(y))**2)
    np.savetxt(out_params, popt, header="m n")
    np.savetxt(out_cov, pcov,   header="2x2 cov matrix")
    # Gráfico: datos + recta con incertidumbres
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, label='Datos medidos')
    # recta de mejor ajuste
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = m * x_fit + n
    dm, dn = np.sqrt(np.diag(pcov))
    plt.plot(x_fit, y_fit, '-', label=f'Ajuste: y={m:.3f}±{dm:.3f}·x + {n:.3f}±{dn:.3f}')
    plt.xlabel('Dosis')
    plt.ylabel('OD a 663 nm')
    plt.title(f'Curva de calibración (R² = {r2:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfico de espectros con línea vertical
    plt.figure(figsize=(6, 4))
    for dose, (wl, od) in sorted(zip(x, zip(*[np.loadtxt(Path(data_dir)/fn, skiprows=19, usecols=[0,1], unpack=True, delimiter=None, dtype=float) for fn in os.listdir(data_dir) if fn.lower().endswith('.txt')]))):
        if not np.any(np.abs(wl - 663) <= 0.5):
            print(f"⚠️  No se encontraron puntos en 663 nm para la dosis {dose}, se omite.")
            continue
        plt.plot(wl, od, label=f'{dose}')
    plt.axvline(663, linestyle='--', linewidth=1.2, color='red', label='λ = 663 nm')
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Absorbancia')
    plt.title('Espectros a distintas dosis')
    plt.legend(title='Dosis')
    plt.tight_layout()
    plt.show()

    print(f"m = {m:.6e}, n = {n:.6e}")
    print(f"cov =\n{pcov}")

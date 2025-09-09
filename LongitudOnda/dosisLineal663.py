import os, re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def read_OD663(path, header=19, target=663, tol=0.5):
    wl, od = np.loadtxt(path, skiprows=header, usecols=[0,1],
                        delimiter=None, unpack=True, dtype=float)
    mask = np.abs(wl-target) <= tol
    if mask.any():
        vals = od[mask]
        return np.mean(vals), np.std(vals, ddof=1) if vals.size > 1 else 0.0
    return None, None

def dose_and_uncert(y, sy, m, n, cov):
    dm, dn = np.sqrt(np.diag(cov))
    cov_mn = cov[0,1]

    x = (y - n) / m

    dx_dm = -(y - n) / m**2
    dx_dn = -1 / m
    dx_dy =  1 / m             # opcional: pon sy=0 si no quieres incluir OD

    var = (dx_dm**2)*dm**2 + (dx_dn**2)*dn**2 + 2*dx_dm*dx_dn*cov_mn
    var += (dx_dy**2)*sy**2
    return x, np.sqrt(var)

if __name__ == "__main__":
    spectra_dir = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Datos19mayo\RC6\OD_resultados"            # <-- carpeta con espectros
    script_dir  = Path(__file__).resolve().parent
    m, n = np.loadtxt(script_dir/"linear_params.txt")
    cov   = np.loadtxt(script_dir/"linear_cov.txt")

    for fn in sorted(os.listdir(spectra_dir)):
        if not fn.lower().endswith('.txt'): 
            continue
        y, sy = read_OD663(Path(spectra_dir)/fn)
        if y is None:
            print(f"{fn}: sin 663 nm")
            continue
        dose, sdose = dose_and_uncert(y, sy, m, n, cov)
        print(f"{fn:35s} OD@663={y:.3f}±{sy:.3f}  Dosis={dose:.3f}±{sdose:.3f}")

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from natsort import natsorted

# === CONFIGURACIÓN ===
#carpeta_muestras = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_espectrometro_FC_Ciencias\OD_resultados'  # CAMBIAR a tu ruta
#carpeta_muestras = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\RC_nn\OD_resultados'  # CAMBIAR a tu ruta
#carpeta_muestras=r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_radiocromic_ocean_espectrometro\suavizados'
carpeta_muestras = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\RC_nn\OD_resultados\suavizados'  # CAMBIAR a tu ruta
archivo_calibracion = "resultados/curva_calibracion_total.csv"
rango_min, rango_max =  605, 665 
max_picos = 10

# === FUNCIONES ===
def lorentziana(x, I, xc, w):
    return (2 * I / np.pi) * (w / (4 * (x - xc)**2 + w**2))

def suma_lorentzianas(x, *params):
    y = np.zeros_like(x)
    for i in range(len(params) // 3):
        I, xc, w = params[3*i:3*i+3]
        y += lorentziana(x, I, xc, w)
    return y

def ajustar_espectro(x, y):
    mask = (x >= rango_min) & (x <= rango_max)
    x = x[mask]
    y = y[mask]

    idx_picos, _ = find_peaks(y, prominence=0.001, distance=10)
    idx_picos = idx_picos[:max_picos]
    if len(idx_picos) == 0:
        return None, None, None

    p0 = []
    for idx in idx_picos:
        I0 = y[idx] * np.pi
        xc0 = x[idx]
        w0 = 10.0
        p0.extend([I0, xc0, w0])

    try:
        popt, _ = curve_fit(suma_lorentzianas, x, y, p0=p0, maxfev=50000)
        return x, y, popt
    except:
        return None, None, None

# === CARGAR CURVA DE CALIBRACIÓN ===
df_cal = pd.read_csv(archivo_calibracion)
X_cal = df_cal["Área total (∑Ii)"].values.reshape(-1, 1)
y_cal = df_cal["Dosis [Gy]"].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_cal)
model = LinearRegression()
model.fit(X_poly, y_cal)

# === PROCESAR MÚLTIPLES MUESTRAS ===
archivos = natsorted(glob.glob(os.path.join(carpeta_muestras, "*.txt")))
os.makedirs("resultados/lote", exist_ok=True)

resultados = []

print(f"\n🔍 Procesando {len(archivos)} archivos...")

for archivo in archivos:
    nombre = os.path.basename(archivo)
    try:
        df = pd.read_csv(archivo, sep=r"\s+|,", engine="python", skiprows=1, names=["Wavelength", "OD"])
        x = df["Wavelength"].values
        y = df["OD"].values

        x_fit, y_fit, popt = ajustar_espectro(x, y)
        if popt is None:
            print(f"❌ {nombre}: error en ajuste")
            resultados.append([nombre, np.nan, np.nan])
            continue

        area = sum([popt[3*i] for i in range(len(popt)//3)])
        dosis = model.predict(poly.transform([[area]]))[0]

        # Graficar
        plt.figure(figsize=(6, 4))
        plt.plot(x_fit, y_fit, label="OD", color="black")
        plt.plot(x_fit, suma_lorentzianas(x_fit, *popt), '--', label="Ajuste", color="red")
        plt.title(f"{nombre}\nDosis estimada: {dosis:.2f} Gy")
        plt.xlabel("Longitud de onda (nm)")
        plt.ylabel("OD")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"resultados/lote/{nombre.replace('.txt', '_ajuste.png')}")
        plt.close()

        print(f"✅ {nombre}: dosis = {dosis:.2f} Gy")
        resultados.append([nombre, area, dosis])

    except Exception as e:
        print(f"⚠️ {nombre}: error inesperado → {e}")
        resultados.append([nombre, np.nan, np.nan])

# === GUARDAR RESULTADOS ===
df_out = pd.DataFrame(resultados, columns=["Archivo", "Área total (∑Ii)", "Dosis estimada (Gy)"])
df_out.to_csv("resultados/lote/dosis_estimadas_lote.csv", index=False)

print("\n✅ Dosis estimadas guardadas en: resultados/lote/dosis_estimadas_lote.csv")

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from natsort import natsorted
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# === CONFIGURACIÓN ===
#para datos de ocean, el rango de longitud de onda es 610-670 nm, sin 0.7 incluido
x_min, x_max = 605, 665  # Rango espectral útil para el ajuste
carpeta=r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_radiocromic_ocean_espectrometro\suavizados'
dosis = [0.1, 0.3, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20]  # Orden fijo

#para datos de FC, el rango de longitud de onda es 600-650 nm
#x_min, x_max = 600, 650  # Rango espectral útil para el ajuste
#carpeta = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_espectrometro_FC_Ciencias\OD_resultados' # Cambia esta ruta a la carpeta donde están los archivos .txt
#dosis = [0.1, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # Orden fijo



max_picos = 10  # Cuántos picos como máximo detectar por espectro
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
    # Filtrar rango espectral útil
    mask = (x >= x_min) & (x <= x_max)
    x = x[mask]
    y = y[mask]

    # Detección de picos
    idx_picos, _ = find_peaks(y, prominence=0.001, distance=10)
    idx_picos = idx_picos[:max_picos]

    if len(idx_picos) == 0:
        return None, None

    # Estimaciones iniciales
    p0 = []
    for idx in idx_picos:
        I0 = y[idx] * np.pi
        xc0 = x[idx]
        w0 = 10.0
        p0.extend([I0, xc0, w0])

    try:
        popt, _ = curve_fit(suma_lorentzianas, x, y, p0=p0, maxfev=50000)
        return x, popt
    except Exception as e:
        print(f"⚠️ Error en ajuste: {e}")
        return None, None

# === INICIO DEL PROCESAMIENTO ===
archivos = natsorted(glob.glob(os.path.join(carpeta, "*.txt")))
os.makedirs("resultados", exist_ok=True)

areas_totales = []

print("\n🚀 Procesando espectros...")
for i, archivo in enumerate(archivos):
    print(f"{i+1:02d}/{len(archivos)}: {os.path.basename(archivo)}")
    df = pd.read_csv(archivo, sep=r"\s+|,", engine="python", skiprows=1, names=["Wavelength", "OD"])
    x = df["Wavelength"].values
    y = df["OD"].values

    x_fit, popt = ajustar_espectro(x, y)

    if popt is None:
        areas_totales.append(np.nan)
        continue

    # Calcular área total
    area_total = sum([popt[3*i] for i in range(len(popt)//3)])
    areas_totales.append(area_total)

    # Graficar ajuste
    plt.figure(figsize=(6, 4))
    plt.plot(x_fit, y[(x >= x_min) & (x <= x_max)], label="OD", color='black')
    plt.plot(x_fit, suma_lorentzianas(x_fit, *popt), '--', label="Ajuste Lorentziano", color='red')
    plt.title(f"Ajuste: {os.path.basename(archivo)}")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("OD")
    plt.legend()
    plt.tight_layout()
    nombre_base = os.path.splitext(os.path.basename(archivo))[0]
    plt.savefig(f"resultados/{nombre_base}_ajuste.png")
    plt.close()

# === CURVA DE CALIBRACIÓN ===
print("\n📈 Generando curva de calibración...")

if len(areas_totales) != len(dosis):
    print("❌ Número de espectros y dosis no coincide.")
else:
    X = np.array(dosis).reshape(-1, 1)
    y = np.array(areas_totales)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    # Guardar resultados
    df_out = pd.DataFrame({
        "Dosis [Gy]": dosis,
        "Área total (∑Ii)": y,
        "Modelo": y_pred
    })
    df_out.to_csv("resultados/curva_calibracion_total.csv", index=False)

    # Gráfico
    plt.figure(figsize=(6, 4))
    plt.scatter(dosis, y, label="Datos", color='blue')
    plt.plot(dosis, y_pred, '--', label="Ajuste cuadrático", color='red')
    plt.xlabel("Dosis [Gy]")
    plt.ylabel("Área total (∑Ii)")
    plt.title("Curva de calibración")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultados/curva_calibracion_total.png")
    plt.show()
    plt.close()
    

    print("✅ Curva de calibración generada.")

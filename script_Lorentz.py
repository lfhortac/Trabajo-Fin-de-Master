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

# ---------- Funciones Lorentzianas ----------

def lorentzian(x, I, xc, w):
    return (2 * I / np.pi) * (w / (4 * (x - xc)**2 + w**2))

def multi_lorentzian(x, *params):
    n = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n):
        I, xc, w = params[3*i], params[3*i+1], params[3*i+2]
        y += lorentzian(x, I, xc, w)
    return y

# ---------- Ajuste múltiple ----------

def ajustar_lorentziano(x, y, max_picos=6):
    mask = (x >= 450) & (x <= 750)
    x = x[mask]
    y = y[mask]

    idx_picos, _ = find_peaks(y, prominence=0.001, distance=10)
    idx_picos = idx_picos[:max_picos]

    if len(idx_picos) == 0:
        return None

    p0 = []
    for idx in idx_picos:
        I0 = y[idx] * np.pi
        xc0 = x[idx]
        w0 = 5.0
        p0.extend([I0, xc0, w0])

    try:
        popt, _ = curve_fit(multi_lorentzian, x, y, p0=p0, maxfev=50000)
        return popt
    except RuntimeError as e:
        print(f"⚠️ No convergió el ajuste múltiple: {e}")
        return None

# ---------- Ajuste simple del pico principal ----------

def ajustar_principal(x, y):
    mask = (x >= 600) & (x <= 700)
    x = x[mask]
    y = y[mask]

    idx_max = np.argmax(y)
    I0 = y[idx_max] * np.pi
    xc0 = x[idx_max]
    w0 = 10.0
    p0 = [I0, xc0, w0]

    try:
        popt, _ = curve_fit(lorentzian, x, y, p0=p0, maxfev=50000)
        print(f"✅ Ajuste pico principal: I={popt[0]:.4f}, xc={popt[1]:.2f}, w={popt[2]:.2f}")
        return popt
    except Exception as e:
        print(f"❌ Error en ajuste del pico principal: {e}")
        return None

# ---------- Lectura de archivos ----------
#carpeta=r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_radiocromic_ocean_espectrometro'
carpeta = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_espectrometro_FC_Ciencias\OD_resultados'
datos = glob.glob(os.path.join(carpeta, "*.txt"))
archivos_ordenados = natsorted(datos)
os.makedirs("resultados", exist_ok=True)
# Dosis conocidas en orden (corresponden al orden de los archivos)
dosis = [0.1, 0.3, 0.5,0.7, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

for archivo in archivos_ordenados:
    print(f"\n📂 Procesando: {os.path.basename(archivo)}")
    df = pd.read_csv(archivo, sep=r"\s+|,", engine="python", skiprows=1, names=["Wavelength", "OD"])
    x = df["Wavelength"].values
    y = df["OD"].values
    mask = (x >= 600) & (x <= 700)
    x = x[mask]
    y = y[mask]
    parametros = ajustar_lorentziano(x, y)

    # Si falla el múltiple, intentar con uno solo
    if parametros is None:
        print("🔁 Probando con ajuste solo del pico principal...")
        parametros = ajustar_principal(x, y)
        if parametros is None:
            print("❌ No se pudo ajustar este archivo.")
            continue
        else:
            resultados = [["Pico 1", parametros[0], parametros[1], parametros[2]]]
            y_fit = lorentzian(x, *parametros)
    else:
        resultados = []
        n_picos = len(parametros) // 3
        for i in range(n_picos):
            I = parametros[3*i]
            xc = parametros[3*i+1]
            w = parametros[3*i+2]
            print(f"🔹 Pico {i+1}: I={I:.5f}, xc={xc:.2f}, w={w:.2f}")
            resultados.append([f"Pico {i+1}", I, xc, w])
        y_fit = multi_lorentzian(x, *parametros)

    # Guardar resultados
    nombre_base = os.path.splitext(os.path.basename(archivo))[0]
    df_out = pd.DataFrame(resultados, columns=["Pico", "Intensidad", "Centro", "Ancho"])
    df_out.to_csv(f"resultados/{nombre_base}_ajuste.csv", index=False)

    # Graficar ajuste
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="OD original")
    plt.plot(x, y_fit, '--', label="Ajuste Lorentziano")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("OD")
    plt.title(f"Ajuste - {nombre_base}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/{nombre_base}_ajuste.png")
    plt.close()

# ---------- Construcción de la curva de calibración ----------
print("\n📈 Construyendo curva de calibración desde orden fijo...")

# Buscar los archivos CSV ya ajustados
csvs = sorted(glob.glob("resultados/*_ajuste.csv"))

if len(csvs) != len(dosis):
    print(f"❌ El número de archivos ({len(csvs)}) no coincide con el número de dosis ({len(dosis)})")
else:
    areas = []
    for archivo in csvs:
        df = pd.read_csv(archivo)
        if df.empty:
            print(f"⚠️ Archivo vacío: {archivo}")
            areas.append(np.nan)
        else:
            areas.append(df["Intensidad"].iloc[0])

    # Ajuste polinomial de grado 2
    X = np.array(dosis).reshape(-1, 1)
    y = np.array(areas)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    # Mostrar fórmula
    coef = model.coef_
    intercept = model.intercept_
    print(f"✅ Ajuste cuadrático: y = {intercept:.4f} + {coef[1]:.4f}·x + {coef[2]:.4f}·x²")

    # Guardar curva
    df_cal = pd.DataFrame({
        "Dosis [Gy]": dosis,
        "Área pico principal": y,
        "Ajuste (modelo)": y_pred
    })
    df_cal.to_csv("resultados/curva_calibracion.csv", index=False)

    print("\n👉 Diagnóstico de áreas:")
    for d, a in zip(dosis, areas):
        print(f"Dosis {d} Gy → Área: {a}")
    # Graficar
    plt.figure(figsize=(6, 4))
    plt.scatter(dosis, y, label="Datos", color='blue')
    plt.plot(dosis, y_pred, color='red', linestyle='--', label="Ajuste cuadrático")
    plt.xlabel("Dosis [Gy]")
    plt.ylabel("Área del pico principal")
    plt.title("Curva de calibración")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultados/curva_calibracion.png")
    plt.close()    


import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from natsort import natsorted
import matplotlib.pyplot as plt

# --- FUNCIÓN GAUSSIANA ---
def gauss(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

# --- CALIBRACIÓN (de experimentos anteriores) ---
# Estos valores deben venir del ajuste a tus curvas calibradas (ordenados correctamente)
dosis_calibracion = [0.1, 0.3, 0.5 , 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
parametros_a_calibracion = [
    0.05541109253211352,
    0.15714804930337153,
    0.25493256168760386,
    0.469067059466968,
    0.4344744251983481,
    0.7419642751649856,
    1.1392623041128715,
    1.452718982588203,
    1.7169318005623997,
    1.8256231388085362,
    2.028483090483305,
    2.080014786110838,
    2.0709421612900445,
    2.123916624839456
]  # reemplazá con tus valores reales

# --- Crear interpolador inverso (a → dosis) ---
interp_a_to_dosis = interp1d(parametros_a_calibracion, dosis_calibracion, kind='cubic', fill_value='extrapolate')

# --- RUTA DE LOS NUEVOS ESPECTROS ---
od_folder = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\RC_nn\OD_resultados'
archivos = natsorted([f for f in os.listdir(od_folder) if f.endswith('.txt')])

resultados = []

for archivo in archivos:
    ruta_archivo = os.path.join(od_folder, archivo)
    try:
        df = pd.read_csv(ruta_archivo, sep=r'\s+', engine='python', header=None, skiprows=1, comment='#')
        if df.shape[1] < 2:
            print(f"⚠️ {archivo} no tiene suficientes columnas.")
            continue

        x = df[0].astype(float).values
        y = df[1].astype(float).values

        # Filtrar rango 610–680 nm
        mask = (x >= 610) & (x <= 680)
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 3:
            print(f"⚠️ Muy pocos puntos para ajustar en {archivo}.")
            continue

        # Ajustar gaussiana
        a_ini = max(y_fit)
        b_ini = x_fit[np.argmax(y_fit)]
        c_ini = 10
        popt, _ = curve_fit(gauss, x_fit, y_fit, p0=[a_ini, b_ini, c_ini])
        a, b, c = popt

        # Estimar dosis por interpolación inversa
        dosis_estimada = float(interp_a_to_dosis(a))

        resultados.append({
            'Archivo': archivo,
            'a (altura)': a,
            'b (nm)': b,
            'c': c,
            'Dosis estimada (Gy)': dosis_estimada
        })

        print(f"✅ {archivo}: a = {a:.3f} → Dosis ≈ {dosis_estimada:.2f} Gy")

    except Exception as e:
        print(f"❌ Error con {archivo}: {e}")

# --- GUARDAR RESULTADOS ---
df_resultados = pd.DataFrame(resultados)
csv_out = os.path.join(od_folder, 'Dosis_estimadas_interp.csv')
df_resultados.to_csv(csv_out, index=False)
print(f"\n📁 Resultados guardados en: {csv_out}")

# --- GRAFICAR CURVA DE CALIBRACIÓN USADA ---
plt.figure(figsize=(8, 5))
a_fit = np.linspace(min(parametros_a_calibracion), max(parametros_a_calibracion), 200)
d_fit = interp_a_to_dosis(a_fit)
plt.plot(parametros_a_calibracion, dosis_calibracion, 'o', label='Datos calibración')
plt.plot(a_fit, d_fit, '-', label='Interpolación inversa (a → dosis)')
plt.xlabel('Altura del Pico Gaussiano (a)')
plt.ylabel('Dosis (Gy)')
plt.title('Interpolación inversa de calibración')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

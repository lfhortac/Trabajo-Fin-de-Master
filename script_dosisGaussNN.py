import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from natsort import natsorted
import matplotlib.pyplot as plt

# --- Definir función gaussiana ---
def gauss(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

# --- Función para estimar dosis a partir de a ---
def estimar_dosis(a):
    return 10**((a - 0.08) / 0.71) - 0.83

# --- Ruta a los OD generados ---
od_folder = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_radiocromic_ocean_espectrometro"

# --- Preparar resultados ---
resultados = []

# --- Leer archivos ordenados ---
archivos = natsorted([f for f in os.listdir(od_folder) if f.endswith('.txt')])

for archivo in archivos:
    ruta_archivo = os.path.join(od_folder, archivo)
    try:
        df = pd.read_csv(
            ruta_archivo,
            sep=r'\s+',
            engine='python',
            header=None,
            skiprows=1,
            comment='#'
        )
    except Exception as e:
        print(f"❌ Error al leer {archivo}: {e}")
        continue

    if df.shape[1] < 2:
        print(f"⚠️ Archivo {archivo} no tiene suficientes columnas.")
        continue

    try:
        longitud_onda = df[0].astype(float).values
        od = df[1].astype(float).values
    except ValueError as e:
        print(f"⚠️ Error convirtiendo a float en {archivo}: {e}")
        continue

    # Filtrar para ajuste en rango 610–680 nm
    mascara = (longitud_onda >= 610) & (longitud_onda <= 650)
    x_rango = longitud_onda[mascara]
    y_rango = od[mascara]

    if len(x_rango) < 3:
        print(f"⚠️ Muy pocos puntos en el rango para {archivo}")
        continue

    # Ajuste gaussiano
    try:
        a_ini = max(y_rango)
        b_ini = x_rango[np.argmax(y_rango)]
        c_ini = 10
        popt, _ = curve_fit(gauss, x_rango, y_rango, p0=[a_ini, b_ini, c_ini])
        a, b, c = popt
    except Exception as e:
        print(f"⚠️ Error ajustando gaussiana en {archivo}: {e}")
        continue

    # Calcular dosis estimada desde 'a'
    dosis_estimada = estimar_dosis(a)

    # Guardar resultados
    resultados.append({
        'Archivo': archivo,
        'a (altura pico)': a,
        'b (posición pico)': b,
        'c (ancho)': c,
        'Dosis estimada (Gy)': dosis_estimada
    })

# --- Guardar resultados ---
df_resultados = pd.DataFrame(resultados)
output_csv = os.path.join(od_folder, 'Dosis_estimadas_por_gauss.csv')
df_resultados.to_csv(output_csv, index=False)

print(f"\n✅ Ajustes completados. Resultados guardados en:\n{output_csv}")


# --- Graficar resultados ---


# --- Preparar datos para graficar ---
a_vals = df_resultados['a (altura pico)'].values
dosis_vals = df_resultados['Dosis estimada (Gy)'].values

# --- Graficar ---
plt.figure(figsize=(8, 5))
plt.plot(a_vals, dosis_vals, 'o-', color='purple', label='Estimación de dosis')
plt.xlabel('Altura del Pico Gaussiano (a)')
plt.ylabel('Dosis estimada (Gy)')
plt.title('Dosis estimada vs. Altura del Pico')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
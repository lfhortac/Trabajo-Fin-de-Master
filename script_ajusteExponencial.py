import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def leer_archivos_txt(carpeta):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.txt')]
    datos = {}
    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        try:
            data = np.loadtxt(ruta, skiprows=17)
            datos[archivo] = data
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
    return datos

def exponencial_decreciente(x, a, b, c):
    return a * np.exp(-b * x) + c

def ajustar_exponencial(data, x_min=640, x_max=680):
    x = data[:, 0]
    y = data[:, 1]
    mask = (x >= x_min) & (x <= x_max)
    x_rango = x[mask]
    y_rango = y[mask]

    if len(x_rango) < 5:
        print("⚠️ Muy pocos puntos en el rango 640–680")
        return None

    x_offset = x_rango.min()
    x_norm = x_rango - x_offset

    a0 = y_rango.max() - y_rango.min()
    b0 = 0.05
    c0 = y_rango.min()

    try:
        popt, _ = curve_fit(
            exponencial_decreciente,
            x_norm,
            y_rango,
            p0=[a0, b0, c0],
            maxfev=10000
        )
        return popt, x_offset
    except RuntimeError as e:
        print("❌ No se pudo ajustar:", e)
        return None

def graficar_datos_y_ajustes(datos, x_min, x_max):
    parametros_a = []
    nombres_ordenados = sorted(datos.keys())

    plt.figure(figsize=(10, 6))

    for archivo in nombres_ordenados:
        data = datos[archivo]
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y, label=archivo)

        result = ajustar_exponencial(data, x_min, x_max)
        if result is not None:
            (a, b, c), x_offset = result
            print(f"→ Archivo: {archivo}, ajuste: y = {a:.4f} * exp(-{b:.4f} * (x - {x_offset:.1f})) + {c:.4f}")
            parametros_a.append((archivo, a))
        else:
            parametros_a.append((archivo, None))

    plt.xlabel("X")
    plt.ylabel("Y (escala log)")
    plt.yscale('log')
    plt.legend()
    plt.title("Espectros y Ajustes Exponenciales")
    plt.grid()
    plt.show()

    return parametros_a

# ========== INICIO DEL PROCESO ==========

carpeta = "2025_03_18_radiocromic_ocean_espectrometro"
datos = leer_archivos_txt(carpeta)
x_min = 640
x_max = 680

# Ajustar y recolectar valores 'a'
parametros_a = graficar_datos_y_ajustes(datos, x_min, x_max)

# Valores experimentales conocidos (14 valores para 14 archivos)
valores_x = [0.1, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# Asegurarse de que los valores_x y los parámetros a tengan la misma longitud

# Filtrar para quedarnos solo con los 'a' válidos
# Separar 'a' y 'b' de los resultados
solo_a = []
solo_b = []

for archivo, a in parametros_a:
    if a is not None:
        solo_a.append(a)
    else:
        solo_a.append(np.nan)

# Necesitamos volver a extraer 'b' desde los archivos
solo_b = []
nombres_ordenados = sorted(datos.keys())
for archivo in nombres_ordenados:
    data = datos[archivo]
    result = ajustar_exponencial(data, x_min, x_max)
    if result is not None:
        (_, b, _), _ = result
        solo_b.append(b)
    else:
        solo_b.append(np.nan)

# Graficar con dos ejes
fig, ax1 = plt.subplots(figsize=(8, 5))

# Eje Y para 'a'
color_a = 'tab:blue'
ax1.set_xlabel("Dosis (Gy)")
ax1.set_ylabel("Parámetro a", color=color_a)
ax1.plot(valores_x, solo_a, 'o-', color=color_a, label="Parámetro a")
ax1.tick_params(axis='y', labelcolor=color_a)

# Eje Y para 'b'
ax2 = ax1.twinx()
color_b = 'tab:red'
ax2.set_ylabel("Parámetro b", color=color_b)
ax2.plot(valores_x, solo_b, 's--', color=color_b, label="Parámetro b")
ax2.tick_params(axis='y', labelcolor=color_b)

# Título y leyendas
plt.title("Parámetros a y b [a * np.exp(-b * x) + c] vs Dosis (Gy)")
ax1.legend(loc='upper left')
fig.tight_layout()
plt.grid(True)
plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import savgol_filter

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

def ajustar_exponencial(data, x_min, x_max):
    x = data[:, 0]
    y = data[:, 1]
    mask = (x >= x_min) & (x <= x_max)
    x_rango = x[mask]
    y_rango = y[mask]

    # Aplicar suavizado a Y
    if len(y_rango) >= 7:
        y_rango = savgol_filter(y_rango, window_length=5, polyorder=2)
    if len(x_rango) < 5:
        print(" Muy pocos puntos en el rango 640–680")
        return None

    x_offset = x_rango.min()
    x_norm = x_rango - x_offset

    a0 = y_rango.max() - y_rango.min()
    b0 = 0.05
    c0 = y_rango.min()

    try:
        popt, pcov = curve_fit(
            exponencial_decreciente,
            x_norm,
            y_rango,
            p0=[a0, b0, c0],
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))  # errores estándar
        return (popt, perr, x_offset)
    except RuntimeError as e:
        print(" No se pudo ajustar:", e)
        return None

def graficar_datos_y_ajustes(datos, valores_x, x_min, x_max):
    parametros_a = []
    nombres_ordenados = sorted(datos.keys())

    plt.figure(figsize=(10, 6))

    for i, archivo in enumerate(nombres_ordenados):
        data = datos[archivo]
        dosis = valores_x[i]

        # Usar rango especial para dosis bajas
        if dosis < 1.0:
                x_min_local, x_max_local = 650, x_max
        else:
                x_min_local, x_max_local = x_min, x_max

        result = ajustar_exponencial(data, x_min_local, x_max_local)
        if result is None:
            continue

        (popt, perr, x_offset) = result
        a, b, c = popt
        a_err, b_err, c_err = perr
        print(f"→ Archivo: {archivo}")
        print(f"   a = {a:.4f} ± {a_err:.4f}")
        print(f"   b = {b:.4f} ± {b_err:.4f}")
        print(f"   c = {c:.4f} ± {c_err:.4f}")
        parametros_a.append((archivo, a, a_err, b, b_err))

        # Graficar curva original
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y, label=f"Datos {archivo}")

        # Graficar curva ajustada
        x_fit = np.linspace(x_min_local, x_max_local, 200)
        y_fit = exponencial_decreciente(x_fit - x_offset, a, b, c)
        plt.plot(x_fit, y_fit, '--', label=f"Ajuste {archivo}")




        


    plt.xlabel("X")
    plt.ylabel("Y (escala log)")
    #plt.yscale('log')
    #plt.legend()
    plt.title("Espectros y Ajustes Exponenciales")
    plt.grid()
    plt.show()

    return parametros_a

# ========== INICIO DEL PROCESO ==========

carpeta = "2025_03_18_radiocromic_ocean_espectrometro"
datos = leer_archivos_txt(carpeta)
x_min = 660
x_max = 680

# Valores experimentales conocidos (14 valores para 14 archivos)
valores_x = [0.1, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# Ajustar y recolectar valores 'a'
parametros_a = graficar_datos_y_ajustes(datos, valores_x, x_min, x_max)


# Asegurarse de que los valores_x y los parámetros a tengan la misma longitud

# Filtrar para quedarnos solo con los 'a' válidos
# Separar 'a' y 'b' de los resultados
valores_filtrados = []
solo_a = []
solo_a_err = []
solo_b = []
solo_b_err = []

for i, (archivo, a, a_err, b, b_err) in enumerate(parametros_a):
    if None not in (a, a_err, b, b_err):
        valores_filtrados.append(valores_x[i])
        solo_a.append(a)
        solo_a_err.append(a_err)
        solo_b.append(b)
        solo_b_err.append(b_err)


fig, ax1 = plt.subplots(figsize=(8, 5))

# Eje izquierdo para 'a'
color_a = 'tab:blue'
ax1.set_xlabel("Dosis (Gy)")
ax1.set_ylabel("Parámetro a", color=color_a)
ax1.errorbar(valores_filtrados, solo_a, yerr=solo_a_err, fmt='o-', color=color_a, label="Parámetro a", capsize=4)
ax1.tick_params(axis='y', labelcolor=color_a)

# Eje derecho para 'b'
ax2 = ax1.twinx()
color_b = 'tab:red'
ax2.set_ylabel("Parámetro b", color=color_b)
ax2.errorbar(valores_filtrados, solo_b, yerr=solo_b_err, fmt='s--', color=color_b, label="Parámetro b", capsize=4)
ax2.tick_params(axis='y', labelcolor=color_b)

plt.title("Parámetros a y b [a * exp(-b * x) + c] vs Dosis (Gy)")
fig.tight_layout()
plt.grid(True)
plt.show()



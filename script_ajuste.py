import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def logaritmica(x, a, b, c):
    return a * np.log(x + b) + c


def ajustar_logaritmica(data, x_min, x_max):
    x = data[:, 0]
    y = data[:, 1]
    
    # Filtrar el rango
    mask = (x >= x_min) & (x <= x_max)
    x_rango = x[mask]
    y_rango = y[mask]

    # Evitar logaritmos inválidos (ej: b*x + c <= 0)
    # Un truco inicial es normalizar o usar valores pequeños de c
    try:
        popt, _ = curve_fit(logaritmica, x_rango, y_rango, p0=[-1, 1, 0])
        return popt  # Devuelve a, b, c
    except RuntimeError as e:
        print("No se pudo ajustar:", e)
        return None
    

def graficar_datos(datos, x_min=None, x_max=None):
    plt.figure(figsize=(10, 6))
    
    for archivo, data in datos.items():
        if data.ndim == 2 and data.shape[1] >= 2:
         x = data[:, 0]
         y = data[:, 1]
        plt.plot(x, y, label=archivo)
        if x_min and x_max:
            parametros = ajustar_logaritmica(data, x_min, x_max)
            if parametros is not None:
                a, b, c = parametros
                print(f"→ Archivo: {archivo}, ajuste: y = {a:.4f} * log({b:.4f} * x + {c:.4f})")
    
    plt.xscale('log')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Gráfica de todos los archivos TXT")
    plt.grid()
    plt.show()

# Carpeta donde están los archivos
carpeta = "2025_03_18_radiocromic_ocean_espectrometro"
datos = leer_archivos_txt(carpeta)

# Rango para el cálculo
x_min = 640
x_max = 680

graficar_datos(datos, x_min, x_max)
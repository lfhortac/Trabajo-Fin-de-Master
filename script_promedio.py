import os
import numpy as np
import matplotlib.pyplot as plt

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

def calcular_promedio_derivada(data, x_min, x_max):
    x = data[:, 0]
    y = data[:, 1]
    
    # Calcular la derivada numérica
    dy_dx = np.gradient(y, x)
    
    # Filtrar el rango
    mask = (x >= x_min) & (x <= x_max)
    
    # Calcular el promedio solo dentro del rango
    promedio = np.mean(dy_dx[mask])
    
    return promedio

def graficar_datos(datos, x_min=None, x_max=None):
    plt.figure(figsize=(10, 6))
    
    for archivo, data in datos.items():
        if data.ndim == 2 and data.shape[1] >= 2:
            x = data[:, 0]
            y = data[:, 1]
            plt.plot(x, y, label=archivo)
            if x_min and x_max:
                prom = calcular_promedio_derivada(data, x_min, x_max)
                print(f"→ Archivo: {archivo}, promedio de derivada: {prom:.6f}")
        else:
            print(f"Formato desconocido en {archivo}, no se grafica.")
    
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
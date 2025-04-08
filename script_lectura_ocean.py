import os
import numpy as np
import matplotlib.pyplot as plt

def leer_archivos_txt(carpeta):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.txt')]
    datos = {}
    
    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        try:
            data = np.loadtxt(ruta, skiprows=19)
            datos[archivo] = data
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
    
    return datos

def graficar_datos(datos):
    plt.figure(figsize=(10, 6))
    
    for archivo, data in datos.items():
        if data.ndim == 1:
            plt.plot(data, label=archivo)
        elif data.ndim == 2 and data.shape[1] >= 2:
            plt.plot(data[:, 0], data[:, 1], label=archivo)
        else:
            print(f"Formato desconocido en {archivo}, no se grafica.")
    
    plt.xlabel("X")
    #plt.yscale("log")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Gráfica de todos los archivos TXT")
    plt.grid()
    plt.show()

# Carpeta donde están los archivos
carpeta = "2025_03_18_espectrometro_FC_Ciencias"
datos = leer_archivos_txt(carpeta)
graficar_datos(datos)

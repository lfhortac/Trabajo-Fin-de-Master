#Este codigo es para leer los archivos de espectros y graficarlos, asumiendo que ya tenemos la densidad óptica calculada.
#Este codigo solo lee los archivos .txt y grafica la densidad óptica calculada.
#El formato de los archivos .txt es el siguiente:
#El primer bloque de 19 líneas es el encabezado, que no se necesita leer.
#A partir de la línea 20 se encuentran los datos, que son dos columnas: la primera es el número de puntos y la segunda es la densidad óptica.
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
    
    plt.xlim(400, 800) 
    plt.xlabel("X")
    #plt.yscale("log")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Gráfica de todos los archivos TXT")
    plt.grid()
    plt.show()

# Carpeta donde están los archivos
carpeta =r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\RC_nn\OD_resultados\suavizados' # Cambia esta ruta a la carpeta donde están los archivos .txt
datos = leer_archivos_txt(carpeta)
graficar_datos(datos)

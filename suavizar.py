import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Ruta donde están los archivos .txt
ruta=r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_radiocromic_ocean_espectrometro'
# Crear carpeta para guardar resultados suavizados (si no existe)
ruta_salida = os.path.join(ruta, 'suavizados')
os.makedirs(ruta_salida, exist_ok=True)

# Filtrar archivos .txt
archivos = sorted([f for f in os.listdir(ruta) if f.endswith('.txt')])

# Graficar
plt.figure(figsize=(12, 6))

for archivo in archivos:
    ruta_archivo = os.path.join(ruta, archivo)
    datos = np.genfromtxt(ruta_archivo, skip_header=19, max_rows=1000)
    if datos.ndim == 1:
        continue  # Si el archivo no tiene datos válidos, saltar
    elif datos.ndim == 2 and datos.shape[1] >= 2:
        pass  # Datos válidos
    else:
        print(f"Formato desconocido en {archivo}, no se grafica.")
        continue 
    x = datos[:, 0]
    y = datos[:, 1]

    # Aplicar suavizado
    y_suave = savgol_filter(y, window_length=11, polyorder=3)

    # Guardar archivo suavizado
    salida = np.column_stack((x, y_suave))
    nombre_salida = archivo.replace('.txt', '_suavizado.txt')
    np.savetxt(os.path.join(ruta_salida, nombre_salida), salida, fmt='%.6f')

    # Agregar a la gráfica
    plt.plot(x, y_suave, label=archivo)

plt.title("Espectros suavizados de las radiocromicas")
plt.xlabel("X")
plt.ylabel("Y suavizado")
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

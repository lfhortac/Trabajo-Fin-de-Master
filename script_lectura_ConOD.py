#Este codigo es para leer los archivos de espectros y graficarlos, asumiendo que ya tenemos la densidad óptica calculada.
#Este codigo solo lee los archivos .txt y grafica la densidad óptica calculada.
#El formato de los archivos .txt es el siguiente:
#El primer bloque de 19 líneas es el encabezado, que no se necesita leer.
#A partir de la línea 20 se encuentran los datos, que son dos columnas: la primera es el número de honda y la segunda es la densidad óptica.
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def leer_archivos_txt(carpeta):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.txt')]
    datos = {}
    
    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        try:
            data = np.loadtxt(ruta, skiprows=19,max_rows=1000)
            datos[archivo] = data
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
    
    return datos

def extract_dose_from_filename(filename):
    # Busca #dosis# o la primera cifra
    m = re.search(r'#(\d+\.?\d*)#', filename)
    if m:
        return float(m.group(1))
    m2 = re.search(r'(\d+\.?\d*)', filename)
    return float(m2.group(1)) if m2 else None

def graficar_datos(datos):
    # --- Configuración de estilo científico con LaTeX ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # --- Graficado ---
    fig, ax = plt.subplots(figsize=(6.6, 4.4))  # proporción 3:2

    for archivo, data in datos.items():
        if data.ndim == 1:
            ax.plot(
                data,
                lw=1.2,
                label=fr"{extract_dose_from_filename(archivo)}\,Gy"
            )
        elif data.ndim == 2 and data.shape[1] >= 2:
            ax.plot(
                data[:, 0], data[:, 1],
                lw=1.2,
                label=fr"{extract_dose_from_filename(archivo)}\,Gy"
            )
        else:
            print(f"Formato desconocido en {archivo}, no se grafica.")     

    # Ejes y etiquetas
    ax.set_xlabel(r'Longitud de onda (nm)')
    ax.set_ylabel(r'Densidad óptica (OD)')

    # Líneas de referencia
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    #ax.axvline(650, color='blue', linewidth=1, linestyle='-', label=r'$650\,\mathrm{nm}$')
    #ax.axvline(705, color='blue', linewidth=1, linestyle='-', label=r'$705\,\mathrm{nm}$')
    # Línea de referencia en 663 nm
    #ax.axvline(663, color='blue', linewidth=1.2, linestyle='--', label=r'$663\,\mathrm{nm}$')
    ax.set_xlim(400, 800)
    ax.set_ylim(-2, 4)

    # --- Región sombreada entre 650 y 705 nm ---
    #ax.axvspan(650, 705, color='blue', alpha=0.15, label=r'Región $650{-}705\,\mathrm{nm}$')

    # Estilo de ticks y grilla
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.5)

    # Leyenda
    ax.legend(fontsize=7, ncol=2, loc="upper left", frameon=False)

    fig.tight_layout()
    plt.show()

# Carpeta donde están los archivos
carpeta =r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Lorentz\2025_03_18_radiocromic_ocean_espectrometro' # Cambia esta ruta a la carpeta donde están los archivos .txt
datos = leer_archivos_txt(carpeta)
graficar_datos(datos)

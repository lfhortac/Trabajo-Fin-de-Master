import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import tkinter as tk
from tkinter import filedialog

# Función para definir un pico lorentziano
def lorentzian(x, amp, center, width):
    return amp * width**2 / ((x - center)**2 + width**2)

# Función para definir el modelo completo (suma de 5 lorentzianos)
def five_lorentzians(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        center = params[i+1]
        width = params[i+2]
        y = y + lorentzian(x, amp, center, width)
    return y

# Función para seleccionar el archivo
def seleccionar_archivo():
    root = tk.Tk()
    root.withdraw()
    file_path = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Lorentz\2025_03_18_radiocromic_ocean_espectrometro\suavizados\OD_Radicromic_n10_#10.0#_suavizado.txt"
    return file_path

# Función principal
def analizar_espectro(ruta_archivo=None):
    if ruta_archivo is None:
        ruta_archivo = seleccionar_archivo()
        if not ruta_archivo:
            print("No se seleccionó ningún archivo.")
            return
    
    # Leer el archivo, saltando las primeras 19 líneas
    try:
        datos = np.loadtxt(ruta_archivo, skiprows=19,max_rows=1040)
        print(f"Archivo cargado exitosamente: {ruta_archivo}")
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return
    
    # Extraer longitud de onda (x) y densidad óptica (y)
    longitud_onda = datos[:, 0]
    densidad_optica = datos[:, 1]
    
    # Definir el rango de interés
    rango_min = 650
    rango_max = 710
    
    # Encontrar índices para el rango de interés
    indices_rango = (longitud_onda >= rango_min) & (longitud_onda <= rango_max)
    x_rango = longitud_onda[indices_rango]
    y_rango = densidad_optica[indices_rango]
    
    # Valores iniciales para los 5 picos lorentzianos
    # [amplitud1, centro1, ancho1, amplitud2, centro2, ancho2, ...]
    p0 = [
        max(y_rango)*0.5, 660, 5,
        max(y_rango)*0.6, 670, 5,
        max(y_rango)*0.7, 680, 5,
        max(y_rango)*0.8, 690, 5,
        max(y_rango)*0.9, 700, 5
    ]
    
    # Límites para los parámetros (todos positivos, centros dentro del rango extendido)
    bounds_lower = [0, 640, 0] * 5
    bounds_upper = [np.inf, 720, 20] * 5
    
    try:
        # Ajustar los picos lorentzianos
        print("Ajustando picos lorentzianos...")
        params, _ = curve_fit(five_lorentzians, x_rango, y_rango, p0=p0, 
                             bounds=(bounds_lower, bounds_upper), maxfev=10000)
        print("Ajuste completado con éxito.")
    except Exception as e:
        print(f"Error en el ajuste: {e}")
        print("Usando parámetros iniciales para la visualización.")
        params = p0
    
        # --- Configuración estética científica con LaTeX ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # --- Crear figura ---
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=120)  # proporción 3:2

    # Espectro completo
    ax.plot(
        longitud_onda, densidad_optica,
        color='gray', lw=1.2, alpha=0.6,
        label=r'Espectro completo'
    )

    # Límites y sombreado de la región de interés
    ax.axvspan(rango_min, rango_max, alpha=0.1, color='gray')
    ax.axvline(rango_min, color='black', linestyle='--', lw=1,
            label=fr'Límite {rango_min}\,nm')
    ax.axvline(rango_max, color='black', linestyle='--', lw=1,
            label=fr'Límite {rango_max}\,nm')

    # Rango extendido para el ajuste
    x_extended = np.linspace(min(longitud_onda), max(longitud_onda), 1000)

    # Colores para picos individuales
    colores = ['m', 'g', 'b', 'c', 'y']

    # Picos individuales
    for i in range(0, len(params), 3):
        amp = params[i]
        center = params[i+1]
        width = params[i+2]

        y_pico = lorentzian(x_extended, amp, center, width)
        ax.plot(
            x_extended, y_pico,
            color=colores[i//3 % len(colores)],
            lw=1.0,
            label=fr'Pico {i//3+1}: {center:.1f}\,nm'
        )

    # Suma de todos los picos
    y_fit_total = five_lorentzians(x_extended, *params)
    ax.plot(
        x_extended, y_fit_total,
        color='red', lw=1.5,
        label=r'Suma de picos'
    )

    # --- Estética de la gráfica ---
    ax.set_xlim(450, 800)
    ax.set_ylim(-0.2, 2.6)
    ax.set_xlabel(r'Longitud de onda (nm)')
    ax.set_ylabel(r'Densidad óptica (OD)')
    ax.set_xticks(np.arange(450, 801, 25))
    ax.set_yticks(np.arange(-0.2, 2.61, 0.2))

    # Ticks y grilla
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.5)

    # Leyenda
    ax.legend(fontsize=8, frameon=False, loc='upper left', ncol=2)

    fig.tight_layout()
    plt.show()
    # Mostrar parámetros de los picos en consola
    print("\nParámetros de los picos lorentzianos:")
    for i in range(0, len(params), 3):
        print(f"Pico {i//3+1}: Amplitud = {params[i]:.4f}, Centro = {params[i+1]:.2f} nm, Ancho = {params[i+2]:.2f}")

if __name__ == "__main__":
    analizar_espectro()

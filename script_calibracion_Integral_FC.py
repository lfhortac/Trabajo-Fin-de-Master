import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Ruta a los resultados OD
od_path = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_espectrometro_FC_Ciencias\OD_resultados'

# Lista de dosis (orden debe coincidir con los archivos OD)
dosis = [0.1, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Leer archivos OD y calcular integrales en 660–680 nm
integraciones = []
archivos = sorted([f for f in os.listdir(od_path) if f.startswith('OD_') and f.endswith('.txt')])

for archivo in archivos:
    ruta = os.path.join(od_path, archivo)
    datos = pd.read_csv(ruta, sep=r'\s+', header=None, skiprows=1).values
    
    longitud_onda = datos[:, 0]
    od = datos[:, 1]
    
    # Filtrar rango 660–680 nm
    mascara = (longitud_onda >= 660) & (longitud_onda <= 680)
    x = longitud_onda[mascara]
    y = od[mascara]
    
    if len(x) == 0:
        print(f"No hay datos entre 660–680 nm en {archivo}, se omite.")
        continue

    integral = np.trapezoid(y, x)
    integraciones.append(integral)

# Validación de longitud
if len(integraciones) != len(dosis):
    print(" El número de espectros no coincide con el número de dosis.")
else:
    # Ajuste lineal
    slope, intercept, r_value, _, _ = linregress(dosis, integraciones)
    ajuste = [slope * val + intercept for val in integraciones]

    # Gráfica
    plt.figure(figsize=(8, 6))
    plt.scatter(dosis, integraciones, color='blue', label='Datos experimentales')
    # Nueva recta con ejes invertidos: y = m·x + b → x = (y - b)/m
    x_fit = np.linspace(min(dosis), max(dosis), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Ajuste lineal\nInt = {slope:.4f}·Dosis + {intercept:.4f}\n$R^2$ = {r_value**2:.4f}')
    plt.xlabel('Dosis (J/cm²)')
    plt.ylabel('Integral OD (660–680 nm)')
    plt.title('Curva de calibración: OD vs. Dosis')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
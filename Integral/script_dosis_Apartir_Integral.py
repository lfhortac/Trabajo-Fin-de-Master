# Este script calcula la dosis a partir de los OD generados por el espectrofotómetro,
# usando una regresión lineal (previamente calculada) y guarda los resultados en un CSV.

import os
import numpy as np
import pandas as pd
import json
import re
# --- Parámetros configurables ---
od_folder = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Datos19mayo\RC6\OD_resultados'
rango_min, rango_max = 649, 705
# Leer los parámetros de calibración desde un archivo JSON
with open('parametros_calibracion.json', 'r') as f:
    parametros = json.load(f)

slope_int = parametros['slope']
intercept_int = parametros['intercept']
slope_err = parametros['std_err']
intercept_err = parametros['intercept_err']
# Mostrar los parámetros de calibración
print(f"📈 Usando calibración: dosis = {slope_int:.4f} * área + {intercept_int:.4f}")
       # pendiente de la recta (inversa del coeficiente)


# --- Preparar procesamiento ---
resultados = []



def orden_natural(texto):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', texto)]



# --- Procesamiento de archivos ---
for archivo in sorted((f for f in os.listdir(od_folder) if f.endswith('.txt')), key=orden_natural):
    if not archivo.endswith('.txt'):
        continue

    ruta_archivo = os.path.join(od_folder, archivo)

    try:
        df = pd.read_csv(
            ruta_archivo,
            sep=r'\s+',
            engine='python',
            header=None,
            skiprows=19,
            nrows=1000, 
            usecols=(0, 1),
            comment='#'
        )
    except Exception as e:
        print(f"❌ Error al leer {archivo}: {e}")
        continue

    if df.shape[1] < 2:
        print(f"⚠️ Archivo {archivo} no tiene suficientes columnas.")
        continue

    try:
        x = df[0].astype(float).values
        y = df[1].astype(float).values
    except ValueError as e:
        print(f"⚠️ Error de conversión en {archivo}: {e}")
        continue

    # Filtrar el rango de interés
    mascara = (x >= rango_min) & (x <= rango_max)
    x_filtrado = x[mascara]
    y_filtrado = y[mascara]

    if len(x_filtrado) < 2:
        print(f"⚠️ Muy pocos datos en el rango {rango_min}-{rango_max} nm en {archivo}")
        continue

    # Integración y cálculo de dosis
    area = np.trapezoid(y_filtrado, x_filtrado)
    # Suponiendo x_filtrado es tu array de abscisas:
    x = x_filtrado
    n = len(x)
    w = np.empty(n)

    # Peso en el primer punto
    w[0] = (x[1] - x[0]) / 2
    # Pesos en los puntos intermedios
    w[1:-1] = (x[2:] - x[:-2]) / 2
    # Peso en el último punto
    w[-1] = (x[-1] - x[-2]) / 2

    # Ahora puedes calcular area_err (si sigma_y = 0.01 constante)
    area_err = 0.01 * np.sqrt(np.sum(w**2))
    # Calcular la dosis usando la regresión lineal
    if slope_int == 0:
        print(f"⚠️ Error: pendiente de la recta de calibración es cero en {archivo}. No se puede calcular la dosis.")
        continue
    dosis = 1 / slope_int * area + intercept_int / slope_int
                # Para y = m*x, la inversa es x = y/m
                # Derivada parcial
    # derivadas parciales
    d_d_m = -area/(slope_int**2)
    d_d_area = 1/slope_int
    d_d_b = -1/slope_int
    area_err = 0.01 * np.sqrt(np.sum(w**2))

    # propagación combinada
    error = np.sqrt(
    (d_d_m   * slope_err)**2 +
    (d_d_area* area_err )**2 +
    (d_d_b   * intercept_err)**2
    )


    print(f"✅ {archivo}: dosis = {dosis:.3f} Gy, error = {error:.3f} Gy")
    resultados.append({
        'Archivo': archivo,
        'Integrado_OD': area,
        'Dosis': dosis,
        'Error': error,
    })

# --- Guardar resultados ---
if resultados:
    df_resultados = pd.DataFrame(resultados)
    output_csv = os.path.join(od_folder, 'Dosis_resultados.csv')
    df_resultados.to_csv(output_csv, index=False)
    print(f"✅ Integración y cálculo de dosis completado. Resultados guardados en:\n{output_csv}")
else:
    print("⚠️ No se generaron resultados. Revisa los archivos de entrada.")

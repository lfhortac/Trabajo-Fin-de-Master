#Este script calcula la dosis a partir de los OD generados por el espectrófotometro,
#usando una regresion lineal(prebiamnete calculada en otro script) y guarda los resultados en un CSV.
import os
import numpy as np
import pandas as pd

# Ruta a los OD generados
od_folder = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_espectrometro_FC_Ciencias\OD_resultados'

# Lista para guardar resultados
resultados = []

# Recorrer archivos
for archivo in os.listdir(od_folder):
    if archivo.endswith('.txt'):
        ruta_archivo = os.path.join(od_folder, archivo)
        try:
            df = pd.read_csv(
                ruta_archivo,
                sep=r'\s+',
                engine='python',
                header=None,
                skiprows=1,      #  Saltamos la línea con 'Wavelength OD'
                comment='#'
            )
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
            continue

        if df.shape[1] < 2:
            print(f"Archivo {archivo} no tiene suficientes columnas.")
            continue

        try:
            longitud_onda = df[0].astype(float).values
            od = df[1].astype(float).values
        except ValueError as e:
            print(f"Error convirtiendo a float en {archivo}: {e}")
            continue

        # Filtro de rango
        mascara = (longitud_onda >= 660) & (longitud_onda <= 680)
        x = longitud_onda[mascara]
        y = od[mascara]

        if len(x) == 0:
            print(f"No hay datos en el rango para {archivo}")
            continue

        # Integración
        valor_integrado = np.trapezoid(y, x)

        # Cálculo de dosis
        dosis = (valor_integrado / 0.57) + (0.37 / 0.57)  # Ajustar según la regresión lineal
        # Guardar resultados

        resultados.append({
            'Archivo': archivo,
            'Integrado_OD': valor_integrado,
            'Dosis': dosis
        })

# Guardar CSV
df_resultados = pd.DataFrame(resultados)
output_csv = os.path.join(od_folder, 'Dosis_resultados.csv')
df_resultados.to_csv(output_csv, index=False)

print("Integración y cálculo de dosis completado correctamente.")

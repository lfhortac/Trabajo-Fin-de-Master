import os
import numpy as np
import pandas as pd

# Ruta a los OD generados
od_folder = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_radiocromic_ocean_espectrometro'

# Lista para guardar resultados
resultados = []



def despejar_dosis(area_integrada):
    a = 0.01
    b = 0.45
    c = -0.14 - area_integrada

    discriminante = b**2 - 4 * a * c

    if discriminante < 0:
        return None  # No hay soluciones reales

    sqrt_disc = np.sqrt(discriminante)
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)

    return x1, x2  # Devuelve ambas soluciones


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

        # Cálculo de dosis con función inversa cuadrática
        dosis_roots = despejar_dosis(valor_integrado)

        if dosis_roots is None:
            print(f"No hay solución real para {archivo}")
            continue

        positivas = [x for x in dosis_roots if x >= 0]

        if not positivas:
            print(f"Ninguna solución positiva para {archivo}")
            continue

        dosis_valida = min(positivas)

        resultados.append({
            'Archivo': archivo,
            'Integrado_OD': valor_integrado,
            'Dosis_1': dosis_roots[0],
            'Dosis_2': dosis_roots[1],
            'Dosis_valida': dosis_valida
        })

# Guardar CSV
df_resultados = pd.DataFrame(resultados)
output_csv = os.path.join(od_folder, 'Dosis_resultados.csv')
df_resultados.to_csv(output_csv, index=False)

print("Integración y cálculo de dosis completado correctamente.")

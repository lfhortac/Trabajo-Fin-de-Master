import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from natsort import natsorted


# --- ANTES DE PROCESAR ARCHIVOS ---
dosis = [0.1, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10, 12, 16, 18, 20]  # Ajusta según corresponda
parametros_b = []

# --- DEFINIR FUNCIÓN GAUSSIANA ---
def gauss(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

# --- CONFIGURACIÓN ---
carpeta = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_radiocromic_ocean_espectrometro"
rango_min = 610
rango_max = 650

# Lista para guardar resultados
resultados = []

# Leer todos los archivos .txt
archivos = natsorted([f for f in os.listdir(carpeta) if f.endswith('.txt')])

for archivo in archivos:
    ruta = os.path.join(carpeta, archivo)
    try:
        data = pd.read_csv(ruta, sep=r'\s+', skiprows=1, header=None).values
        x = data[:, 0]
        y = data[:, 1]

        # Filtrar rango 610–680
        mask = (x >= rango_min) & (x <= rango_max)
        x_rango = x[mask]
        y_rango = y[mask]

        if len(x_rango) < 3:
            print(f"⚠️ Muy pocos puntos en el rango para {archivo}, se omite.")
            continue

        # Parámetros iniciales
        a_ini = max(y_rango)
        b_ini = x_rango[np.argmax(y_rango)]
        c_ini = 10

        # Ajuste Gaussiano
        popt, _ = curve_fit(gauss, x_rango, y_rango, p0=[a_ini, b_ini, c_ini])
        a, b, c = popt

        resultados.append({
            'Archivo': archivo,
            'a': a,
            'b (nm)': b,
            'c': c
        })

        # --- DENTRO DEL BUCLE DE PROCESAMIENTO, después del ajuste exitoso ---
        # ...
        a, b, c = popt
        resultados.append({'Archivo': archivo, 'a': a, 'b (nm)': b, 'c': c})
        parametros_b.append(b)  # ← Guardamos 'a' directamente

     

    except Exception as e:
        print(f"❌ Error procesando {archivo}: {e}")

# --- Guardar resultados ---
df_resultados = pd.DataFrame(resultados)
output_csv = os.path.join(carpeta, 'Dosis_estimadas_por_gauss.csv')
df_resultados.to_csv(output_csv, index=False)   


if len(dosis) != len(parametros_b):
    print("❌ La cantidad de dosis no coincide con la cantidad de espectros procesados.")
else:
    plt.figure(figsize=(8, 5))
    plt.plot(dosis, parametros_b, 'o-', color='green', label='Altura Gaussiana (a)')
    plt.xlabel('Dosis (Gy)')
    plt.ylabel('Altura del Pico OD (a)')
    plt.title('Dosis vs. Altura de Pico (Gauss)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- FUNCIÓN LOGARÍTMICA ---
def log_func(x, a, b, c):
    return a * np.log(x + b) + c

# --- VALORES INICIALES (ajustables) ---
a_ini = 1
b_ini = 1
c_ini = 0

# --- AJUSTE ---
popt, pcov = curve_fit(log_func, dosis, parametros_b, p0=[a_ini, b_ini, c_ini])
a_fit, b_fit, c_fit = popt

# --- GENERAR CURVA AJUSTADA ---
x_fit = np.linspace(min(dosis), max(dosis), 300)
y_fit = log_func(x_fit, a_fit, b_fit, c_fit)

# --- CALCULAR R² ---
residuos = np.array(parametros_b) - log_func(np.array(dosis), *popt)
ss_res = np.sum(residuos**2)
ss_tot = np.sum((parametros_b - np.mean(parametros_b))**2)
r2 = 1 - (ss_res / ss_tot)

# --- GRAFICAR ---
plt.figure(figsize=(8, 5))
plt.plot(dosis, parametros_b, 'o', label='Datos (a)', color='blue')
plt.plot(x_fit, y_fit, 'r--', label=f'Ajuste Logarítmico\n$a$={a_fit:.2f}, $b$={b_fit:.2f}, $c$={c_fit:.2f}\n$R^2$ = {r2:.4f}')
plt.xlabel('Dosis (Gy)')
plt.ylabel('Altura del Pico OD (a)')
plt.title('a * np.log(x + b) + c: Dosis vs Altura Pico')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

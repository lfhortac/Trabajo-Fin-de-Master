import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress



def leer_archivos_txt(carpeta):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.txt')]
    datos = {}
    
    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        try:
            data = np.loadtxt(ruta, skiprows=17)
            datos[archivo] = data
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
    
    return datos

def integrar_area(data, x_min, x_max):
    x = data[:, 0]
    y = data[:, 1]
    mask = (x >= x_min) & (x <= x_max)
    x_rango = x[mask]
    y_rango = y[mask]

    if len(x_rango) < 2:
        print("⚠️ Muy pocos puntos para integrar.")
        return None

    area = np.trapz(y_rango, x_rango)
    return area

# --- PARTE PRINCIPAL ---
x_min = 660 
x_max = 680
carpeta = "2025_03_18_radiocromic_ocean_espectrometro"
datos = leer_archivos_txt(carpeta)

valores_x = [0.1, 0.3, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18,20]

areas = []
for i, archivo in enumerate(sorted(datos.keys())):
    data = datos[archivo]
    area = integrar_area(data, x_min, x_max)
    if area is not None:
        areas.append((archivo, area))
        print(f"→ {archivo}: Área = {area:.4f}")
    else:
        areas.append((archivo, None))

# --- GRAFICAR TODAS LAS CURVAS ---
def graficar_datos(datos):
    plt.figure(figsize=(10, 6))
    for archivo, data in datos.items():
        if data.ndim == 2 and data.shape[1] >= 2:
            plt.plot(data[:, 0], data[:, 1], label=archivo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Gráfica de todos los archivos TXT")
    plt.grid()
    plt.show()

graficar_datos(datos)

# --- GRAFICAR INTEGRALES ---
valores_filtrados = []
areas_filtradas = []

for i, (archivo, area) in enumerate(areas):
    if area is not None:
        valores_filtrados.append(valores_x[i])
        areas_filtradas.append(area)

# Ajuste lineal
slope, intercept, r_value, p_value, std_err = linregress(valores_filtrados, areas_filtradas)

print(f"Recta ajustada: Área = {slope:.4f} * Dosis + {intercept:.4f}")
print(f"R² = {r_value**2:.4f}")



x_fit = np.linspace(min(valores_filtrados), max(valores_filtrados), 200)
y_fit = slope * x_fit + intercept

plt.figure(figsize=(8, 5))
plt.plot(valores_filtrados, areas_filtradas, 'o', label="Área integrada")
plt.plot(x_fit, y_fit, 'r--', label=f"Ajuste lineal: y = {slope:.2f}x + {intercept:.2f}")
plt.xlabel("Dosis (Gy)")
plt.ylabel("Área bajo la curva (660–680)")
plt.title("Área vs Dosis con Ajuste Lineal")
plt.grid(True)
plt.legend()
plt.show()
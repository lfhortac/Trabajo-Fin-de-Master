# -*- coding: utf-8 -*-
"""
Este script integra el área bajo la curva de espectros radiocrómicos en un rango dado
y ajusta una recta a (Dosis, Área). Produce figura con barras de error y estilo científico.
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ---------- CONFIGURACIÓN ----------
x_min = 649
x_max = 705
carpeta = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Integral\2025_04_28_calibracionFebrero'

# Si conoces la dosis de cada archivo (orden natural), ponla aquí:
valores_x = [0.1, 0.3, 0.5, 0.7, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Si tienes incertidumbres por punto (misma longitud que valores_x) ponlas aquí (opcional):
areas_err = None  # ejemplo: [0.01, 0.01, ...]  # o deja en None

# ---------- UTILIDADES ----------
def orden_natural(texto):
    """Orden tipo humano: archivo2.txt < archivo10.txt."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', texto)]

def leer_archivos_txt(carpeta):
    """Lee todos los .txt de la carpeta, ordenados de forma natural."""
    archivos = sorted((f for f in os.listdir(carpeta) if f.lower().endswith('.txt')),
                      key=orden_natural)
    datos = {}
    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        try:
            # Ajusta skiprows / max_rows si tu formato cambia
            data = np.loadtxt(ruta, skiprows=19, max_rows=1000, usecols=(0, 1))
            if data.ndim == 1:  # por seguridad, re-forma a Nx2
                data = data.reshape(-1, 2)
            datos[archivo] = data
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
    return datos

def integrar_area(data, x_min, x_max):
    """Integra el área por trapecios en [x_min, x_max]."""
    x = data[:, 0]
    y = data[:, 1]
    mask = (x >= x_min) & (x <= x_max)
    x_rango = x[mask]
    y_rango = y[mask]
    if x_rango.size < 2:
        print("⚠️ Muy pocos puntos para integrar en el rango.")
        return None
    return np.trapezoid(y_rango, x_rango)

def graficar_espectros(datos):
    """Grafica todos los espectros con estilo LaTeX."""
    plt.rcParams.update({
        "text.usetex": True,        # comenta esta línea si no tienes LaTeX instalado
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=120)
    for archivo, data in datos.items():
        if data.ndim == 2 and data.shape[1] >= 2:
            ax.plot(data[:, 0], data[:, 1], lw=1.0, label=archivo)
    ax.set_xlabel(r'Longitud de onda (nm)')
    ax.set_ylabel(r'Densidad óptica (OD)')
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc='best')
    fig.tight_layout()
    plt.show()

# ---------- LECTURA E INTEGRACIÓN ----------
datos = leer_archivos_txt(carpeta)

# Reporte rápido
if not datos:
    raise RuntimeError("No se encontraron archivos .txt legibles en la carpeta especificada.")

# Integra cada archivo (orden natural) y asocia a valores_x
archivos_ordenados = sorted(datos.keys(), key=orden_natural)

if len(valores_x) != len(archivos_ordenados):
    print(f"⚠️ Atención: número de dosis ({len(valores_x)}) ≠ número de archivos ({len(archivos_ordenados)}).")
    print("   Se tomarán solo los primeros min(n_archivos, n_dosis) pares en orden natural.")
N = min(len(valores_x), len(archivos_ordenados))

areas = []
for i in range(N):
    archivo = archivos_ordenados[i]
    data = datos[archivo]
    area = integrar_area(data, x_min, x_max)
    areas.append((archivo, area))
    if area is not None:
        print(f"→ {archivo}: Área = {area:.6f}")
    else:
        print(f"→ {archivo}: Área = None")

# ---------- PREPARACIÓN PARA AJUSTE ----------
valores_filtrados = []
areas_filtradas = []

for i, (archivo, area) in enumerate(areas):
    if area is not None:
        valores_filtrados.append(valores_x[i])
        areas_filtradas.append(area)

valores_filtrados = np.asarray(valores_filtrados, float)
areas_filtradas = np.asarray(areas_filtradas, float)

if valores_filtrados.size < 2:
    raise RuntimeError("No hay suficientes puntos válidos para ajustar una recta.")

# ---------- AJUSTE LINEAL + INCERTIDUMBRES ----------
res = linregress(valores_filtrados, areas_filtradas)
slope = res.slope
intercept = res.intercept
r_value = res.rvalue
slope_err = res.stderr

# intercept_stderr en SciPy >= 1.7
if hasattr(res, "intercept_stderr") and (res.intercept_stderr is not None):
    intercept_err = res.intercept_stderr
else:
    yhat = slope * valores_filtrados + intercept
    s_res = np.sqrt(np.sum((areas_filtradas - yhat)**2) / (len(valores_filtrados) - 2))
    Sxx = np.sum((valores_filtrados - valores_filtrados.mean())**2)
    intercept_err = s_res * np.sqrt(1/len(valores_filtrados) + (valores_filtrados.mean()**2)/Sxx)

print("\n--- Recta de calibración ---")
print(f"Área = ({slope:.6f} ± {slope_err:.6f})·Dosis + ({intercept:.6f} ± {intercept_err:.6f})")
print(f"R² = {r_value**2:.6f}")

# Curva del ajuste
x_fit = np.linspace(valores_filtrados.min(), valores_filtrados.max(), 400)
y_fit = slope * x_fit + intercept

# ---------- BARRAS DE ERROR EN LOS DATOS ----------
# Si no hay areas_err, usar RMSE del ajuste como barra visual homogénea
rmse = np.sqrt(np.sum((areas_filtradas - (slope*valores_filtrados + intercept))**2) / (len(valores_filtrados) - 2))
if isinstance(areas_err, (list, tuple, np.ndarray)) and len(areas_err) == len(areas_filtradas):
    yerr_plot = np.asarray(areas_err, float)
else:
    yerr_plot = np.full_like(areas_filtradas, rmse)

# ---------- GUARDAR RESULTADOS ----------
# JSON con parámetros
parametros = {
    'slope': float(slope),
    'intercept': float(intercept),
    'slope_err': float(slope_err),
    'intercept_err': float(intercept_err),
    'r2': float(r_value**2),
    'x_min': float(x_min),
    'x_max': float(x_max),
}
with open('parametros_calibracion.json', 'w', encoding='utf-8') as f:
    json.dump(parametros, f, ensure_ascii=False, indent=2)

# CSV con áreas por archivo
with open('areas_integradas.csv', 'w', encoding='utf-8') as f:
    f.write('archivo,dosis_Gy,area\n')
    for i in range(N):
        archivo = archivos_ordenados[i]
        dosis = valores_x[i]
        area = areas[i][1]
        val = "" if area is None else f"{area:.8f}"
        f.write(f"{archivo},{dosis},{val}\n")

print("\nArchivos guardados: parametros_calibracion.json, areas_integradas.csv")

# ---------- GRÁFICAS ----------
# (1) Espectros (opcional)
graficar_espectros(datos)

# (2) Dispersión + ajuste con estilo científico
plt.rcParams.update({
    "text.usetex": True,      # comenta si no tienes LaTeX instalado
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=120)

# Datos con barras de error
ax.errorbar(
    valores_filtrados, areas_filtradas, yerr=yerr_plot,
    fmt='o', mfc='white', mec='black', color='black',
    ecolor='black', elinewidth=1, capsize=3, markersize=6,
    label=r'Área integrada'
)

# Línea de ajuste (continua)
ax.plot(x_fit, y_fit, '-', lw=1.5, color='red', label=r'Ajuste lineal')

# Ejes y estilo
ax.set_xlabel(r'Dosis (Gy)')
ax.set_ylabel(r'Área bajo la curva')
ax.set_xticks(np.arange(0, max(22.1, valores_filtrados.max()+0.1), 2.0))
ax.tick_params(direction='in', which='both')
ax.minorticks_on()
ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.5)

# Anotación
texto = (
    rf"$y = ({slope:.3f}\pm{slope_err:.3f})\,x + ({intercept:.3f}\pm{intercept_err:.3f})$" "\n"
    rf"$R^2 = {r_value**2:.3f}$"
)
ax.annotate(
    texto, xy=(0.98, 0.02), xycoords='axes fraction',
    ha='right', va='bottom', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.6)
)

ax.legend(frameon=False, fontsize=9, loc='upper left')
fig.tight_layout()

# Guardar figura final
fig.savefig('calibracion_area_vs_dosis.png', dpi=300, bbox_inches='tight')
print("Figura guardada: calibracion_area_vs_dosis.png")

plt.show()

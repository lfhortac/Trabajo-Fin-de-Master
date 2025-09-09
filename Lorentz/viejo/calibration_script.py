import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import pickle
import re
import glob

#este codigo esta funcionando con 2 picos y el rango de 650 y 710, maxfev=50000
#     # Parámetros iniciales para n_peaks lorentzianas

def lorentzian(x, amplitude, center, width):
    return (2 * amplitude / np.pi) * (width / (4 * (x - center)**2 + width**2))

def multiple_lorentzians(x, *params):
    """Función para múltiples lorentzianas"""
    n_peaks = len(params) // 3
    result = np.zeros_like(x)
    for i in range(n_peaks):
        amplitude = params[3*i]
        center = params[3*i + 1]
        width = params[3*i + 2]
        result += lorentzian(x, amplitude, center, width)
    return result


x_min = 650.0
x_max = 710.0

def read_spectrophotometer_file(filepath):
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Ajusta este número si realmente quieres omitir 19 líneas
            raw_lines = file.readlines()[19:]

        wavelengths = []
        optical_densities = []
        
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                wl = float(parts[0])
                od = float(parts[1])
            except ValueError:
                # Línea no numérica: la saltamos
                continue

            # Ahora sí podemos comparar numéricamente
            if x_min < wl < x_max:
                wavelengths.append(wl)
                optical_densities.append(od)
        
        return np.array(wavelengths), np.array(optical_densities)

    except Exception as e:
        print(f"Error leyendo archivo {filepath}: {e}")
        # Devuelve arrays vacíos para facilitar manejos posteriores
        return np.array([]), np.array([])

def extract_calibration_value(filename):
    """Extrae el valor numérico del nombre del archivo para calibración"""
    # Buscar patrón como #10.0# en el nombre del archivo
    pattern = r'#(\d+\.?\d*)#'
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    
    # Patrón alternativo: buscar números después de 'n'
    pattern2 = r'n(\d+\.?\d*)'
    match2 = re.search(pattern2, filename)
    if match2:
        return float(match2.group(1))
    
    print(f"No se pudo extraer valor de calibración de: {filename}")
    return None

def fit_lorentzians(wavelengths, optical_densities, n_peaks=5):
    """Ajusta múltiples curvas lorentzianas a los datos"""
    # Estimación inicial de parámetros
    max_od = np.max(optical_densities)
    min_wl, max_wl = np.min(wavelengths), np.max(wavelengths)
    
    # Parámetros iniciales para n_peaks lorentzianas
    initial_params = []
    for i in range(n_peaks):
        amplitude = max_od / n_peaks
        center = min_wl + (i + 1) * (max_wl - min_wl) / (n_peaks + 1)
        width = (max_wl - min_wl) / (n_peaks * 4)
        initial_params.extend([amplitude, center, width])
    
    try:
        # Ajuste de curvas
        popt, pcov = curve_fit(multiple_lorentzians, wavelengths, optical_densities, 
                              p0=initial_params, maxfev=500000)
        return popt, pcov
    except Exception as e:
        print(f"Error en ajuste de curvas: {e}")
        return None, None

def process_calibration_data(data_directory):
    """Procesa todos los archivos de calibración"""
    # Buscar archivos .txt en el directorio
    txt_files = glob.glob(os.path.join(data_directory, "*.txt"))
    
    if not txt_files:
        print(f"No se encontraron archivos .txt en {data_directory}")
        return None, None, None
    
    calibration_values = []
    lorentzian_sums = []
    all_spectra = {}
    
    plt.figure(figsize=(12, 8))
    
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        print(f"Procesando: {filename}")
        
        # Leer datos
        wavelengths, optical_densities = read_spectrophotometer_file(filepath)
        if wavelengths is None:
            continue
        
        # Extraer valor de calibración
        cal_value = extract_calibration_value(filename)
        if cal_value is None:
            continue
        
        # Ajustar curvas lorentzianas
        params, _ = fit_lorentzians(wavelengths, optical_densities)
        if params is None:
            continue
        
        # Calcular suma de lorentzianas ajustadas
        fitted_curve = multiple_lorentzians(wavelengths, *params)
        lorentzian_sum = np.sum(fitted_curve)
        
        calibration_values.append(cal_value)
        lorentzian_sums.append(lorentzian_sum)
        all_spectra[filename] = (wavelengths, optical_densities, fitted_curve)
        
        # Graficar espectro original
        plt.plot(wavelengths, optical_densities, 'o-', alpha=0.7, 
                label=f'{filename} (cal: {cal_value})', markersize=2)
    
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Densidad Óptica')
    plt.title('Espectros de Calibración')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return np.array(calibration_values), np.array(lorentzian_sums), all_spectra

def perform_linear_regression(calibration_values, lorentzian_sums):
    """Realiza regresión lineal para el modelo de calibración"""
    if len(calibration_values) < 2:
        print("Se necesitan al menos 2 puntos para regresión lineal")
        return None
    
    # Preparar datos para regresión
    X = calibration_values.reshape(-1, 1)
    y = lorentzian_sums
    
    # Crear y ajustar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Calcular R²
    r_squared = model.score(X, y)
    
    print(f"Parámetros del modelo de calibración:")
    print(f"Pendiente: {model.coef_[0]:.6f}")
    print(f"Intersección: {model.intercept_:.6f}")
    print(f"R²: {r_squared:.6f}")
    
    # Graficar regresión
    plt.figure(figsize=(10, 6))
    plt.scatter(calibration_values, lorentzian_sums, color='blue', alpha=0.7)
    
    # Línea de regresión
    x_line = np.linspace(min(calibration_values), max(calibration_values), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, 'r-', linewidth=2, 
             label=f'y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}\nR² = {r_squared:.4f}')
    
    plt.xlabel('Valor de Calibración')
    plt.ylabel('Suma de Curvas Lorentzianas')
    plt.title('Regresión Lineal - Modelo de Calibración')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model

def save_calibration_model(model, filename='calibration_model.pkl'):
    """Guarda el modelo de calibración"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo guardado en: {filename}")
    except Exception as e:
        print(f"Error guardando modelo: {e}")

# Función principal
def main():
    # Directorio con archivos de calibración
    data_directory = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\2025_03_18_radiocromic_ocean_espectrometro\suavizados"  # Cambiar por tu directorio
    
    print("=== SCRIPT DE CALIBRACIÓN ===")
    print(f"Procesando archivos en: {data_directory}")
    
    # Crear directorio de ejemplo si no existe
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Directorio creado: {data_directory}")
        print("Por favor, coloca tus archivos .txt de calibración en este directorio")
        return
    
    # Procesar datos de calibración
    cal_values, lorentz_sums, spectra = process_calibration_data(data_directory)
    
    if cal_values is None or len(cal_values) == 0:
        print("No se pudieron procesar los datos de calibración")
        return
    
    # Realizar regresión lineal
    model = perform_linear_regression(cal_values, lorentz_sums)
    
    if model is not None:
        # Guardar modelo
        save_calibration_model(model)
        print("¡Calibración completada exitosamente!")
    else:
        print("Error en la regresión lineal")

if __name__ == "__main__":
    main()

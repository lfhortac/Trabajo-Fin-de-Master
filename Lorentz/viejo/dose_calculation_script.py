import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
import glob
import pandas as pd



x_min = 650.0
x_max = 710.0

#def lorentzian(x, amplitude, center, width):
#    """Función lorentziana para ajuste de curvas"""
#    return amplitude / (1 + ((x - center) / width) ** 2)


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
            # Filtrar por rango de longitudes de onda
            if x_min < wl < x_max:
                wavelengths.append(wl)
                optical_densities.append(od)
        
        return np.array(wavelengths), np.array(optical_densities)
    except Exception as e:
        print(f"Error leyendo archivo {filepath}: {e}")
        # Devuelve arrays vacíos para facilitar manejos posteriores
        return np.array([]), np.array([])

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
        popt, pcov = curve_fit(multiple_lorentzians, wavelengths, optical_densities, p0=initial_params, maxfev=50000)
        return popt, pcov
    except Exception as e:
        print(f"Error en ajuste de curvas: {e}")
        return None, None

def load_calibration_model(filename='calibration_model.pkl'):
    """Carga el modelo de calibración guardado"""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo cargado desde: {filename}")
        print(f"Pendiente: {model.coef_[0]:.6f}")
        print(f"Intersección: {model.intercept_:.6f}")
        return model
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None

def calculate_dose(lorentzian_sum, model):
    """Calcula la dosis usando el modelo de calibración"""
    # Aplicar modelo inverso: dosis = (suma - intersección) / pendiente
    dose = (lorentzian_sum - model.intercept_) / model.coef_[0]
    std = np.std(dose, ddof=2)  # Desviación estándar de la dosis
    if std < 0:
        std = 0.0  # Asegurar que la desviación estándar no sea negativa
    return dose, std

def process_dose_files(data_directory, model):
    """Procesa archivos para cálculo de dosis"""
    # Buscar archivos .txt en el directorio
    txt_files = glob.glob(os.path.join(data_directory, "*.txt"))
    
    if not txt_files:
        print(f"No se encontraron archivos .txt en {data_directory}")
        return None
    
    results = []
    
    plt.figure(figsize=(12, 8))
    
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        print(f"Procesando: {filename}")
        
        # Leer datos
        wavelengths, optical_densities = read_spectrophotometer_file(filepath)
        if wavelengths is None:
            continue
        
        # Ajustar curvas lorentzianas
        params, _ = fit_lorentzians(wavelengths, optical_densities)
        if params is None:
            continue
        
        # Calcular suma de lorentzianas ajustadas
        fitted_curve = multiple_lorentzians(wavelengths, *params)
        lorentzian_sum = np.sum(fitted_curve)
        
        # Calcular dosis
        
        dose, std = calculate_dose(lorentzian_sum, model)
        

        
        results.append({
            'filename': filename,
            'lorentzian_sum': lorentzian_sum,
            'calculated_dose': dose,
            'dose_std': std
        })
        
        # Graficar espectro
        plt.plot(wavelengths, optical_densities, 'o-', alpha=0.7, 
                label=f'{filename} (dosis: {dose:.3f})', markersize=3)
        plt.plot(wavelengths, fitted_curve, '--', alpha=0.8)
    
    plt.xlabel('Longitud de onda (nm)')
    plt.ylabel('Densidad Óptica')
    plt.title('Espectros de Muestra y Ajustes Lorentzianos')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results

def save_dose_results(results, filename='dose_results.csv'):
    """Guarda los resultados de dosis en un archivo CSV"""
    if not results:
        print("No hay resultados para guardar")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Resultados guardados en: {filename}")
    
    # Mostrar resumen
    print("\n=== RESUMEN DE RESULTADOS ===")
    for result in results:
        print(f"{result['filename']}: {result['calculated_dose']:.3f} ± {result['dose_std']:.3f} unidades de dosis")

def main():
    print("=== SCRIPT DE CÁLCULO DE DOSIS ===")
    
    # Cargar modelo de calibración
    model = load_calibration_model()
    if model is None:
        print("No se pudo cargar el modelo de calibración")
        print("Asegúrate de haber ejecutado primero el script de calibración")
        return
    
    # Directorio con archivos de muestra
    data_directory = r"C:\Users\luis-\Downloads\TFM\DatosEspectrometria\Lorentz\2025_04_28_RC#15del_2025_03_25\suavizados"  # Cambiar por tu directorio
    
    print(f"Procesando archivos en: {data_directory}")
    
    # Crear directorio de ejemplo si no existe
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Directorio creado: {data_directory}")
        print("Por favor, coloca tus archivos .txt de muestra en este directorio")
        return
    
    # Procesar archivos y calcular dosis
    results = process_dose_files(data_directory, model)
    
    if results:
        # Guardar resultados
        save_dose_results(results)
        print("¡Cálculo de dosis completado exitosamente!")
    else:
        print("No se pudieron procesar los archivos de muestra")

if __name__ == "__main__":
    main()

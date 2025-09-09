import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt # Solo si quieres visualizar algo

def read_spectrophotometer_file(filepath, header_lines=19):
    # (Misma función que en tu script de calibración)
    wavelengths, optical_densities = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[header_lines:]
    for line in lines:
        parts = line.strip().replace(',', '.').split()
        if len(parts) < 2: continue
        try:
            wl, od = float(parts[0]), float(parts[1])
            wavelengths.append(wl)
            optical_densities.append(od)
        except ValueError: continue
    return np.array(wavelengths), np.array(optical_densities)

def find_od_at_wavelength(wavelengths, ods, target=663.0, tol=0.5):
    # (Misma función, pero ahora devuelve OD y su desviación estándar como error)
    mask = np.abs(wavelengths - target) <= tol
    if np.any(mask):
        # El error de la OD es la desviación estándar de las mediciones cercanas
        return np.mean(ods[mask]), np.std(ods[mask])
    return None, None
def parse_params_file(params_filepath):
    """Lee el archivo de parámetros y devuelve los valores y sus errores."""
    params = {}
    with open(params_filepath, 'r', encoding='utf-8') as f:  # fuerza UTF-8
        for line in f:
            if '=' in line:
                key, val_str = line.split('=')
                key = key.strip()
                if '±' in val_str or 'Â±' in val_str:   # contempla ambas
                    val_str = val_str.replace('Â', '')  # limpia caracteres raros
                    val, err = val_str.split('±')
                    # Limpieza por seguridad
                    val = re.sub(r'[^\d\.\-+eE]', '', val).strip()
                    err = re.sub(r'[^\d\.\-+eE]', '', err).strip()
                    params[key] = float(val)
                    params[f"sigma_{key}"] = float(err)
                else:
                    # Caso sin error asociado (ej. R2)
                    try:
                        params[key] = float(val_str.strip())
                    except ValueError:
                        pass  # ignora si no es numérico
    # Crea la matriz de covarianza (asumiendo no correlación por simplicidad)
    pcov = np.diag([
        params.get("sigma_a", 0)**2,
        params.get("sigma_b", 0)**2,
        params.get("sigma_c", 0)**2
    ])
    return params, pcov

def calculate_dose_from_od(od_value, params):
    """Invierte el modelo exponencial para encontrar la dosis."""
    a = params['a']
    b = params['b']
    c = params['c']
    
    # Prevenir logaritmos de números no positivos
    if (od_value - c) / a <= 0:
        return np.nan
        
    dose = (1/b) * np.log((od_value - c) / a)
    return dose

def propagate_error_dose(od_value, sigma_od, params, pcov):
    """Propaga el error desde los parámetros y la OD hasta la dosis."""
    a, b, c = params['a'], params['b'], params['c']
    sigma_a, sigma_b, sigma_c = params['sigma_a'], params['sigma_b'], params['sigma_c']
    
    # Derivadas parciales de la función de dosis D(a, b, c, OD)
    # D = (1/b) * log((OD-c)/a)
    dD_da = -1 / (a * b)
    dD_db = - (1/b**2) * np.log((od_value - c) / a)
    dD_dc = -1 / (b * (od_value - c))
    dD_dOD = 1 / (b * (od_value - c))
    
    # Varianza total (suma de contribuciones)
    # Usamos la matriz de covarianza para una propagación más precisa
    # Si pcov no tiene correlaciones, esto es igual a la suma de (derivada * sigma)**2
    var_a = dD_da**2 * pcov[0, 0]
    var_b = dD_db**2 * pcov[1, 1]
    var_c = dD_dc**2 * pcov[2, 2]
    # Añadimos la incertidumbre de la propia medición de OD
    var_od = dD_dOD**2 * sigma_od**2

    total_variance = var_a + var_b + var_c + var_od
    return np.sqrt(total_variance)


def parse_params_file(params_filepath):
    """Lee el archivo de parámetros y devuelve los valores y el SER."""
    params = {}
    with open(params_filepath, 'r', encoding='utf-8') as f:  # fuerza UTF-8
        for line in f:
            if '=' in line:
                key, val_str = line.split('=')
                key = key.strip()
                
                # Elimina caracteres raros tipo Â
                val_str = val_str.replace('Â', '')
                
                if '±' in val_str:
                    val, err = val_str.split('±')
                    # Limpieza de caracteres no numéricos
                    val = re.sub(r'[^\d\.\-+eE]', '', val).strip()
                    err = re.sub(r'[^\d\.\-+eE]', '', err).strip()
                    params[key] = float(val)
                    params[f"sigma_{key}"] = float(err)
                else:
                    val_str = re.sub(r'[^\d\.\-+eE]', '', val_str).strip()
                    params[key] = float(val_str)
    return params

def calculate_dose_from_od(od_value, params):
    # ... (esta función se mantiene igual) ...
    a, b, c = params['a'], params['b'], params['c']
    if (od_value - c) / a <= 0: return np.nan
    return (1/b) * np.log((od_value - c) / a)

def propagate_error_stable(od_value, sigma_od, params):
    """Propaga el error de forma estable usando el SER."""
    a, b, c = params['a'], params['b'], params['c']
    ser = params.get('SER', 0.001) # Usa el SER del archivo, o un valor por defecto
    
    # Derivada de la dosis con respecto a la OD
    # D = (1/b) * log((OD-c)/a)
    dD_dOD = 1 / (b * (od_value - c))
    
    # La incertidumbre en la predicción de la dosis tiene dos componentes:
    # 1. La incertidumbre del propio ajuste (representada por el SER)
    # 2. La incertidumbre de la nueva medición de OD
    
    # Error en la dosis debido al error del ajuste
    # Esto es una simplificación, pero es robusta. El error en la Dosis será proporcional al error en la OD (SER).
    error_ajuste_en_dosis = np.abs(dD_dOD * ser)

    # Error en la dosis debido al error de la nueva medición
    error_medicion_en_dosis = np.abs(dD_dOD * sigma_od)
    
    # Sumamos los errores en cuadratura
    total_error = np.sqrt(error_ajuste_en_dosis**2 + error_medicion_en_dosis**2)
    
    return total_error

# --- PROGRAMA PRINCIPAL ---
if __name__ == '__main__':
    # Carpeta con archivos a procesar
    folder_with_files = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\LongitudOnda\2025_04_28_RC#15del_2025_03_25\suavizados'
    params_file = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\LongitudOnda\calibration_params.txt'
    UNCERTAINTY_IN_OD_MEASUREMENT = 0.0001

    try:
        # 1. Leer parámetros (incluyendo SER)
        params = parse_params_file(params_file)

        # 2. Iterar sobre todos los .txt de la carpeta
        folder = Path(folder_with_files)
        txt_files = list(folder.glob("*.txt"))

        if not txt_files:
            print("No se encontraron archivos .txt en la carpeta.")
        else:
            for filepath in txt_files:
                wl, od = read_spectrophotometer_file(filepath)
                od_663, sigma_od_local = find_od_at_wavelength(wl, od)
                sigma_od_final = max(sigma_od_local, UNCERTAINTY_IN_OD_MEASUREMENT)

                # 3. Calcular dosis
                dose_calculated = calculate_dose_from_od(od_663, params)

                # 4. Propagar error con el método estable
                error_in_dose = propagate_error_stable(od_663, sigma_od_final, params)

                # --- RESULTADOS ---
                print(f"\nArchivo: {filepath.name}")
                print(f"  OD@663 = {od_663:.4f} ± {sigma_od_final:.4f}")
                print(f"  Dosis  = {dose_calculated:.3f} ± {error_in_dose:.3f} Gy")

    except Exception as e:
        print(f"Ha ocurrido un error: {e}")

#Este codigo: es para cuando NO tenemos la densidad optica y toca sacarla, apartir de la referencia.
#Este codigo: crea una carpeta OD_resultados y guarda los resultados de la densidad optica en ella.
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Ruta base
base_path = r'C:\Users\luis-\Downloads\TFM\DatosEspectrometria\RC_nn'
archivo_referencia = 'Radio_0_spectra.txt'
# Leer I0 (referencia)
i0_file = os.path.join(base_path, 'referencias', archivo_referencia)
i0_data = np.loadtxt(i0_file)
i0 = i0_data[:, 1]  # segunda columna

# Buscar todos los archivos de espectros menos la referencia
spectra_files = [f for f in os.listdir(base_path) if f.endswith('.txt') and archivo_referencia not in f]

# Crear una carpeta para guardar resultados
output_path = os.path.join(base_path, 'OD_resultados')
os.makedirs(output_path, exist_ok=True)

# Graficar
plt.figure(figsize=(10, 6))

for filename in spectra_files:
    filepath = os.path.join(base_path, filename)
    data = np.loadtxt(filepath)
    
    wavelength = data[:, 0]  # Asumimos que la primera columna es longitud de onda
    intensity = data[:, 1]   # Segunda columna es I(l)
    
    # Asegurarse de que tengan misma longitud
    if len(i0) != len(intensity):
        print(f"Longitudes diferentes en {filename}, omitiendo...")
        continue
    
    # Calcular OD
    od = -np.log10(intensity / i0)
    
    # Guardar archivo OD
    od_output = np.column_stack((wavelength, od))
    output_file = os.path.join(output_path, f'OD_{filename}')
    np.savetxt(output_file, od_output, fmt='%.6f', header='Wavelength\tOD', comments='')
    
    # Graficar
    plt.plot(wavelength, od, label=filename)

plt.xlabel('Longitud de onda (nm)')
plt.ylabel('Densidad Óptica (OD)')
plt.title('Espectros de Densidad Óptica')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
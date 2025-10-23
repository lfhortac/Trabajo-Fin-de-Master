import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, 
                            QListWidget, QListWidgetItem, QMessageBox, QGroupBox, 
                            QFormLayout, QSpinBox, QTableWidget, QTableWidgetItem, 
                            QHeaderView, QSplitter, QScrollArea, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QIcon, QFont
from PyQt5.QtCore import Qt, QRect, QSize, QThread, pyqtSignal, QObject
from PIL import Image
from scipy.optimize import curve_fit
import time

# Clase para procesar imágenes en un hilo separado
class ImageProcessor(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_path, crop_area):
        super().__init__()
        self.image_path = image_path
        self.crop_area = crop_area
    
    def process(self):
        try:
            # Abrir la imagen con PIL
            img = np.array(Image.open(self.image_path))
            
            # Extraer el área recortada
            x, y, width, height = self.crop_area
            cut = img[y:y+height, x:x+width]
            
            # Separar canales
            cut_R, cut_G, cut_B = cut[...,0], cut[...,1], cut[...,2]
            
            # Calcular medias y desviaciones estándar
            red_mean, red_std = cut_R.mean(), cut_R.std()
            green_mean, green_std = cut_G.mean(), cut_G.std()
            blue_mean, blue_std = cut_B.mean(), cut_B.std()
            
            # Emitir resultados
            self.finished.emit({
                'red_mean': red_mean,
                'red_std': red_std,
                'green_mean': green_mean,
                'green_std': green_std,
                'blue_mean': blue_mean,
                'blue_std': blue_std
            })
        except Exception as e:
            self.error.emit(str(e))

# Clase para ajustar curvas en un hilo separado
class CurveFitter(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, doses, red_values, green_values, blue_values, red_std, green_std, blue_std):
        super().__init__()
        self.doses = doses
        self.red_values = red_values
        self.green_values = green_values
        self.blue_values = blue_values
        self.red_std = red_std
        self.green_std = green_std
        self.blue_std = blue_std
    
    def fit(self):
        try:
            # Función modelo: Dosis = a + b / (pixel - c)
            def model(x, a, b, c):
                return a + b / (x - c)
            
            # Estimaciones iniciales
            def make_initial(pixels):
                return [np.median(self.doses), 
                       (max(self.doses)-min(self.doses))/(max(pixels)-min(pixels)), 
                       min(pixels)-1]
            
            p0_R = make_initial(self.red_values)
            p0_G = make_initial(self.green_values)
            p0_B = make_initial(self.blue_values)
            
            # Ajuste robusto
            def do_fit(x, y, p0):
                mfes = [20000, 50000, 100000]
                for mf in mfes:
                    try:
                        popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=mf)
                        return popt, pcov
                    except RuntimeError:
                        print(f"curve_fit falló con maxfev={mf}; reintentando...")
                raise RuntimeError("El ajuste de calibración falló después de aumentar maxfev")
            
            # Ajustar canales
            red_params, pcov_R = do_fit(self.red_values, self.doses, p0_R)
            green_params, pcov_G = do_fit(self.green_values, self.doses, p0_G)
            blue_params, pcov_B = do_fit(self.blue_values, self.doses, p0_B)
            
            # Calcular intervalos de confianza del 95%
            from scipy.stats import t
            tval = t.ppf(0.975, len(self.doses)-3)
            red_ci = tval * np.sqrt(np.diag(pcov_R))
            green_ci = tval * np.sqrt(np.diag(pcov_G))
            blue_ci = tval * np.sqrt(np.diag(pcov_B))
            
            # Emitir resultados
            self.finished.emit({
                'red_params': red_params,
                'green_params': green_params,
                'blue_params': blue_params,
                'red_ci': red_ci,
                'green_ci': green_ci,
                'blue_ci': blue_ci
            })
        except Exception as e:
            self.error.emit(str(e))
    
    def closeEvent(self, event):
        # Al cerrar la ventana, para bien el hilo:
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()   # espera a que termine
        super().closeEvent(event)        

class CalibrationModel:
    def __init__(self):
        self.images = []
        self.doses = []
        self.red_values = []
        self.green_values = []
        self.blue_values = []
        self.red_std = []
        self.green_std = []
        self.blue_std = []
        self.red_params = None
        self.green_params = None
        self.blue_params = None
        self.red_ci = None
        self.green_ci = None
        self.blue_ci = None
    
    def add_image(self, image_path, dose, crop_area, rgb_data=None):
        """Añade una imagen al modelo de calibración"""
        try:
            if rgb_data is None:
                # Abrir la imagen con PIL
                img = np.array(Image.open(image_path))
                
                # Extraer el área recortada
                x, y, width, height = crop_area
                cut = img[y:y+height, x:x+width]
                
                # Separar canales
                cut_R, cut_G, cut_B = cut[...,0], cut[...,1], cut[...,2]
                
                # Calcular medias y desviaciones estándar
                red_mean, red_std = cut_R.mean(), cut_R.std()
                green_mean, green_std = cut_G.mean(), cut_G.std()
                blue_mean, blue_std = cut_B.mean(), cut_B.std()
            else:
                # Usar datos RGB proporcionados
                red_mean, red_std = rgb_data['red_mean'], rgb_data['red_std']
                green_mean, green_std = rgb_data['green_mean'], rgb_data['green_std']
                blue_mean, blue_std = rgb_data['blue_mean'], rgb_data['blue_std']
            
            # Guardar datos
            self.images.append({
                'path': image_path,
                'dose': dose,
                'crop_area': crop_area,
                'red_mean': red_mean,
                'red_std': red_std,
                'green_mean': green_mean,
                'green_std': green_std,
                'blue_mean': blue_mean,
                'blue_std': blue_std
            })
            
            # Actualizar listas para ajuste
            self.doses.append(dose)
            self.red_values.append(red_mean)
            self.green_values.append(green_mean)
            self.blue_values.append(blue_mean)
            self.red_std.append(red_std)
            self.green_std.append(green_std)
            self.blue_std.append(blue_std)
            
            return True
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
            return False
    
    def remove_image(self, index):
        """Elimina una imagen del modelo"""
        if 0 <= index < len(self.images):
            self.images.pop(index)
            self.doses.pop(index)
            self.red_values.pop(index)
            self.green_values.pop(index)
            self.blue_values.pop(index)
            self.red_std.pop(index)
            self.green_std.pop(index)
            self.blue_std.pop(index)
            return True
        return False
    
    def update_dose(self, index, dose):
        """Actualiza la dosis de una imagen"""
        if 0 <= index < len(self.images):
            self.images[index]['dose'] = dose
            self.doses[index] = dose
            return True
        return False
    
    def get_image_index(self, image_path):
        """Obtiene el índice de una imagen por su ruta"""
        for i, img in enumerate(self.images):
            if img['path'] == image_path:
                return i
        return -1
    
    def save_parameters(self, filename="CalibParameters.txt"):
        """Guarda los parámetros de calibración en un archivo"""
        if self.red_params is None or self.green_params is None or self.blue_params is None:
            return False
        
        try:
            params = np.vstack((self.red_params, self.green_params, self.blue_params)).T
            np.savetxt(filename, params, fmt='% .7e', delimiter='  ')
            return True
        except Exception as e:
            print(f"Error al guardar parámetros: {e}")
            return False
    
    def save_std_dev(self, filename="DoseStd.txt"):
        """Guarda las desviaciones estándar en un archivo"""
        if not self.red_std or not self.green_std or not self.blue_std:
            return False
        
        try:
            stds = np.vstack((self.red_std, self.green_std, self.blue_std)).T
            np.savetxt(filename, stds, fmt='% .7e', delimiter='  ')
            return True
        except Exception as e:
            print(f"Error al guardar desviaciones estándar: {e}")
            return False

class ImageCanvas(FigureCanvas):
    """Canvas para mostrar y seleccionar áreas en imágenes"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # Configurar estilo oscuro
        self.fig.patch.set_facecolor('#2D2D30')
        self.axes.set_facecolor('#1E1E1E')
        self.axes.tick_params(colors='white')
        
        super(ImageCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.image = None
        self.image_array = None
        self.crop_rect = None
        self.crop_start = None
        self.crop_size = (100, 100)
        self.is_dragging = False
        
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
    
    def load_image(self, image_path):
        """Carga una imagen en el canvas"""
        try:
            self.image_array = np.array(Image.open(image_path))
            
            # Limpiar ejes antes de mostrar nueva imagen
            self.axes.clear()
            
            self.image = self.axes.imshow(self.image_array)
            self.crop_rect = None
            
            # Ajustar límites de los ejes
            self.axes.set_xlim(0, self.image_array.shape[1])
            self.axes.set_ylim(self.image_array.shape[0], 0)
            
            self.fig.tight_layout()
            self.draw_idle()  # Usar draw_idle en lugar de canvas.draw para mejor rendimiento
            return True
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return False
    
    def set_crop_size(self, width, height):
        """Establece el tamaño del área de recorte"""
        self.crop_size = (width, height)
        if self.crop_rect:
            self.update_crop_rect(self.crop_rect.get_x(), self.crop_rect.get_y())
    
    def on_press(self, event):
        """Maneja el evento de presionar el botón del ratón"""
        if event.inaxes != self.axes or self.image is None:
            return
        
        self.is_dragging = True
        self.crop_start = (event.xdata, event.ydata)
        
        if self.crop_rect:
            self.crop_rect.remove()
        
        self.crop_rect = self.axes.add_patch(
            plt.Rectangle((event.xdata, event.ydata), 
                         self.crop_size[0], self.crop_size[1],
                         linewidth=2, edgecolor='r', facecolor='none')
        )
        self.draw_idle()
    
    def on_motion(self, event):
        """Maneja el evento de mover el ratón"""
        if not self.is_dragging or event.inaxes != self.axes or self.crop_rect is None:
            return
        
        self.update_crop_rect(event.xdata, event.ydata)
    
    def on_release(self, event):
        """Maneja el evento de soltar el botón del ratón"""
        self.is_dragging = False
    
    def update_crop_rect(self, x, y):
        """Actualiza la posición del rectángulo de recorte"""
        if self.image is None or self.crop_rect is None:
            return
        
        # Asegurar que el rectángulo esté dentro de los límites de la imagen
        height, width = self.image_array.shape[:2]
        
        x = max(0, min(width - self.crop_size[0], x))
        y = max(0, min(height - self.crop_size[1], y))
        
        self.crop_rect.set_xy((x, y))
        self.draw_idle()
    
    def get_crop_area(self):
        """Devuelve el área de recorte actual"""
        if self.crop_rect is None:
            return None
        
        x, y = self.crop_rect.get_xy()
        return (int(x), int(y), int(self.crop_size[0]), int(self.crop_size[1]))
    
    def set_crop_area(self, crop_area):
        """Establece el área de recorte desde coordenadas existentes"""
        if self.image is None:
            return
        
        x, y, width, height = crop_area
        
        if self.crop_rect:
            self.crop_rect.remove()
        
        self.crop_rect = self.axes.add_patch(
            plt.Rectangle((x, y), width, height,
                         linewidth=2, edgecolor='r', facecolor='none')
        )
        self.draw_idle()

# Clase para crear un canvas individual para cada canal de color
class SingleChannelCanvas(FigureCanvas):
    """Canvas para mostrar la curva de calibración de un solo canal"""
    def __init__(self, parent=None, width=6, height=5, dpi=100, channel='red', title='Canal Rojo'):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Configurar estilo
        self.fig.patch.set_facecolor('white')
        self.axes.set_facecolor('white')
        self.axes.tick_params(colors='black')
        self.axes.xaxis.label.set_color('black')
        self.axes.yaxis.label.set_color('black')
        self.axes.title.set_color('black')
        
        # Configurar ejes
        self.axes.set_xlabel('Valor de Píxel')
        self.axes.set_ylabel('Dosis (Gy)')
        self.axes.set_title(title)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        self.channel = channel
        self.channel_color = channel
    
    def plot_calibration(self, model):
        """Dibuja la curva de calibración para un canal específico"""
        if self.channel == 'red' and model.red_params is None:
            return
        elif self.channel == 'green' and model.green_params is None:
            return
        elif self.channel == 'blue' and model.blue_params is None:
            return
        
        self.axes.clear()
        
        # Configurar ejes
        self.axes.set_xlabel('Valor de Píxel')
        self.axes.set_ylabel('Dosis (Gy)')
        
        if self.channel == 'red':
            self.axes.set_title('Curva de Calibración - Canal Rojo')
            values = model.red_values
            params = model.red_params
        elif self.channel == 'green':
            self.axes.set_title('Curva de Calibración - Canal Verde')
            values = model.green_values
            params = model.green_params
        else:  # blue
            self.axes.set_title('Curva de Calibración - Canal Azul')
            values = model.blue_values
            params = model.blue_params
        
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Función modelo
        def model_func(x, a, b, c):
            return a + b / (x - c)
        
        # Dibujar puntos de datos
        self.axes.scatter(values, model.doses, color=self.channel_color, label='Datos', alpha=0.7)
        
        # Generar puntos para la curva
        x_min, x_max = min(values), max(values)
        x_range = np.linspace(x_min, x_max, 100)
        
        # Dibujar curva ajustada
        try:
            y_values = [model_func(x, *params) for x in x_range 
                      if abs(x - params[2]) > 1]
            x_values = [x for x in x_range if abs(x - params[2]) > 1]
            self.axes.plot(x_values, y_values, color=self.channel_color, linestyle='-', label='Ajuste')
        except Exception as e:
            print(f"Error al dibujar curva: {e}")
        
        self.axes.legend()
        self.fig.tight_layout()
        self.draw_idle()

class CalibrationCurvesCanvas(FigureCanvas):
    """Canvas para mostrar las curvas de calibración en una sola gráfica."""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        # Fondo blanco
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        self.ax.tick_params(colors='black')
        self.ax.xaxis.label.set_color('black')
        self.ax.yaxis.label.set_color('black')
        self.ax.title.set_color('black')

        # Etiquetas
        self.ax.set_xlabel('Valor de Píxel')
        self.ax.set_ylabel('Dosis (Gy)')
        self.ax.set_title('Curvas calibración')
        self.ax.grid(True, linestyle='--', alpha=0.5)

    def plot_calibration(self, model):
        if (model.red_params is None or model.green_params is None or model.blue_params is None):
            return

        # Función de ajuste
        def f(x, a, b, c): return a + b/(x-c)

        data = [
            (model.red_values, model.red_params, 'red', 'Canal Rojo'),
            (model.green_values, model.green_params, 'green', 'Canal Verde'),
            (model.blue_values, model.blue_params, 'blue', 'Canal Azul'),
        ]

        self.ax.clear()

        for vals, params, color, label in data:
            # puntos
            self.ax.scatter(vals, model.doses, color=color, alpha=0.7, label=f'Datos {label}')
            # curva ajustada
            xs = np.linspace(min(vals), max(vals), 200)
            xs = xs[np.abs(xs - params[2]) > 1]  # evitar singularidad
            self.ax.plot(xs, f(xs, *params), color=color, linestyle='-', label=f'Ajuste {label}')

        self.ax.set_xlabel('Valor de Píxel')
        self.ax.set_ylabel('Dosis (Gy)')
        self.ax.set_title('Curvas calibración')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend()

        self.draw_idle()



class ResidualsCanvas(FigureCanvas):
    """Canvas para mostrar residuos, cada canal en un subplot separado."""
    def __init__(self, parent=None, width=12, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(1, 3, sharey=True)
        super().__init__(self.fig)
        self.setParent(parent)

        # Fondo blanco
        self.fig.patch.set_facecolor('white')
        for ax in self.axes:
            ax.set_facecolor('white')
            ax.tick_params(colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.title.set_color('black')

        # Etiquetas
        colors = ['Rojo', 'Verde', 'Azul']
        for ax, color in zip(self.axes, colors):
            ax.set_xlabel('Valor de Píxel')
            ax.set_title(f'Residuales {color}')
        self.axes[0].set_ylabel('Residuos (Gy)')
        self.fig.tight_layout()

    def plot_residuals(self, model):
        if (model.red_params is None or model.green_params is None or model.blue_params is None):
            return

        def f(x, a, b, c): return a + b/(x-c)

        data = [
            (model.red_values, model.red_params, 'red'),
            (model.green_values, model.green_params, 'green'),
            (model.blue_values, model.blue_params, 'blue'),
        ]

        # Calcular y dibujar residuos
        for ax, (vals, params, color) in zip(self.axes, data):
            ax.clear()
            res = [model.doses[i] - f(x, *params) for i, x in enumerate(vals)]
            ax.scatter(vals, res, color=color, alpha=0.7)
            ax.axhline(0, linestyle='--', color='gray', linewidth=1)
            ax.grid(True, linestyle='--', alpha=0.5)

        self.draw_idle()

class InstructionsDialog(QWidget):
    """Diálogo para mostrar instrucciones de uso"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Instrucciones de Uso")
        self.setGeometry(200, 200, 800, 600)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Establecer fondo blanco para la ventana
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, Qt.white)
        palette.setColor(QPalette.WindowText, Qt.black)
        self.setPalette(palette)
        
        # Título
        title_label = QLabel("Instrucciones de Uso")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 15px; color: black;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # Área de desplazamiento para el texto
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Configurar el fondo blanco para el área de desplazamiento
        scroll_area.setAutoFillBackground(True)
        palette = scroll_area.palette()
        palette.setColor(QPalette.Window, Qt.white)
        scroll_area.setPalette(palette)
        
        # Widget de contenido
        content_widget = QWidget()
        content_widget.setAutoFillBackground(True)
        palette = content_widget.palette()
        palette.setColor(QPalette.Window, Qt.white)
        palette.setColor(QPalette.WindowText, Qt.black)
        content_widget.setPalette(palette)
        
        content_layout = QVBoxLayout(content_widget)
        
        # Texto de instrucciones
        instructions = [
            ("1. Inicio", "La aplicación tiene tres pestañas principales: Inicio, Cargar Imágenes y Visualizar Calibración."),
            ("2. Cargar Imágenes", "Haga clic en 'Cargar Imágenes' para seleccionar las imágenes TIFF de calibración."),
            ("3. Seleccionar Región", "Para cada imagen:"),
            ("", "   - Ingrese el ancho y alto de la selección en píxeles."),
            ("", "   - Haga clic en la imagen para posicionar la selección."),
            ("", "   - La selección se mostrará como un rectángulo rojo."),
            ("4. Ingresar Dosis", "Para cada imagen:"),
            ("", "   - Ingrese la dosis correspondiente en el campo 'Dosis (Gy)', IMPORTANTE USAR NOTACIÓN DECIMAL CON PUNTO PARA TODOS LOS VALORES."),
            ("", "   - Haga clic en 'Ingresar Dosis' para guardar el valor."),
            ("5. Realizar Calibración", "Una vez que haya procesado todas las imágenes:"),
            ("", "   - Haga clic en 'Realizar Calibración' para iniciar el proceso de calibración."),
            ("", "   - La aplicación ajustará las curvas de calibración para los canales rojo, verde y azul."),
            ("6. Visualizar Resultados", "En la pestaña 'Visualizar Calibración':"),
            ("", "   - Vea las curvas de calibración con colores correspondientes a cada canal."),
            ("", "   - Examine el análisis de residuos."),
            ("", "   - Consulte la tabla de parámetros con intervalos de confianza."),
            ("", "   - Revise la tabla de datos brutos."),
            ("7. Guardar Resultados", "Puede guardar:"),
            ("", "   - Los parámetros de calibración (botón 'Guardar Parámetros')."),
            ("", "   - Las desviaciones estándar (botón 'Guardar Desviaciones')."),
            ("", "   - Las imágenes de las curvas (botón 'Guardar Imágenes')."),
            ("8. Modelo de Calibración", "La aplicación utiliza el modelo:"),
            ("", "   - Dosis = a + b / (pixel - c)"),
            ("", "   - Donde a, b y c son los parámetros ajustados para cada canal."),
            ("9. Consejos", "Para obtener mejores resultados:"),
            ("", "   - Use al menos 5 imágenes con diferentes dosis."),
            ("", "   - Seleccione áreas homogéneas en las imágenes."),
            ("", "   - Asegúrese de que las dosis cubran el rango de interés."),
        ]
        
        for title, text in instructions:
            if title:
                label = QLabel(title)
                label.setStyleSheet("font-weight: bold; font-size: 18px; margin-top: 15px; color: black;")
                content_layout.addWidget(label)
            
            text_label = QLabel(text)
            text_label.setStyleSheet("font-size: 16px; color: black;")
            text_label.setWordWrap(True)
            content_layout.addWidget(text_label)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        
        # Botón de cerrar
        close_button = QPushButton("Cerrar")
        close_button.setStyleSheet("font-size: 16px;")
        close_button.clicked.connect(self.close)
        
        layout.addWidget(title_label)
        layout.addWidget(scroll_area)
        layout.addWidget(close_button)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.model = CalibrationModel()
        self.current_image_path = None
        self.image_processor_thread = None
        self.curve_fitter_thread = None
        
        # Crear canvases individuales para cada canal
        self.red_canvas = SingleChannelCanvas(channel='red', title='Canal Rojo')
        self.green_canvas = SingleChannelCanvas(channel='green', title='Canal Verde')
        self.blue_canvas = SingleChannelCanvas(channel='blue', title='Canal Azul')
        
        self.init_ui()
        self.set_dark_theme()
        self.set_button_font_size()
    
    def set_dark_theme(self):
        """Configura el tema oscuro para la aplicación"""
        dark_palette = QPalette()
        
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        QApplication.setPalette(dark_palette)
        QApplication.setStyle("Fusion")
    
    def set_button_font_size(self):
        """Aumenta el tamaño de fuente de todos los botones"""
        # Recorrer todos los widgets de la aplicación
        for widget in self.findChildren(QPushButton):
            font = widget.font()
            font.setPointSize(12)  # Aumentar tamaño de fuente
            widget.setFont(font)
    
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        self.setWindowTitle("Calibración de Películas Radiocrómicas")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central y layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Crear pestañas
        self.tabs = QTabWidget()
        self.home_tab = QWidget()
        self.load_tab = QWidget()
        self.visualize_tab = QWidget()
        
        self.tabs.addTab(self.home_tab, "Inicio")
        self.tabs.addTab(self.load_tab, "Cargar Imágenes")
        self.tabs.addTab(self.visualize_tab, "Visualizar Calibración")
        
        main_layout.addWidget(self.tabs)
        
        # Configurar pestañas
        self.setup_home_tab()
        self.setup_load_tab()
        self.setup_visualize_tab()
    
    def setup_home_tab(self):
        """Configura la pestaña de inicio"""
        layout = QVBoxLayout(self.home_tab)
        
        # Título
        title_label = QLabel("Calibración de Películas Radiocrómicas")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        
        subtitle_label = QLabel("Una interfaz para la calibración y análisis de películas radiocrómicas")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 16px; color: #AAAAAA; margin-bottom: 40px;")
        
        # Botones principales en lista vertical
        buttons_layout = QVBoxLayout()
        
        # Botón Cargar Imágenes
        load_button = QPushButton("Cargar Imágenes")
        load_button.setMinimumHeight(50)
        load_button.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        
        # Botón Visualizar Calibración
        visualize_button = QPushButton("Visualizar Calibración")
        visualize_button.setMinimumHeight(50)
        visualize_button.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        
        # Botón Instrucciones
        instructions_button = QPushButton("Instrucciones")
        instructions_button.setMinimumHeight(50)
        instructions_button.clicked.connect(self.show_instructions)
        
        # Botón Salir
        exit_button = QPushButton("Salir")
        exit_button.setMinimumHeight(50)
        exit_button.clicked.connect(self.close)
        
        buttons_layout.addWidget(load_button)
        buttons_layout.addWidget(visualize_button)
        buttons_layout.addWidget(instructions_button)
        buttons_layout.addWidget(exit_button)
        
        # Contenedor para centrar los botones
        buttons_container = QWidget()
        buttons_container.setLayout(buttons_layout)
        buttons_container.setMaximumWidth(400)
        
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(buttons_container, alignment=Qt.AlignCenter)
        layout.addStretch()
    
    def setup_load_tab(self):
        """Configura la pestaña de carga de imágenes"""
        layout = QHBoxLayout(self.load_tab)
        
        # Panel izquierdo (controles)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Grupo de carga de imágenes
        load_group = QGroupBox("Controles")
        load_layout = QVBoxLayout(load_group)
        
        load_button = QPushButton("Cargar Imágenes")
        load_button.clicked.connect(self.load_images)
        load_layout.addWidget(load_button)
        
        # Grupo de tamaño de selección
        size_group = QGroupBox("Tamaño de Selección")
        size_layout = QFormLayout(size_group)
        
        self.width_input = QSpinBox()
        self.width_input.setRange(10, 1000)
        self.width_input.setValue(100)
        self.width_input.valueChanged.connect(self.update_crop_size)
        
        self.height_input = QSpinBox()
        self.height_input.setRange(10, 1000)
        self.height_input.setValue(100)
        self.height_input.valueChanged.connect(self.update_crop_size)
        
        size_layout.addRow("Ancho (px):", self.width_input)
        size_layout.addRow("Alto (px):", self.height_input)
        
        # Grupo de dosis
        dose_group = QGroupBox("Dosis")
        dose_layout = QVBoxLayout(dose_group)
        
        dose_form_layout = QFormLayout()
        self.dose_input = QLineEdit()
        self.dose_input.setPlaceholderText("Ingrese la dosis en Gy")
        dose_form_layout.addRow("Dosis (Gy):", self.dose_input)
        
        enter_dose_button = QPushButton("Ingresar Dosis")
        enter_dose_button.clicked.connect(self.enter_dose)
        
        dose_layout.addLayout(dose_form_layout)
        dose_layout.addWidget(enter_dose_button)
        
        # Añadir grupos al panel izquierdo
        left_layout.addWidget(load_group)
        left_layout.addWidget(size_group)
        left_layout.addWidget(dose_group)
        left_layout.addStretch()
        
        # Botón de calibración en la esquina inferior izquierda
        calibrate_button = QPushButton("Realizar Calibración")
        calibrate_button.clicked.connect(self.perform_calibration)
        left_layout.addWidget(calibrate_button)
        
        # Panel central (visualización de imagen)
        central_panel = QWidget()
        central_layout = QVBoxLayout(central_panel)
        
        self.image_canvas = ImageCanvas(central_panel, width=8, height=6)
        central_layout.addWidget(self.image_canvas)
        
        # Panel inferior (lista de imágenes)
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        
        list_label = QLabel("Lista de Imágenes")
        list_label.setStyleSheet("font-weight: bold;")
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.select_image_from_list)
        
        bottom_layout.addWidget(list_label)
        bottom_layout.addWidget(self.image_list)
        
        # Crear un splitter para dividir los paneles
        splitter_v = QSplitter(Qt.Vertical)
        splitter_v.addWidget(central_panel)
        splitter_v.addWidget(bottom_panel)
        splitter_v.setSizes([600, 200])
        
        splitter_h = QSplitter(Qt.Horizontal)
        splitter_h.addWidget(left_panel)
        splitter_h.addWidget(splitter_v)
        splitter_h.setSizes([300, 900])
        
        layout.addWidget(splitter_h)
    
    def setup_visualize_tab(self):
        """Configura la pestaña de visualización de calibración"""
        layout = QVBoxLayout(self.visualize_tab)
        
        # Botones superiores
        top_layout = QHBoxLayout()
        
        save_params_button = QPushButton("Guardar Parámetros")
        save_params_button.clicked.connect(self.save_parameters)
        
        save_std_button = QPushButton("Guardar Desviaciones")
        save_std_button.clicked.connect(self.save_std_dev)
        
        save_images_button = QPushButton("Guardar Imágenes")
        save_images_button.clicked.connect(self.save_calibration_images)
        
        top_layout.addWidget(save_params_button)
        top_layout.addWidget(save_std_button)
        top_layout.addWidget(save_images_button)
        top_layout.addStretch()
        
        # Pestañas de visualización
        vis_tabs = QTabWidget()
        
        # Pestaña de curvas
        curves_tab = QWidget()
        curves_layout = QVBoxLayout(curves_tab)
        
        self.calibration_canvas = CalibrationCurvesCanvas(curves_tab, width=8, height=6)
        self.residuals_canvas = ResidualsCanvas(curves_tab, width=8, height=4)
        
        curves_layout.addWidget(self.calibration_canvas)
        curves_layout.addWidget(self.residuals_canvas)
        
        # Pestaña de parámetros
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        
        params_label = QLabel("Parámetros de Calibración")
        params_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        params_label.setAlignment(Qt.AlignCenter)
        
        model_desc = QLabel("Modelo: Dosis = a + b / (pixel - c)")
        model_desc.setAlignment(Qt.AlignCenter)
        
        self.params_table = QTableWidget(3, 5)
        self.params_table.setHorizontalHeaderLabels(["Canal", "a", "b", "c", "95% CI"])
        self.params_table.verticalHeader().setVisible(False)
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Configurar filas de la tabla
        red_item = QTableWidgetItem("Rojo")
        red_item.setForeground(QColor(255, 0, 0))
        self.params_table.setItem(0, 0, red_item)
        
        green_item = QTableWidgetItem("Verde")
        green_item.setForeground(QColor(0, 255, 0))
        self.params_table.setItem(1, 0, green_item)
        
        blue_item = QTableWidgetItem("Azul")
        blue_item.setForeground(QColor(0, 0, 255))
        self.params_table.setItem(2, 0, blue_item)
        
        params_layout.addWidget(params_label)
        params_layout.addWidget(model_desc)
        params_layout.addWidget(self.params_table)
        
        # Pestaña de datos
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        data_label = QLabel("Datos de Medición")
        data_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        data_label.setAlignment(Qt.AlignCenter)
        
        self.data_table = QTableWidget(0, 7)
        self.data_table.setHorizontalHeaderLabels([
            "Dosis (Gy)", "Media Rojo", "Desv. Rojo", 
            "Media Verde", "Desv. Verde", 
            "Media Azul", "Desv. Azul"
        ])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.data_table)
        
        # Añadir pestañas
        vis_tabs.addTab(curves_tab, "Curvas de Calibración")
        vis_tabs.addTab(params_tab, "Parámetros")
        vis_tabs.addTab(data_tab, "Datos")
        
        layout.addLayout(top_layout)
        layout.addWidget(vis_tabs)
    
    def show_instructions(self):
        """Muestra el diálogo de instrucciones"""
        instructions_dialog = InstructionsDialog(self)
        instructions_dialog.show()
    
    def load_images(self):
        """Carga imágenes desde el sistema de archivos"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Imágenes (*.tiff *.tif *.png *.jpg *.jpeg)")
        
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            
            for filename in filenames:
                # Añadir a la lista de imágenes
                item = QListWidgetItem(os.path.basename(filename))
                item.setData(Qt.UserRole, filename)
                self.image_list.addItem(item)
            
            # Seleccionar la primera imagen
            if self.image_list.count() > 0 and not self.current_image_path:
                self.image_list.setCurrentRow(0)
                self.select_image_from_list(self.image_list.item(0))
    
    def select_image_from_list(self, item):
        """Selecciona una imagen de la lista"""
        if item is None:
            return
        
        image_path = item.data(Qt.UserRole)
        self.current_image_path = image_path
        
        # Cargar imagen en el canvas
        self.image_canvas.load_image(image_path)
        
        # Buscar si ya existe dosis y área de recorte para esta imagen
        for img in self.model.images:
            if img['path'] == image_path:
                self.dose_input.setText(str(img['dose']))
                if 'crop_area' in img:
                    self.image_canvas.set_crop_area(img['crop_area'])
                return
        
        # Si no existe, limpiar el campo de dosis
        self.dose_input.clear()
    
    def update_crop_size(self):
        """Actualiza el tamaño del área de recorte"""
        width = self.width_input.value()
        height = self.height_input.value()
        self.image_canvas.set_crop_size(width, height)
    
    def enter_dose(self):
        """Guarda la dosis para la imagen actual"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Advertencia", "No hay imagen seleccionada.")
            return
        
        # Verificar que haya un área de recorte
        crop_area = self.image_canvas.get_crop_area()
        if crop_area is None:
            QMessageBox.warning(self, "Advertencia", "Seleccione un área en la imagen primero.")
            return
        
        # Verificar dosis
        try:
            dose = float(self.dose_input.text())
            if dose <= 0:
                raise ValueError("La dosis debe ser mayor que cero")
        except ValueError:
            QMessageBox.warning(self, "Advertencia", "Ingrese una dosis válida mayor que cero.")
            return
        
        # Procesar imagen en un hilo separado
        self.process_image_async(self.current_image_path, dose, crop_area)
    
    def process_image_async(self, image_path, dose, crop_area):
        """Procesa una imagen en un hilo separado"""
        # Crear procesador de imágenes
        self.image_processor = ImageProcessor(image_path, crop_area)
        
        # Crear hilo
        self.image_processor_thread = QThread()
        self.image_processor.moveToThread(self.image_processor_thread)
        
        # Conectar señales
        self.image_processor_thread.started.connect(self.image_processor.process)
        self.image_processor.finished.connect(self.on_image_processed)
        self.image_processor.error.connect(self.on_image_process_error)
        self.image_processor.finished.connect(self.image_processor_thread.quit)
        self.image_processor.finished.connect(self.image_processor.deleteLater)
        self.image_processor_thread.finished.connect(self.image_processor_thread.deleteLater)
        
        # Almacenar datos para usar en el callback
        self.image_processor.image_path = image_path
        self.image_processor.dose = dose
        self.image_processor.crop_area = crop_area
        
        # Iniciar hilo
        self.image_processor_thread.start()
        
        # Mostrar mensaje de procesamiento
        QMessageBox.information(self, "Procesando", "Procesando imagen, por favor espere...")
    
    def on_image_processed(self, rgb_data):
        """Callback cuando se completa el procesamiento de la imagen"""
        # Obtener datos del procesador
        image_path = self.image_processor.image_path
        dose = self.image_processor.dose
        crop_area = self.image_processor.crop_area
        
        # Añadir o actualizar imagen en el modelo
        index = self.model.get_image_index(image_path)
        if index >= 0:
            # Actualizar imagen existente
            self.model.remove_image(index)
        
        # Añadir imagen con datos RGB
        self.model.add_image(image_path, dose, crop_area, rgb_data)
        
        # Actualizar lista de imágenes
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole) == image_path:
                item.setText(f"{os.path.basename(image_path)} - {dose} Gy")
                break
        
        QMessageBox.information(self, "Éxito", f"Dosis de {dose} Gy guardada para la imagen.")
    
    def on_image_process_error(self, error_msg):
        """Callback cuando hay un error en el procesamiento de la imagen"""
        QMessageBox.critical(self, "Error", f"Error al procesar la imagen: {error_msg}")
    
    def perform_calibration(self):
        """Realiza la calibración con las imágenes procesadas"""
        if len(self.model.images) < 3:
            QMessageBox.warning(
                self, 
                "Advertencia", 
                "Se necesitan al menos 3 imágenes con diferentes dosis para realizar la calibración."
            )
            return
        
        # Verificar que todas las imágenes tengan dosis
        for img in self.model.images:
            if img['dose'] <= 0:
                QMessageBox.warning(
                    self, 
                    "Advertencia", 
                    f"La imagen {os.path.basename(img['path'])} tiene una dosis inválida."
                )
                return
        
        # Ajustar curvas de calibración en un hilo separado
        self.fit_curves_async()
    
    def fit_curves_async(self):
        """Ajusta las curvas de calibración en un hilo separado"""
        # Crear ajustador de curvas
        self.curve_fitter = CurveFitter(
            self.model.doses,
            self.model.red_values,
            self.model.green_values,
            self.model.blue_values,
            self.model.red_std,
            self.model.green_std,
            self.model.blue_std
        )
        
        # Crear hilo
        self.curve_fitter_thread = QThread()
        self.curve_fitter.moveToThread(self.curve_fitter_thread)
        
        # Conectar señales
        self.curve_fitter_thread.started.connect(self.curve_fitter.fit)
        self.curve_fitter.finished.connect(self.on_curves_fitted)
        self.curve_fitter.error.connect(self.on_curve_fit_error)
        self.curve_fitter.finished.connect(self.curve_fitter_thread.quit)
        self.curve_fitter.finished.connect(self.curve_fitter.deleteLater)
        self.curve_fitter_thread.finished.connect(self.curve_fitter_thread.deleteLater)
        
        # Iniciar hilo
        self.curve_fitter_thread.start()
        
        # Mostrar mensaje de procesamiento
        QMessageBox.information(self, "Procesando", "Ajustando curvas de calibración, por favor espere...")
    
    def on_curves_fitted(self, fit_data):
        """Callback cuando se completa el ajuste de curvas"""
        # Actualizar modelo con parámetros ajustados
        self.model.red_params = fit_data['red_params']
        self.model.green_params = fit_data['green_params']
        self.model.blue_params = fit_data['blue_params']
        self.model.red_ci = fit_data['red_ci']
        self.model.green_ci = fit_data['green_ci']
        self.model.blue_ci = fit_data['blue_ci']
        
        # Actualizar visualizaciones
        self.update_calibration_view()
        
        # Actualizar canvases individuales
        self.red_canvas.plot_calibration(self.model)
        self.green_canvas.plot_calibration(self.model)
        self.blue_canvas.plot_calibration(self.model)
        
        # Cambiar a la pestaña de visualización
        self.tabs.setCurrentIndex(2)
        
        QMessageBox.information(
            self, 
            "Éxito", 
            "Calibración completada con éxito."
        )
    
    def on_curve_fit_error(self, error_msg):
        """Callback cuando hay un error en el ajuste de curvas"""
        QMessageBox.critical(self, "Error", f"Error al ajustar las curvas: {error_msg}")
    
    def update_calibration_view(self):
        """Actualiza la vista de calibración"""
        # Actualizar gráficos
        self.calibration_canvas.plot_calibration(self.model)
        self.residuals_canvas.plot_residuals(self.model)
        
        # Actualizar tabla de parámetros
        if self.model.red_params is not None and self.model.green_params is not None and self.model.blue_params is not None:
            # Parámetros rojos
            self.params_table.setItem(0, 1, QTableWidgetItem(f"{self.model.red_params[0]:.4f}"))
            self.params_table.setItem(0, 2, QTableWidgetItem(f"{self.model.red_params[1]:.4f}"))
            self.params_table.setItem(0, 3, QTableWidgetItem(f"{self.model.red_params[2]:.4f}"))
            self.params_table.setItem(0, 4, QTableWidgetItem(
                f"±{self.model.red_ci[0]:.4f}, ±{self.model.red_ci[1]:.4f}, ±{self.model.red_ci[2]:.4f}"
            ))
            
            # Parámetros verdes
            self.params_table.setItem(1, 1, QTableWidgetItem(f"{self.model.green_params[0]:.4f}"))
            self.params_table.setItem(1, 2, QTableWidgetItem(f"{self.model.green_params[1]:.4f}"))
            self.params_table.setItem(1, 3, QTableWidgetItem(f"{self.model.green_params[2]:.4f}"))
            self.params_table.setItem(1, 4, QTableWidgetItem(
                f"±{self.model.green_ci[0]:.4f}, ±{self.model.green_ci[1]:.4f}, ±{self.model.green_ci[2]:.4f}"
            ))
            
            # Parámetros azules
            self.params_table.setItem(2, 1, QTableWidgetItem(f"{self.model.blue_params[0]:.4f}"))
            self.params_table.setItem(2, 2, QTableWidgetItem(f"{self.model.blue_params[1]:.4f}"))
            self.params_table.setItem(2, 3, QTableWidgetItem(f"{self.model.blue_params[2]:.4f}"))
            self.params_table.setItem(2, 4, QTableWidgetItem(
                f"±{self.model.blue_ci[0]:.4f}, ±{self.model.blue_ci[1]:.4f}, ±{self.model.blue_ci[2]:.4f}"
            ))
        
        # Actualizar tabla de datos
        self.data_table.setRowCount(len(self.model.doses))
        
        for i in range(len(self.model.doses)):
            self.data_table.setItem(i, 0, QTableWidgetItem(f"{self.model.doses[i]:.2f}"))
            self.data_table.setItem(i, 1, QTableWidgetItem(f"{self.model.red_values[i]:.2f}"))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{self.model.red_std[i]:.2f}"))
            self.data_table.setItem(i, 3, QTableWidgetItem(f"{self.model.green_values[i]:.2f}"))
            self.data_table.setItem(i, 4, QTableWidgetItem(f"{self.model.green_std[i]:.2f}"))
            self.data_table.setItem(i, 5, QTableWidgetItem(f"{self.model.blue_values[i]:.2f}"))
            self.data_table.setItem(i, 6, QTableWidgetItem(f"{self.model.blue_std[i]:.2f}"))
    
    def save_parameters(self):
        """Guarda los parámetros de calibración"""
        if (
            self.model.red_params is None or self.model.red_params.size == 0 or
            self.model.green_params is None or self.model.green_params.size == 0 or
            self.model.blue_params is None or self.model.blue_params.size == 0
        ):
            QMessageBox.warning(self, "Error", "Faltan parámetros de calibración para alguno de los canales.")
            return
        
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Archivos de texto (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        file_dialog.selectFile("CalibParameters.txt")
        
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            if self.model.save_parameters(filename):
                QMessageBox.information(self, "Éxito", f"Parámetros guardados en {filename}")
            else:
                QMessageBox.critical(self, "Error", "Error al guardar los parámetros.")
    
    def save_std_dev(self):
        """Guarda las desviaciones estándar"""
        if not self.model.red_std or not self.model.green_std or not self.model.blue_std:
            QMessageBox.warning(self, "Advertencia", "No hay datos de desviación estándar para guardar.")
            return
        
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Archivos de texto (*.txt)")
        file_dialog.setDefaultSuffix("txt")
        file_dialog.selectFile("DoseStd.txt")
        
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            if self.model.save_std_dev(filename):
                QMessageBox.information(self, "Éxito", f"Desviaciones guardadas en {filename}")
            else:
                QMessageBox.critical(self, "Error", "Error al guardar las desviaciones.")
    
    def save_calibration_images(self):
        """Guarda las imágenes de calibración"""
        if (
            self.model.red_params is None or self.model.red_params.size == 0 or
            self.model.green_params is None or self.model.green_params.size == 0 or
            self.model.blue_params is None or self.model.blue_params.size == 0
        ):
            QMessageBox.warning(self, "Error", "Faltan parámetros de calibración para alguno de los canales.")
            return
        
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar directorio para guardar imágenes")
        if not directory:
            return
        
        # Guardar imagen de curvas de calibración combinadas
        self.calibration_canvas.fig.savefig(
            os.path.join(directory, "calibration_curves_combined.png"), 
            dpi=300, 
            bbox_inches='tight'
        )
        
        # Guardar imagen de residuos
        self.residuals_canvas.fig.savefig(
            os.path.join(directory, "residuals.png"), 
            dpi=300, 
            bbox_inches='tight'
        )
        
        # Guardar imágenes individuales para cada canal
        # Canal rojo
        self.red_canvas.plot_calibration(self.model)
        self.red_canvas.fig.savefig(
            os.path.join(directory, "calibration_curve_red.png"),
            dpi=300,
            bbox_inches='tight'
        )
        
        # Canal verde
        self.green_canvas.plot_calibration(self.model)
        self.green_canvas.fig.savefig(
            os.path.join(directory, "calibration_curve_green.png"),
            dpi=300,
            bbox_inches='tight'
        )
        
        # Canal azul
        self.blue_canvas.plot_calibration(self.model)
        self.blue_canvas.fig.savefig(
            os.path.join(directory, "calibration_curve_blue.png"),
            dpi=300,
            bbox_inches='tight'
        )
        
        QMessageBox.information(
            self, 
            "Éxito", 
            f"Imágenes guardadas en {directory}:\n"
            f"- calibration_curves_combined.png\n"
            f"- residuals.png\n"
            f"- calibration_curve_red.png\n"
            f"- calibration_curve_green.png\n"
            f"- calibration_curve_blue.png"
        )

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
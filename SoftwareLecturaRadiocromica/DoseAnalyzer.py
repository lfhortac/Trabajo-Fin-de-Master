import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Importar antes de usar 3D
import matplotlib.cm as cm
import customtkinter as ctk
from datetime import datetime


# Parámetros de calibración
pars = np.loadtxt('CalibParameters.txt').reshape(3, 3)
redCali, greenCali, blueCali = pars[:, 0], pars[:, 1], pars[:, 2]

# Compatibilidad Pillow
try:
    RESAMPLING = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLING = Image.LANCZOS

class DoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dose Analysis Tool")
        
        # Maximizar la ventana al inicio
        self.root.state('zoomed')  # Para Windows
        try:
            self.root.attributes('-zoomed', True)  # Para Linux
        except:
            pass
        
        self.detected_circles = []
        self.manual_circles = []  # Lista para almacenar círculos añadidos manualmente
        self.subcircles_data = []  # Para almacenar datos de los subcírculos
        self.selected_subcircle = None  # Para almacenar el subcírculo seleccionado

        # --- Paleta de colores ---
        fondo = "#1E1E2F"
        texto = "#FFFFFF"
        boton_color = "#3E4A61"
        boton_activo = "#556178"
        entrada_fondo = "#2C2F48"

        # --- Frame principal con layout nuevo (controles a la izquierda) ---
        main_frame = tk.Frame(root, bg=fondo)
        main_frame.pack(fill="both", expand=True)

        # Frame de controles (ahora a la izquierda con ancho aumentado)
        self.info_frame = tk.Frame(main_frame, width=300, bg=fondo)  # Aumentado de 250 a 300
        self.info_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.info_frame.pack_propagate(False)  # Evitar que el frame se reduzca

        # --- Etiqueta título ---
        tk.Label(self.info_frame, text="📊 Dose Tool", font=("Segoe UI", 14, "bold"),
                bg=fondo, fg=texto).pack(pady=(0, 20))

         # Función para crear botones estilizados (más pequeños)
        def styled_button(master, text, command):
            return tk.Button(master, text=text, command=command,
                            bg=boton_color, fg=texto, font=("Segoe UI", 9),  # Reducido de 10 a 9
                            relief="flat", bd=0, padx=8, pady=5,  # Reducido padx de 10 a 8, pady de 6 a 5
                            activebackground=boton_activo, activeforeground=texto)

        # Reorganización de botones - Grupo 1: Operaciones principales
        main_buttons_frame = tk.LabelFrame(self.info_frame, text="Operaciones Principales", 
                                          bg=fondo, fg=texto, font=("Segoe UI", 10, "bold"))
        main_buttons_frame.pack(pady=5, fill="x", padx=5)
        
        styled_button(main_buttons_frame, "Cargar Imagen", self.load_image).pack(pady=3, fill="x")
        styled_button(main_buttons_frame, "Detect Radiocromicas", self.detectar_areas_radiocromicas).pack(pady=3, fill="x")
        styled_button(main_buttons_frame, "Nombrar Radiocromicas", self.mostrar_dialogo_nombres).pack(pady=3, fill="x")
        styled_button(main_buttons_frame, "Detectar círculos", self.detectar_circulos_y_calcular_dosis).pack(pady=3, fill="x")
        styled_button(main_buttons_frame, "Mapa 3D de dosis", self.generate_dose_map_3d).pack(pady=3, fill="x")

        # Grupo 3: Guardar datos
        save_frame = tk.LabelFrame(self.info_frame, text="Guardar Datos", 
                                  bg=fondo, fg=texto, font=("Segoe UI", 10, "bold"))
        save_frame.pack(pady=10, fill="x", padx=5)
        
        # Entrada nombre
        tk.Label(save_frame, text="Nombre medición:", bg=fondo, fg=texto).pack()
        self.name_entry = tk.Entry(save_frame, width=20, bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.name_entry.pack(pady=(0, 5))
        styled_button(save_frame, "Guardar medición", self.save_measurement).pack(pady=3, fill="x")
        styled_button(save_frame, "Ver lista de dosis", self.show_dose_list).pack(pady=3, fill="x")

        # Recuadro para mostrar resultados
        self.result_frame = tk.Frame(self.info_frame, bg="#2C2F48", bd=1, relief="solid")
        self.result_frame.pack(pady=10, fill="x", padx=5)

        self.dose_label = tk.Label(self.result_frame, text="Dosis: -", font=("Segoe UI", 11),
                                bg="#2C2F48", fg="white", anchor="w", justify="left")
        self.dose_label.pack(fill="x", padx=10, pady=(8, 0))

        self.dose_neta_label = tk.Label(self.result_frame, text="Dosis neta: -", font=("Segoe UI", 11),
                                bg="#2C2F48", fg="white", anchor="w", justify="left")
        self.dose_neta_label.pack(fill="x", padx=10, pady=(0, 0))

        self.std_label = tk.Label(self.result_frame, text="Desviación estándar: -", font=("Segoe UI", 10),
                                bg="#2C2F48", fg="white", anchor="w", justify="left")
        self.std_label.pack(fill="x", padx=10, pady=(0, 8))

        # --- Frame del Canvas (en el centro) ---
        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        # Crear un frame contenedor para el canvas que se expandirá
        self.canvas_container = tk.Frame(self.canvas_frame)
        self.canvas_container.grid(row=0, column=0, sticky="nsew")
        
        # Configurar el canvas para que se expanda con la ventana
        self.canvas = tk.Canvas(self.canvas_container, bg="black", cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        # Scrollbars
        self.x_scroll = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.x_scroll.grid(row=1, column=0, sticky="ew")

        self.y_scroll = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.y_scroll.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(xscrollcommand=self.x_scroll.set, yscrollcommand=self.y_scroll.set)

        # Configurar el grid para que se expanda
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        # --- Frame de herramientas (ahora a la derecha) ---
        self.tools_frame = tk.Frame(main_frame, width=250, bg=fondo)
        self.tools_frame.pack(side="right", fill="y", padx=10, pady=10)

        # Grupo 2: Herramientas de medición (ahora a la derecha)
        tools_panel = tk.LabelFrame(self.tools_frame, text="Herramientas de Medición", 
                                   bg=fondo, fg=texto, font=("Segoe UI", 10, "bold"))
        tools_panel.pack(pady=10, fill="x", padx=5)
        
        # Selector de forma (círculo o rectángulo)
        shape_frame = tk.Frame(tools_panel, bg=fondo)
        shape_frame.pack(pady=5, fill="x")
        
        tk.Label(shape_frame, text="Seleccionar forma:", bg=fondo, fg=texto).pack(anchor="w")
        
        self.shape_var = tk.StringVar(value="circle")
        
        shape_options = tk.Frame(shape_frame, bg=fondo)
        shape_options.pack(fill="x", pady=2)
        
        circle_rb = tk.Radiobutton(shape_options, text="Círculo", variable=self.shape_var, 
                                  value="circle", command=self.on_shape_change,
                                  bg=fondo, fg=texto, selectcolor=boton_color, 
                                  activebackground=fondo, activeforeground=texto)
        circle_rb.pack(side="left", padx=(0, 10))
        
        rect_rb = tk.Radiobutton(shape_options, text="Rectángulo", variable=self.shape_var, 
                                value="rectangle", command=self.on_shape_change,
                                bg=fondo, fg=texto, selectcolor=boton_color, 
                                activebackground=fondo, activeforeground=texto)
        rect_rb.pack(side="left")
        
        # Frame para dimensiones
        self.dimensions_frame = tk.Frame(tools_panel, bg=fondo)
        self.dimensions_frame.pack(pady=5, fill="x")
        
        # Controles para círculo
        self.circle_controls = tk.Frame(self.dimensions_frame, bg=fondo)
        self.circle_controls.pack(fill="x")
        
        tk.Label(self.circle_controls, text="Radio (px):", bg=fondo, fg=texto).pack(side="left", padx=5)
        self.radius_entry = tk.Entry(self.circle_controls, width=5, bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.radius_entry.pack(side="left", padx=5)
        self.radius_entry.insert(0, "25")
        
        # Controles para rectángulo
        self.rect_controls = tk.Frame(self.dimensions_frame, bg=fondo)
        
        tk.Label(self.rect_controls, text="Ancho (px):", bg=fondo, fg=texto).grid(row=0, column=0, sticky="e", padx=5)
        self.width_entry = tk.Entry(self.rect_controls, width=5, bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.width_entry.grid(row=0, column=1, padx=5)
        self.width_entry.insert(0, "25")

        tk.Label(self.rect_controls, text="Alto (px):", bg=fondo, fg=texto).grid(row=1, column=0, sticky="e", padx=5)
        self.height_entry = tk.Entry(self.rect_controls, width=5, bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.height_entry.grid(row=1, column=1, padx=5)
        self.height_entry.insert(0, "25")

        # Inicialmente mostrar los controles según la forma seleccionada
        self.on_shape_change()
        
        # Botones de herramientas
        styled_button(tools_panel, "Añadir círculo manual", self.add_manual_circle).pack(pady=3, fill="x")
        styled_button(tools_panel, "Eliminar último círculo manual", self.remove_last_manual_circle).pack(pady=3, fill="x")
        
        # Selección de fondo para restar
        bg_frame = tk.Frame(tools_panel, bg=fondo)
        bg_frame.pack(pady=5, fill="x")
        
        tk.Label(bg_frame, text="Fondo a restar:", bg=fondo, fg=texto).pack(anchor="w")
        self.background_var = tk.DoubleVar(value=0.0)
        self.background_entry = tk.Entry(bg_frame, textvariable=self.background_var, width=8, 
                                        bg=entrada_fondo, fg=texto, insertbackground=texto)
        self.background_entry.pack(side="left", padx=5)
        
        styled_button(bg_frame, "Medir fondo", self.measure_background).pack(side="left", padx=5)

        # Canvas bindings
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-3>", self.start_drag)
        self.canvas.bind("<B3-Motion>", self.do_drag)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # --- Variables internas ---
        self.rect_width = 50
        self.rect_height = 50
        self.circle_radius = 25
        self.default_circle_radius = 25  # Radio por defecto para círculos detectados
        self.pil_img = None
        self.tk_image = None
        self.last_x = None
        self.last_y = None
        self.last_avg_dose = None
        self.last_std_dose = None
        self.next_area_id = 1  # Para generar IDs únicos para radiocromicas
        self.radiochromic_areas = []  # Lista para almacenar áreas radiocromicas detectadas
        self.current_shape = "circle"  # Forma seleccionada por defecto
        self.adding_manual_circle = False  # Flag para añadir círculo manual
        self.current_circle_data = None  # Para almacenar datos del círculo actual
        
        # Crear ventana para mostrar resultados de dosis
        self.create_dose_results_window()
        
        # Ventana para visualización interactiva de subcírculos
        self.subcircles_window = None
        self.subcircles_fig = None
        self.subcircles_ax = None
        self.subcircles_canvas = None
        self.subcircle_positions = []
        self.subcircle_patches = []
        self.subcircle_labels = []
        self.subcircle_info_text = None

    def on_canvas_configure(self, event):
        """Maneja el redimensionamiento del canvas"""
        # Actualizar la región de desplazamiento para incluir todo el contenido
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # Si hay una imagen cargada, asegurarse de que sea visible
        if hasattr(self, 'canvas_img'):
            # Obtener las dimensiones actuales del canvas
            canvas_width = event.width
            canvas_height = event.height
            
            # Obtener las dimensiones de la imagen
            if self.pil_img:
                img_width = self.pil_img.width
                img_height = self.pil_img.height
                
                # Ajustar la región de desplazamiento para incluir toda la imagen
                self.canvas.config(scrollregion=(0, 0, max(canvas_width, img_width), max(canvas_height, img_height)))

    def create_dose_results_window(self):
        """Crea una ventana para mostrar los resultados de dosis"""
        self.results_window = tk.Toplevel(self.root)
        self.results_window.title("Resultados de Dosis")
        self.results_window.geometry("600x500")
        self.results_window.protocol("WM_DELETE_WINDOW", lambda: self.results_window.withdraw())
        self.results_window.withdraw()  # Ocultar inicialmente
        
        # Frame principal
        main_frame = tk.Frame(self.results_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Crear un widget de texto con scroll para mostrar los resultados
        self.results_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, 
                                                     width=70, height=25, 
                                                     font=("Consolas", 10))
        self.results_text.pack(fill="both", expand=True)
        
        # Botón para guardar resultados
        button_frame = tk.Frame(self.results_window)
        button_frame.pack(fill="x", pady=10)
        
        save_button = tk.Button(button_frame, text="Guardar Resultados", 
                               command=self.save_results_to_file,
                               bg="#3E4A61", fg="white", font=("Segoe UI", 10),
                               relief="flat", bd=0, padx=10, pady=6)
        save_button.pack(side="right", padx=10)
        
        # Configurar estilos para el texto
        self.results_text.tag_configure("header", font=("Segoe UI", 11, "bold"))
        self.results_text.tag_configure("subheader", font=("Segoe UI", 10, "bold"))
        self.results_text.tag_configure("normal", font=("Consolas", 10))

    def save_results_to_file(self):
        """Guarda los resultados en un archivo de texto"""
        if not self.radiochromic_areas:
            print("⚠️ No hay datos para guardar.")
            return
            
        # Obtener la fecha y hora actual
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Crear nombre de archivo
        filename = f"Resultados_Dosis_{date_str}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as file:
                # Información general
                file.write("=== RESULTADOS DE ANÁLISIS DE DOSIS ===\n\n")
                
                if hasattr(self, 'image_path'):
                    file.write(f"Imagen: {os.path.basename(self.image_path)}\n")
                file.write(f"Fecha de análisis: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Contar círculos totales
                total_circles = sum(len(area["circles"]) for area in self.radiochromic_areas)
                file.write(f"Número de radiocromicas: {len(self.radiochromic_areas)}\n")
                file.write(f"Número total de círculos: {total_circles}\n\n")
                
                # Detalles por radiocromica
                for area in self.radiochromic_areas:
                    file.write(f"=== {area['name']} ===\n")
                    
                    if not area["circles"]:
                        file.write("  No hay círculos en esta área.\n\n")
                        continue
                    
                    # Obtener el fondo del área
                    background = self.get_area_background(area)
                    file.write(f"  Fondo del área: {background:.4f} Gy\n")
                    
                    # Estadísticas del área
                    doses = [circle["mean_dose"] for circle in area["circles"]]
                    mean_dose = np.mean(doses)
                    std_dose = np.std(doses, ddof=1) if len(doses) > 1 else 0
                    
                    file.write(f"  Círculos: {len(area['circles'])}\n")
                    file.write(f"  Dosis promedio: {mean_dose:.4f} Gy\n")
                    file.write(f"  Desviación: {std_dose:.4f} Gy\n\n")
                    
                    # Tabla de círculos
                    file.write("  ID\tDosis (Gy)\tDosis-Fondo (Gy)\tσ (Gy)\n")
                    file.write("  " + "-" * 50 + "\n")
                    
                    # Ordenar círculos según el esquema proporcionado (A, B, C en la primera fila, D, E, F en la segunda)
                    circles_with_positions = []
                    for i, circle in enumerate(area["circles"]):
                        circles_with_positions.append((circle, i))
                    
                    # Ordenar círculos por posición (primero por fila, luego por columna de derecha a izquierda en la primera fila
                    # y de izquierda a derecha en la segunda fila)
                    sorted_circles = self.sort_circles_by_position(circles_with_positions, area["coords"])
                    
                    for i, (circle, _) in enumerate(sorted_circles):
                        # Asignar letra según el esquema proporcionado
                        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
                        circle_id = f"{area['name']}_{letters[i % len(letters)]}"
                        
                        # Marcar círculos manuales
                        if "manual" in circle and circle["manual"]:
                            circle_id += " (M)"
                            
                        dose = max(0, circle["mean_dose"])  # Asegurar que la dosis no sea negativa
                        dose_bg_corrected = max(0, dose - background)  # Asegurar que la dosis corregida no sea negativa
                        std = circle["std"]
                        
                        file.write(f"  {circle_id}\t{dose:.4f}\t{dose_bg_corrected:.4f}\t{std:.4f}\n")
                    
                    file.write("\n")
                
            print(f"✅ Resultados guardados en '{filename}'")
            
            # Mostrar mensaje en la ventana de resultados
            self.results_text.config(state="normal")
            self.results_text.insert(tk.END, f"\n\n✅ Resultados guardados en '{filename}'", "header")
            self.results_text.config(state="disabled")
            
        except Exception as e:
            print(f"❌ Error al guardar resultados: {e}")

    def save_subcircles_to_file(self):
        """Guarda los resultados de los subcírculos en un archivo de texto"""
        if not self.subcircles_data:
            print("⚠️ No hay datos de subcírculos para guardar.")
            return
            
        # Obtener la fecha y hora actual
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Crear nombre de archivo
        filename = f"Resultados_Pocillos_{date_str}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as file:
                # Información general
                file.write("=== RESULTADOS DE ANÁLISIS DE POCILLOS ===\n\n")
                
                if hasattr(self, 'image_path'):
                    file.write(f"Imagen: {os.path.basename(self.image_path)}\n")
                file.write(f"Fecha de análisis: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Información del círculo principal
                if self.current_circle_data:
                    circle_id = self.current_circle_data.get("id", "Desconocido")
                    area_name = self.current_circle_data.get("area_name", "Desconocida")
                    background = self.current_circle_data.get("background", 0)
                    
                    file.write(f"Círculo principal: {circle_id}\n")
                    file.write(f"Área radiocrómica: {area_name}\n")
                    file.write(f"Fondo: {background:.4f} Gy\n\n")
                
                # Tabla de subcírculos
                file.write("Pocillo\tDosis (Gy)\tσ (Gy)\tError\n")
                file.write("-" * 50 + "\n")
                
                for i, data in enumerate(self.subcircles_data):
                    mean_dose = data.get("mean_dose", 0)
                    std_dose = data.get("std_dose", 0)
                    error = (std_dose / mean_dose) if mean_dose > 0 else 0
                    
                    file.write(f"{i+1}\t{mean_dose:.4f}\t{std_dose:.4f}\t{error:.4f}\n")
                
                # Estadísticas globales
                if self.subcircles_data:
                    doses = [data.get("mean_dose", 0) for data in self.subcircles_data]
                    mean_global = np.mean(doses)
                    std_global = np.std(doses, ddof=1) if len(doses) > 1 else 0
                    cv_global = (std_global / mean_global) if mean_global > 0 else 0
                    
                    file.write("\nEstadísticas globales:\n")
                    file.write(f"Dosis promedio: {mean_global:.4f} Gy\n")
                    file.write(f"Desviación estándar: {std_global:.4f} Gy\n")
                    file.write(f"Coeficiente de variación: {cv_global:.4f}\n")
                
            print(f"✅ Resultados de pocillos guardados en '{filename}'")
            
            # Mostrar mensaje en la ventana de subcírculos si está abierta
            if self.subcircles_window and self.subcircles_window.winfo_exists():
                messagebox.showinfo("Guardado", f"Resultados guardados en '{filename}'")
            
        except Exception as e:
            print(f"❌ Error al guardar resultados de pocillos: {e}")
            if self.subcircles_window and self.subcircles_window.winfo_exists():
                messagebox.showerror("Error", f"Error al guardar resultados: {e}")

    def sort_circles_by_position(self, circles_with_positions, area_coords):
        """Ordena los círculos según el esquema proporcionado"""
        x1, y1, x2, y2 = area_coords
        area_width = x2 - x1
        area_height = y2 - y1
        
        # Determinar el punto medio del área
        mid_y = y1 + area_height / 2
        
        # Separar círculos en fila superior e inferior
        top_row = []
        bottom_row = []
        
        for circle, idx in circles_with_positions:
            if circle["y"] < mid_y:
                top_row.append((circle, idx))
            else:
                bottom_row.append((circle, idx))
        
        # Ordenar fila superior de derecha a izquierda (A, B, C)
        top_row.sort(key=lambda c: -c[0]["x"])
        
        # Ordenar fila inferior de izquierda a derecha (D, E, F)
        bottom_row.sort(key=lambda c: c[0]["x"])
        
        # Combinar las filas
        return top_row + bottom_row

    def show_dose_list(self):
        """Muestra la lista de dosis en la ventana de resultados"""
        if not self.radiochromic_areas:
            print("⚠️ No hay áreas radiocromicas definidas.")
            return
            
        self.update_dose_results_display()
        self.results_window.deiconify()  # Mostrar la ventana

    def get_area_background(self, area):
        """Obtiene el fondo del área usando solo el valor medido con el botón 'Medir fondo'"""
        # Usar solo el valor medido con el botón "Medir fondo"
        return self.background_var.get()

    def measure_background(self):
        """Mide el fondo en un área no irradiada"""
        if self.pil_img is None:
            print("⚠️ No hay imagen cargada.")
            return
            
        # Cambiar el cursor para indicar que estamos en modo de medición de fondo
        self.canvas.config(cursor="target")
        self.canvas.bind("<Button-1>", self.on_background_click)
        
        # Mostrar mensaje al usuario
        print("🎯 Haga clic en un área no irradiada para medir el fondo.")
        
    def on_background_click(self, event):
        """Maneja el clic para medir el fondo"""
        if self.pil_img is None:
            return
            
        # Obtener coordenadas
        x_center = self.canvas.canvasx(event.x)
        y_center = self.canvas.canvasy(event.y)
        
        # Usar un pequeño rectángulo para medir el fondo
        size = 10
        x1 = int(x_center - size)
        y1 = int(y_center - size)
        x2 = int(x_center + size)
        y2 = int(y_center + size)
        
        # Asegurar límites
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, self.pil_img.width)
        y2 = min(y2, self.pil_img.height)
        
        # Obtener bloque de imagen
        cut = np.array(self.pil_img)[y1:y2, x1:x2]
        if cut.size == 0:
            print("⚠️ Región vacía.")
            return
            
        # Calcular dosis de fondo
        background_dose = max(0, self.calcular_dosis_promedio(cut))  # Asegurar que el fondo no sea negativo
        self.background_var.set(round(background_dose, 4))
        
        # Restaurar cursor y binding
        self.canvas.config(cursor="cross")
        self.canvas.bind("<Button-1>", self.on_click)
        
        print(f"🔍 Fondo medido: {background_dose:.4f} Gy")
        
        # Dibujar un pequeño indicador donde se midió el fondo
        self.canvas.delete("background_marker")  # Eliminar marcadores anteriores
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", tags="background_marker")
        self.canvas.create_text(x_center, y_center - 15, text=f"{background_dose:.4f} Gy", 
                               fill="yellow", tags="background_marker")

    def _on_mousewheel(self, event):
        if event.state & 0x0001:
            self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        else:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def start_drag(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_shape_change(self):
        """Maneja el cambio entre círculo y rectángulo"""
        self.current_shape = self.shape_var.get()
        
        # Ocultar y mostrar los controles apropiados
        if self.current_shape == "circle":
            self.rect_controls.pack_forget()
            self.circle_controls.pack(fill="x")
        else:
            self.circle_controls.pack_forget()
            self.rect_controls.pack(fill="x")

    def update_size(self):
        """Actualiza las dimensiones según la forma seleccionada"""
        try:
            if self.current_shape == "circle":
                self.circle_radius = int(self.radius_entry.get())
                print(f"Nuevo radio actualizado: {self.circle_radius}")
            else:
                self.rect_width = int(self.width_entry.get())
                self.rect_height = int(self.height_entry.get())
                print(f"Nuevo tamaño actualizado: {self.rect_width} x {self.rect_height}")
        except ValueError:
            print("Error: valores inválidos.")

    def calcular_dosis_promedio(self, bloque_rgb):
        R, G, B = bloque_rgb[:, :, 0], bloque_rgb[:, :, 1], bloque_rgb[:, :, 2]
        try:
            doses = [
                redCali[0] + (redCali[1] / (np.mean(R[R > 0]) - redCali[2])),
                greenCali[0] + (greenCali[1] / (np.mean(G[G > 0]) - greenCali[2])),
                blueCali[0] + (blueCali[1] / (np.mean(B[B > 0]) - blueCali[2]))
            ]
            return max(0, np.mean(doses))  # Asegurar que la dosis no sea negativa
        except:
            return 0    

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tiff")])
        if not self.image_path:
            print("No se seleccionó imagen.")
            return

        self.original_img = cv2.imread(self.image_path)
        if self.original_img is None:
            print("Error al cargar la imagen.")
            return

        if self.original_img.dtype != np.uint8:
            self.original_img = cv2.convertScaleAbs(self.original_img)

        self.rgb_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(self.rgb_img)

        max_size = 800
        if self.pil_img.width > max_size or self.pil_img.height > max_size:
            self.pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(self.pil_img)
        self.canvas.config(scrollregion=(0, 0, self.pil_img.width, self.pil_img.height))
        self.canvas.delete("all")
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, self.pil_img.width, self.pil_img.height))
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.radiochromic_areas = []
        self.detected_circles = []
        self.manual_circles = []
        self.update_dose_results_display()

        print(f"✅ Imagen cargada: {os.path.basename(self.image_path)}")

    def on_resize(self, event):
        # Manejar el redimensionamiento del canvas
        pass

    def on_leave(self, event):
        # Manejar cuando el cursor sale del canvas
        pass

    def update_dose_results_display(self):
        """Actualiza la ventana de resultados con los datos de todas las áreas"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        if not self.radiochromic_areas:
            self.results_text.insert(tk.END, "No hay áreas radiocromicas definidas.")
            self.results_text.config(state="disabled")
            return
        
        # Información general
        self.results_text.insert(tk.END, "=== RESULTADOS DE ANÁLISIS DE DOSIS ===\n\n", "header")
        
        if hasattr(self, 'image_path'):
            self.results_text.insert(tk.END, f"Imagen: {os.path.basename(self.image_path)}\n", "normal")
        
        now = datetime.now()
        self.results_text.insert(tk.END, f"Fecha: {now.strftime('%Y-%m-%d %H:%M:%S')}\n", "normal")
        
        # Contar círculos totales
        total_circles = sum(len(area["circles"]) for area in self.radiochromic_areas)
        self.results_text.insert(tk.END, f"Número de radiocromicas: {len(self.radiochromic_areas)}\n", "normal")
        self.results_text.insert(tk.END, f"Número total de círculos: {total_circles}\n\n", "normal")
        
        # Detalles por radiocromica
        for area in self.radiochromic_areas:
            self.results_text.insert(tk.END, f"=== {area['name']} ===\n", "header")
            
            if not area["circles"]:
                self.results_text.insert(tk.END, "  No hay círculos en esta área.\n\n", "normal")
                continue
            
            # Obtener el fondo del área
            background = self.get_area_background(area)
            self.results_text.insert(tk.END, f"  Fondo del área: {background:.4f} Gy\n", "normal")
                
            # Estadísticas del área
            doses = [max(0, circle["mean_dose"]) for circle in area["circles"]]  # Asegurar que las dosis no sean negativas
            mean_dose = np.mean(doses)
            std_dose = np.std(doses, ddof=1) if len(doses) > 1 else 0
            
            self.results_text.insert(tk.END, f"  Círculos: {len(area['circles'])}\n", "normal")
            self.results_text.insert(tk.END, f"  Dosis promedio: {mean_dose:.4f} Gy\n", "normal")
            self.results_text.insert(tk.END, f"  Desviación: {std_dose:.4f} Gy\n\n", "normal")
            
            # Tabla de círculos
            self.results_text.insert(tk.END, "  ID\tDosis (Gy)\tDosis-Fondo (Gy)\tσ (Gy)\n", "subheader")
            self.results_text.insert(tk.END, "  " + "-" * 50 + "\n", "normal")
            
            # Ordenar círculos según el esquema proporcionado
            circles_with_positions = []
            for i, circle in enumerate(area["circles"]):
                circles_with_positions.append((circle, i))
            
            sorted_circles = self.sort_circles_by_position(circles_with_positions, area["coords"])
            
            for i, (circle, _) in enumerate(sorted_circles):
                # Asignar letra según el esquema proporcionado
                letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
                circle_id = f"{area['name']}_{letters[i % len(letters)]}"
                
                # Marcar círculos manuales
                if "manual" in circle and circle["manual"]:
                    circle_id += " (M)"
                
                dose = max(0, circle["mean_dose"])  # Asegurar que la dosis no sea negativa
                dose_bg_corrected = max(0, dose - background)  # Asegurar que la dosis corregida no sea negativa
                std = circle["std"]
                
                self.results_text.insert(tk.END, 
                    f"  {circle_id}\t{dose:.4f}\t{dose_bg_corrected:.4f}\t{std:.4f}\n", 
                    "normal")
                
            self.results_text.insert(tk.END, "\n\n", "normal")
            
        self.results_text.config(state="disabled")

    def detectar_areas_radiocromicas(self):
        if self.pil_img is None:
            print("⚠️ No hay imagen cargada.")
            return
            
        img_rgb = np.array(self.pil_img)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Aplicar umbral para detectar áreas oscuras (radiocromicas)
        _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos pequeños
        min_area = 5000  # Ajustar según el tamaño de las áreas radiocromicas
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not valid_contours:
            print("⚠️ No se detectaron áreas radiocromicas.")
            return
            
        print(f"🔍 Se detectaron {len(valid_contours)} áreas radiocromicas.")
        
        # Limpiar áreas anteriores
        self.canvas.delete("radiochromic")
        self.radiochromic_areas = []
        
        # Procesar cada área
        for idx, contour in enumerate(valid_contours):
            # Obtener rectángulo que encierra el contorno
            x, y, w, h = cv2.boundingRect(contour)
            
            # Nombre por defecto
            area_name = f"RC#{idx+1}"
            
            # Crear área radiocromica
            area = {
                "name": area_name,
                "coords": (x, y, x+w, y+h),
                "circles": []  # Lista para almacenar círculos dentro del área
            }
            
            self.radiochromic_areas.append(area)
            
            # Dibujar rectángulo
            self.canvas.create_rectangle(
                x, y, x+w, y+h,
                outline='blue', width=2, tags="radiochromic"
            )
            
            # Añadir etiqueta con nombre
            self.canvas.create_text(
                x+5, y+5,
                text=area_name,
                anchor="nw",
                fill="white",
                tags="radiochromic"
            )
            
        print("✅ Detección de áreas radiocromicas completada.")

    def mostrar_dialogo_nombres(self):
        """Muestra un diálogo para nombrar las áreas radiocromicas"""
        if not self.radiochromic_areas:
            print("⚠️ No hay áreas radiocromicas detectadas.")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Nombrar áreas radiocromicas")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Frame para la lista de áreas
        frame = tk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        tk.Label(frame, text="Asignar nombres a las áreas:").pack(anchor="w")
        
        # Crear entradas para cada área
        entries = []
        for i, area in enumerate(self.radiochromic_areas):
            area_frame = tk.Frame(frame)
            area_frame.pack(fill="x", pady=5)
            
            tk.Label(area_frame, text=f"Área {i+1}:").pack(side="left")
            entry = tk.Entry(area_frame, width=30)
            entry.pack(side="left", padx=5)
            entry.insert(0, area["name"])
            entries.append((area, entry))
        
        # Botón para guardar
        def guardar_nombres():
            for area, entry in entries:
                nombre = entry.get().strip()
                if nombre:
                    area["name"] = nombre
                    
                    # Actualizar etiqueta en el canvas
                    x, y, _, _ = area["coords"]
                    for item in self.canvas.find_withtag("radiochromic"):
                        if self.canvas.type(item) == "text":
                            coords = self.canvas.coords(item)
                            if abs(coords[0] - x - 5) < 10 and abs(coords[1] - y - 5) < 10:
                                self.canvas.itemconfig(item, text=nombre)
                                break
            
            dialog.destroy()
            print("✅ Nombres de áreas actualizados.")
        
        tk.Button(dialog, text="Guardar", command=guardar_nombres).pack(pady=10)

    def remove_last_manual_circle(self):
        """Elimina el último círculo manual añadido"""
        if not self.manual_circles:
            print("⚠️ No hay círculos manuales para eliminar.")
            return
            
        # Obtener el último círculo manual añadido
        x, y, r, area_idx = self.manual_circles.pop()
        
        # Buscar y eliminar el círculo del área correspondiente
        for i, circle in enumerate(self.radiochromic_areas[area_idx]["circles"]):
            if "manual" in circle and circle["manual"] and circle["x"] == x and circle["y"] == y and circle["r"] == r:
                self.radiochromic_areas[area_idx]["circles"].pop(i)
                break
        
        # Eliminar el círculo de la lista de círculos detectados para que no se pueda hacer clic derecho
        for i, (cx, cy, cr) in enumerate(self.detected_circles):
            if cx == x and cy == y and cr == r:
                self.detected_circles.pop(i)
                break
        
        # Redibujar todos los círculos manuales
        self.canvas.delete("manual_circle")
        for x, y, r, area_idx in self.manual_circles:
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline='yellow', width=2, tags="manual_circle"
            )
            
            # Añadir etiqueta con la dosis
            for circle in self.radiochromic_areas[area_idx]["circles"]:
                if "manual" in circle and circle["manual"] and circle["x"] == x and circle["y"] == y:
                    self.canvas.create_text(
                        x, y,
                        text=f"{circle['mean_dose']:.2f} Gy",
                        fill="yellow",
                        tags="manual_circle"
                    )
                    break
        
        # Actualizar la ventana de resultados
        self.update_dose_results_display()
        
        print(f"✅ Último círculo manual eliminado.")

    def add_manual_circle(self):
        """Activa el modo para añadir un círculo manualmente"""
        if self.pil_img is None:
            print("⚠️ No hay imagen cargada.")
            return
            
        if not self.radiochromic_areas:
            print("⚠️ Primero debe detectar áreas radiocromicas.")
            return
            
        # Cambiar el cursor y activar el modo de añadir círculo
        self.canvas.config(cursor="target")
        self.adding_manual_circle = True
        
        # Cambiar el binding del clic
        self.canvas.bind("<Button-1>", self.on_manual_circle_click)
        
        print("🎯 Haga clic para añadir un círculo manual.")

    def on_manual_circle_click(self, event):
        """Maneja el clic para añadir un círculo manual"""
        if not self.adding_manual_circle:
            return
            
        # Obtener coordenadas
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Usar el 80% del radio de los círculos detectados
        #r = int(self.default_circle_radius * 0.8)
        r=35
        # Determinar a qué área radiocromica pertenece
        area_idx = self.find_radiochromic_area(x, y)
        
        if area_idx is None:
            print("⚠️ El círculo debe estar dentro de un área radiocromica.")
            return
            
        # Procesar el círculo
        img_rgb = np.array(self.pil_img)
        resultado = self.procesar_circulo(img_rgb, int(x), int(y), r)
        
        # Marcar como manual
        resultado["manual"] = True
        
        # Añadir círculo al área correspondiente
        self.radiochromic_areas[area_idx]["circles"].append(resultado)
        
        # Añadir a la lista de círculos manuales
        self.manual_circles.append((int(x), int(y), r, area_idx))
        
        # Dibujar el círculo
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline='yellow', width=2, tags="manual_circle"
        )
        
        # Añadir etiqueta con la dosis
        self.canvas.create_text(
            x, y,
            text=f"{resultado['mean_dose']:.2f} Gy",
            fill="yellow",
            tags="manual_circle"
        )
        
        # Actualizar la ventana de resultados
        self.update_dose_results_display()
        
        print(f"✅ Círculo manual añadido en ({int(x)}, {int(y)}) con dosis {resultado['mean_dose']:.4f} Gy")
        
        # Desactivar el modo de añadir círculo
        self.adding_manual_circle = False
        self.canvas.config(cursor="cross")
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Añadir el círculo manual a la lista de círculos detectados para poder hacer clic derecho
        self.detected_circles.append((int(x), int(y), r))

    def find_radiochromic_area(self, x, y):
        """Encuentra el índice del área radiocromica que contiene el punto (x,y)"""
        for i, area in enumerate(self.radiochromic_areas):
            x1, y1, x2, y2 = area["coords"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def on_click(self, event):
        if self.pil_img is None:
            return

        self.update_size()

        # Corregir coordenadas absolutas
        x_center = self.canvas.canvasx(event.x)
        y_center = self.canvas.canvasy(event.y)

        # Limpiar selecciones anteriores
        self.canvas.delete("rect")
        self.canvas.delete("circle")

        # Dibujar y procesar según la forma seleccionada
        if self.current_shape == "circle":
            # Obtener radio del círculo
            r = self.circle_radius
            
            # Dibujar círculo
            self.canvas.create_oval(
                x_center - r, y_center - r, x_center + r, y_center + r,
                outline="red", tags="circle"
            )
            
            # Procesar área circular
            img_rgb = np.array(self.pil_img)
            mask = np.zeros((self.pil_img.height, self.pil_img.width), dtype=np.uint8)
            cv2.circle(mask, (int(x_center), int(y_center)), r, 255, -1)
            
            # Aplicar máscara
            masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
            
            # Recortar región de interés
            x1 = max(0, int(x_center - r))
            y1 = max(0, int(y_center - r))
            x2 = min(self.pil_img.width, int(x_center + r))
            y2 = min(self.pil_img.height, int(y_center + r))
            
            cut = masked[y1:y2, x1:x2]
            
        else:  # Rectángulo
            x1 = int(x_center - self.rect_width // 2)
            y1 = int(y_center - self.rect_height // 2)
            x2 = int(x_center + self.rect_width // 2)
            y2 = int(y_center + self.rect_height // 2)

            # Asegurar límites
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, self.pil_img.width)
            y2 = min(y2, self.pil_img.height)

            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", tags="rect")
            
            cut = np.array(self.pil_img)[y1:y2, x1:x2]
            
        if cut.size == 0:
            self.dose_label.config(text="Dosis: región vacía")
            self.dose_neta_label.config(text="Dosis neta: región vacía")
            return

        # Calcular dosis
        area_idx = self.find_radiochromic_area(x_center, y_center)
        background = 0.0
        if area_idx is not None:
            background = self.get_area_background(self.radiochromic_areas[area_idx])
        
        R, G, B = cut[:, :, 0], cut[:, :, 1], cut[:, :, 2]
        try:
            dose = [
                redCali[0] + (redCali[1] / (np.mean(R[R > 0]) - redCali[2])),
                greenCali[0] + (greenCali[1] / (np.mean(G[G > 0]) - greenCali[2])),
                blueCali[0] + (blueCali[1] / (np.mean(B[B > 0]) - blueCali[2]))
            ]
            avg_dose = max(0, np.mean(dose))  # Asegurar que la dosis no sea negativa
            std_dose = np.std(dose, ddof=1)
            
            # Restar fondo si es necesario
            avg_dose_corrected = max(0, avg_dose - background)  # Asegurar que la dosis corregida no sea negativa

            self.dose_label.config(text=f"Dosis: {avg_dose:.4f} Gy")
            self.dose_neta_label.config(text=f"Dosis neta: {avg_dose_corrected:.4f} Gy")
            self.std_label.config(text=f"Desviación estándar: {std_dose:.4f} Gy")

            # Guardar la medición temporal
            self.last_x = x_center
            self.last_y = y_center
            self.last_avg_dose = avg_dose
            self.last_std_dose = std_dose

        except Exception as e:
            self.dose_label.config(text="Error en cálculo")
            self.dose_neta_label.config(text="Error en cálculo")
            print("Error:", e)

    def save_measurement(self):
        if self.last_x is None or self.last_y is None:
            print("No hay medición lista para guardar.")
            return

        filename = "Mediciones.csv"
        file_exists = os.path.isfile(filename)

        # Leer el nombre escrito
        name = self.name_entry.get().strip()
        if not name:
            name = f"Medicion_{self.last_x}_{self.last_y}"

        # Obtener área y fondo correspondiente
        area_idx = self.find_radiochromic_area(self.last_x, self.last_y)
        background = 0.0
        if area_idx is not None:
            background = self.get_area_background(self.radiochromic_areas[area_idx])
        
        corrected_dose = max(0, self.last_avg_dose - background)  # Asegurar que la dosis corregida no sea negativa

        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter='\t')

            if not file_exists:
                writer.writerow(["Imagen", "Nombre_Medicion", "Dosis_Promedio", "Dosis_Neta", "Desviacion_Estandar", "Fondo"])

            writer.writerow([
                os.path.basename(self.image_path),
                name,
                f"{self.last_avg_dose:.4f}",
                f"{corrected_dose:.4f}",
                f"{self.last_std_dose:.4f}",
                f"{background:.4f}"
            ])

        # Limpiar nombre luego de guardar
        self.name_entry.delete(0, tk.END)

        # Opcional: limpiar medición temporal
        self.last_x = None
        self.last_y = None
        self.last_avg_dose = None
        self.last_std_dose = None

        print(f"Medición '{name}' guardada exitosamente.")

    def generate_dose_map_3d(self):
        if self.pil_img is None:
            print("No hay imagen cargada.")
            return

        # Convertir imagen a array
        img_array = np.array(self.pil_img)

        # Definir resolución del grid (más alto = menos detalle, más rápido)
        step = 5  # píxeles por bloque
        h, w, _ = img_array.shape
        dose_map = []

        # Crear un mapa de fondos por píxel
        background_map = np.zeros((h, w))
        
        # Aplicar el fondo correspondiente a cada área
        for area in self.radiochromic_areas:
            x1, y1, x2, y2 = area["coords"]
            background = self.get_area_background(area)
            background_map[y1:y2, x1:x2] = background

        for y in range(0, h, step):
            row = []
            for x in range(0, w, step):
                block = img_array[y:y+step, x:x+step]
                if block.size == 0:
                    row.append(0)
                    continue
                avg_dose = self.calcular_dosis_promedio(block)
                # Restar fondo correspondiente a la posición
                background = background_map[y, x]
                avg_dose = max(0, avg_dose - background)  # Asegurar que la dosis no sea negativa
                row.append(avg_dose)
            if row:  # Asegurarse de que la fila no esté vacía
                dose_map.append(row)

        dose_map = np.array(dose_map)

        # Crear malla de coordenadas
        X = np.arange(0, dose_map.shape[1])
        Y = np.arange(0, dose_map.shape[0])
        X, Y = np.meshgrid(X, Y)

        valores_validos = dose_map[dose_map > 0]
        if len(valores_validos) == 0:
            print("No hay valores válidos para generar el mapa 3D.")
            return
            
        mean_dose = np.mean(valores_validos)
        min_dose = np.min(valores_validos)
        max_dose = np.max(valores_validos)
       
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        # Establecer límites para mostrar todo el rango desde 0 hasta el máximo
        zmin = 0  # Mostrar desde cero para ver la "caída"
        zmax = max_dose
        ax.set_zlim(zmin, zmax)
        
        # Graficar superficie con todos los valores
        surf = ax.plot_surface(X, Y, dose_map, cmap=cm.viridis)
        
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Dosis neta (Gy)')
        ax.set_title(f"Mapa 3D de dosis")
        ax.set_xlabel("X (bloques)")
        ax.set_ylabel("Y (bloques)")
        ax.set_zlabel("Dosis (Gy)")
        plt.tight_layout()
        plt.show()    

    def detectar_circulos_y_calcular_dosis(self):
        if self.pil_img is None:
            print("⚠️ No hay imagen cargada.")
            return

        if not self.radiochromic_areas:
            print("⚠️ Primero debe detectar áreas radiocromicas.")
            return

        img_rgb = np.array(self.pil_img)
        
        # Limpiar círculos anteriores
        self.canvas.delete("circle_detect")
        self.detected_circles = []
        
        # Guardar el radio de los círculos detectados para usarlo en círculos manuales
        self.default_circle_radius = 25  # Valor por defecto
        
        # Procesar cada área radiocromica por separado
        for area_idx, area in enumerate(self.radiochromic_areas):
            x1, y1, x2, y2 = area["coords"]
            
            # Recortar el área
            area_img = img_rgb[y1:y2, x1:x2]
            
            # Convertir a escala de grises
            area_gray = cv2.cvtColor(area_img, cv2.COLOR_RGB2GRAY)
            area_blur = cv2.medianBlur(area_gray, 5)
            
            # Detectar círculos con Hough
            circles = cv2.HoughCircles(
                area_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
            
            # Limpiar círculos anteriores del área
            # Mantener solo los círculos manuales
            manual_circles = [c for c in area["circles"] if "manual" in c and c["manual"]]
            area["circles"] = manual_circles
            
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                print(f"🔍 Se detectaron {len(circles)} círculos en {area['name']}.")
                
                # Si hay círculos detectados, usar el radio promedio para los círculos manuales
                if len(circles) > 0:
                    avg_radius = int(np.mean([r for _, _, r in circles]))
                    self.default_circle_radius = avg_radius
                
                # Crear una máscara para evitar intersecciones
                intersection_mask = np.zeros((area_img.shape[0], area_img.shape[1]), dtype=np.uint8)
                
                # Primero, dibujar todos los círculos en la máscara
                for cx, cy, r in circles:
                    cv2.circle(intersection_mask, (cx, cy), r, 255, -1)
                
                # Procesar cada círculo
                for idx, (cx, cy, r) in enumerate(circles):
                    # Ajustar coordenadas al sistema global
                    x_global = x1 + cx
                    y_global = y1 + cy
                    
                    # Crear una máscara individual para este círculo
                    circle_mask = np.zeros_like(intersection_mask)
                    cv2.circle(circle_mask, (cx, cy), r, 255, -1)
                    
                    # Identificar intersecciones con otros círculos
                    # Restar la máscara del círculo actual de la máscara total
                    temp_mask = intersection_mask.copy()
                    cv2.circle(temp_mask, (cx, cy), r, 0, -1)  # Quitar el círculo actual
                    
                    # Ahora circle_mask contiene solo este círculo
                    # temp_mask contiene todos los demás círculos
                    # Las intersecciones son donde ambos son 255
                    intersections = cv2.bitwise_and(circle_mask, temp_mask)
                    
                    # Crear una máscara que excluya las intersecciones
                    clean_mask = cv2.subtract(circle_mask, intersections)
                    
                    # Aplicar la máscara limpia a la imagen
                    masked_area = cv2.bitwise_and(area_img, area_img, mask=clean_mask)
                    
                    # Procesar el círculo con la máscara limpia
                    resultado = self.procesar_circulo_con_mascara(img_rgb, x_global, y_global, r, clean_mask, x1, y1)
                    
                    # Guardar resultado en el área
                    area["circles"].append(resultado)
                    
                    # Añadir a la lista global de círculos detectados
                    self.detected_circles.append((x_global, y_global, r))
                    
                    # Dibujar círculo
                    self.canvas.create_oval(
                        x_global - r, y_global - r, x_global + r, y_global + r,
                        outline='green', width=2, tags="circle_detect"
                    )
                    
                    # Obtener la letra según el esquema proporcionado
                    # Ordenar círculos según el esquema proporcionado
                    circles_with_positions = []
                    for i, circle in enumerate(area["circles"]):
                        circles_with_positions.append((circle, i))
                    
                    sorted_circles = self.sort_circles_by_position(circles_with_positions, area["coords"])
                    
                    # Encontrar la posición del círculo actual en la lista ordenada
                    circle_idx = -1
                    for i, (circle, _) in enumerate(sorted_circles):
                        if circle["x"] == x_global and circle["y"] == y_global:
                            circle_idx = i
                            break
                    
                    # Asignar letra según el esquema proporcionado
                    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
                    circle_id = f"{area['name']}_{letters[circle_idx % len(letters)]}"
                    
                    # Añadir etiqueta con ID
                    self.canvas.create_text(
                        x_global, y_global - r - 10,
                        text=circle_id,                        
                        fill="white",
                        tags="circle_detect"
                    )
                    # ←–– Inserción para mostrar la dosis dentro del círculo:
                    dosis = None
                    for key in ("dosis", "mean_dose", "valor"):
                        if key in resultado:
                            dosis = resultado[key]
                            break
                    if dosis is not None:
                        dosis = resultado.get("mean_dose", None)
                        self.canvas.create_text(
                            x_global, y_global,
                            text=f"{dosis:.2f}",
                            fill="yellow",
                            tags="circle_detect"
                        )

            else:
                print(f"⚠️ No se detectaron círculos en {area['name']}.")
        
        # Añadir los círculos manuales a la lista de círculos detectados para poder hacer clic derecho
        for x, y, r, _ in self.manual_circles:
            if (x, y, r) not in self.detected_circles:
                self.detected_circles.append((x, y, r))
        
        # Activar clic derecho sobre círculo
        self.canvas.bind("<Button-3>", self.on_circle_click)
        
        # Actualizar ventana de resultados
        self.update_dose_results_display()
        
        print("✅ Análisis de círculos completado.")

    def procesar_circulo_con_mascara(self, img_rgb, x, y, r, clean_mask, x_offset, y_offset):
        """Procesa un círculo usando una máscara que excluye intersecciones"""
        h, w, _ = img_rgb.shape
        
        # Recorte de la imagen completa
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r)
        y2 = min(h, y + r)
        
        # Ajustar la máscara a las coordenadas globales
        mask_global = np.zeros((h, w), dtype=np.uint8)
        
        # Calcular las coordenadas dentro de la máscara local
        mask_h, mask_w = clean_mask.shape
        
        # Copiar la máscara limpia a la posición correcta en la máscara global
        mask_x1 = max(0, x_offset)
        mask_y1 = max(0, y_offset)
        mask_x2 = min(w, x_offset + mask_w)
        mask_y2 = min(h, y_offset + mask_h)
        
        # Asegurarse de que las dimensiones coincidan
        src_x1 = max(0, -x_offset)
        src_y1 = max(0, -y_offset)
        src_x2 = src_x1 + (mask_x2 - mask_x1)
        src_y2 = src_y1 + (mask_y2 - mask_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1 and mask_x2 > mask_x1 and mask_y2 > mask_y1:
            mask_global[mask_y1:mask_y2, mask_x1:mask_x2] = clean_mask[src_y1:src_y2, src_x1:src_x2]
        
        # Aplicar la máscara a la imagen completa
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_global)
        
        # Recortar la región de interés
        cut = masked_img[y1:y2, x1:x2]
        
        # Obtener valor de fondo
        area_idx = self.find_radiochromic_area(x, y)
        background = 0.0
        if area_idx is not None:
            background = self.get_area_background(self.radiochromic_areas[area_idx])
        
        # Calcular dosis en bloques pequeños
        step = 1
        dose_map = []
        for j in range(0, cut.shape[0], step):
            row = []
            for i in range(0, cut.shape[1], step):
                block = cut[j:j+step, i:i+step]
                if block.size == 0 or np.all(block == 0):
                    row.append(0)
                    continue
                dose = self.calcular_dosis_promedio(block)
                # Restar fondo
                dose = max(0, dose - background)  # Asegurar que la dosis no sea negativa
                row.append(dose)
            if row:  # Asegurarse de que la fila no esté vacía
                dose_map.append(row)
                
        if not dose_map:
            return {
                "x": x,
                "y": y,
                "r": r,
                "mean_dose": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "homo_std": 0,
                "homo_range": 0
            }
                
        dose_map = np.array(dose_map)
        
        # Estadísticas de homogeneidad
        valores_validos = dose_map[dose_map > 0]
        if len(valores_validos) == 0:
            mean_dose = 0
            std_dose = 0
            min_dose = 0
            max_dose = 0
            homogeneity_std = 0
            homogeneity_range = 0
        else:
            mean_dose = np.mean(valores_validos)
            std_dose = np.std(valores_validos, ddof=1)
            min_dose = np.min(valores_validos)
            max_dose = np.max(valores_validos)
            
            homogeneity_std = 100 * (1 - (std_dose / mean_dose)) if mean_dose > 0 else 0
            homogeneity_range = 100 * (1 - ((max_dose - min_dose) / mean_dose)) if mean_dose > 0 else 0
        
        # Calcular dosis bruta (sin restar fondo)
        raw_dose = mean_dose + background
        
        return {
            "x": x,
            "y": y,
            "r": r,
            "mean_dose": raw_dose,  # Dosis bruta
            "std": std_dose,
            "min": min_dose,
            "max": max_dose,
            "homo_std": homogeneity_std,
            "homo_range": homogeneity_range
        }

    def mapa_3d_circulo(self, x, y, r):
        if self.pil_img is None:
            print("⚠️ No hay imagen cargada.")
            return

        img_rgb = np.array(self.pil_img)
        self.procesar_circulo(img_rgb, x, y, r, graficar=True)  

    def on_circle_click(self, event):
        if not self.detected_circles or self.pil_img is None:
            return

        x_click = self.canvas.canvasx(event.x)
        y_click = self.canvas.canvasy(event.y)

        img_rgb = np.array(self.pil_img)

        for (x, y, r) in self.detected_circles:
            dist = np.sqrt((x - x_click)**2 + (y - y_click)**2)
            if dist < r:
                print(f"🖱️ Clic derecho sobre círculo en ({x}, {y}) → mostrando mapa 3D")
                self.procesar_circulo(img_rgb, x, y, r, graficar=True)
                return

    def create_subcircles_window(self, circle_data, dose_map):
        """Crea una ventana para mostrar los subcírculos en patrón 2-4-4-2"""
        if self.subcircles_window and self.subcircles_window.winfo_exists():
            self.subcircles_window.destroy()
        
        # Limpiar los datos de subcírculos para que sean independientes para cada figura
        self.subcircles_data = []
        self.subcircle_positions = []
        self.subcircle_patches = []
        self.subcircle_labels = []
            
        self.subcircles_window = tk.Toplevel(self.root)
        self.subcircles_window.title(f"Análisis de Pocillos - {circle_data['id']}")
        self.subcircles_window.geometry("800x600")  # Reducido de 900x700 a 800x600
        
        # Frame principal
        main_frame = tk.Frame(self.subcircles_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame para el gráfico (izquierda)
        graph_frame = tk.Frame(main_frame)
        graph_frame.pack(side="left", fill="both", expand=True)
        
        # Frame para la información (derecha)
        info_frame = tk.Frame(main_frame)
        info_frame.pack(side="right", fill="y", padx=10)
        
        # Crear figura para el gráfico 2D
        self.subcircles_fig = plt.Figure(figsize=(6, 6), dpi=100)  # Reducido de 8x8 a 6x6
        self.subcircles_ax = self.subcircles_fig.add_subplot(111)
        
        # Crear canvas para mostrar la figura
        self.subcircles_canvas = FigureCanvasTkAgg(self.subcircles_fig, master=graph_frame)
        self.subcircles_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Dibujar el mapa de dosis como fondo
        im = self.subcircles_ax.imshow(dose_map, cmap='viridis', origin='lower')
        self.subcircles_fig.colorbar(im, ax=self.subcircles_ax, label='Dosis (Gy)')
        
        # Calcular el centro y radio del círculo grande
        h, w = dose_map.shape
        centro_x = w // 2
        centro_y = h // 2
        radio_grande = min(w, h) // 2 - 3  # Un poco menor que el tamaño del mapa
        
        # Dibujar círculo grande
        circle_grande = plt.Circle((centro_x, centro_y), radio_grande, fill=False, color='black', linewidth=2)
        self.subcircles_ax.add_patch(circle_grande)
        
        # Añadir etiqueta con el ID del círculo
        self.subcircle_positions = []
        self.subcircle_patches = []
        self.subcircle_labels = []

        # Radio de subcírculo: 1/4.2 del radio grande
        radio_pequeno = radio_grande / 4.2
        diametro = 2 * radio_pequeno

        # Distancias horizontales (entre centros)
        dx_2 = diametro * 1.1  # separación entre 2 círculos
        dx_4 = diametro * 1.1  # separación entre 4 círculos
        dy = diametro * 1.1   # separación entre filas

        # Fila 1 (superior) - 2 círculos
        y1 = centro_y - 1.5 * dy
        for i in range(2):
            x = centro_x - dx_2 / 2 + i * dx_2
            self.subcircle_positions.append((int(x), int(y1)))

        # Fila 2 - 4 círculos
        y2 = centro_y - 0.5 * dy
        for i in range(4):
            x = centro_x - 1.5 * dx_4 + i * dx_4
            self.subcircle_positions.append((int(x), int(y2)))

        # Fila 3 - 4 círculos
        y3 = centro_y + 0.5 * dy
        for i in range(4):
            x = centro_x - 1.5 * dx_4 + i * dx_4
            self.subcircle_positions.append((int(x), int(y3)))

        # Fila 4 (inferior) - 2 círculos
        y4 = centro_y + 1.5 * dy
        for i in range(2):
            x = centro_x - dx_2 / 2 + i * dx_2
            self.subcircle_positions.append((int(x), int(y4)))
        
        for i, (cx, cy) in enumerate(self.subcircle_positions):
            # Crear una máscara para el subcírculo
            mask = np.zeros_like(dose_map)
            y_indices, x_indices = np.ogrid[:dose_map.shape[0], :dose_map.shape[1]]
            dist_from_center = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
            mask[dist_from_center <= radio_pequeno] = 1
            
            # Aplicar la máscara al mapa de dosis
            masked_dose = dose_map * mask
            
            # Calcular estadísticas
            valores_validos = masked_dose[masked_dose > 0]
            if len(valores_validos) > 0:
                mean_dose = np.mean(valores_validos)
                std_dose = np.std(valores_validos, ddof=1)
                
                # Guardar datos del subcírculo
                self.subcircles_data.append({
                    "id": i + 1,
                    "x": cx,
                    "y": cy,
                    "r": radio_pequeno,
                    "mean_dose": mean_dose,
                    "std_dose": std_dose
                })
                
                # Dibujar subcírculo
                circle = plt.Circle((cx, cy), radio_pequeno, fill=True, color='#B2FF66', 
                                   edgecolor='darkgreen', linewidth=1, alpha=0.7)
                self.subcircles_ax.add_patch(circle)
                self.subcircle_patches.append(circle)
                
                # Añadir etiqueta con número
                label = self.subcircles_ax.text(cx, cy, str(i+1), ha='center', va='center', 
                                              color='black', fontweight='bold', fontsize=10)
                self.subcircle_labels.append(label)
        
        # Configurar el gráfico
        self.subcircles_ax.set_title(f"Análisis de Pocillos - {circle_data['id']}")
        self.subcircles_ax.set_xlabel("X (píxeles)")
        self.subcircles_ax.set_ylabel("Y (píxeles)")
        
        # Crear frame para mostrar información del círculo principal
        circle_info_frame = tk.LabelFrame(info_frame, text="Información del Círculo", padx=5, pady=5)
        circle_info_frame.pack(fill="x", pady=10)
        
        tk.Label(circle_info_frame, text=f"ID: {circle_data['id']}").pack(anchor="w")
        tk.Label(circle_info_frame, text=f"Área: {circle_data['area_name']}").pack(anchor="w")
        tk.Label(circle_info_frame, text=f"Fondo: {circle_data['background']:.4f} Gy").pack(anchor="w")
        
        # Crear frame para mostrar información de los subcírculos
        subcircle_info_frame = tk.LabelFrame(info_frame, text="Información de Pocillos", padx=5, pady=5)
        subcircle_info_frame.pack(fill="both", expand=True, pady=10)
        
        # Crear un widget de texto para mostrar la información de los subcírculos
        self.subcircle_info_text = tk.Text(subcircle_info_frame, width=35, height=20, font=("Consolas", 9))
        self.subcircle_info_text.pack(fill="both", expand=True)
        
        # Mostrar información de todos los subcírculos
        self.subcircle_info_text.insert(tk.END, "Pocillo  Dosis (Gy)  σ (Gy)  Error\n")
        self.subcircle_info_text.insert(tk.END, "-" * 40 + "\n")
        
        for data in self.subcircles_data:
            error = (data["std_dose"] / data["mean_dose"]) if data["mean_dose"] > 0 else 0
            self.subcircle_info_text.insert(tk.END, 
                f"{data['id']:2d}      {data['mean_dose']:.4f}   {data['std_dose']:.4f}   {error:.4f}\n")
        
        # Calcular estadísticas globales
        if self.subcircles_data:
            doses = [data["mean_dose"] for data in self.subcircles_data]
            mean_global = np.mean(doses)
            std_global = np.std(doses, ddof=1) if len(doses) > 1 else 0
            cv_global = (std_global / mean_global) if mean_global > 0 else 0
            
            self.subcircle_info_text.insert(tk.END, "\nEstadísticas globales:\n")
            self.subcircle_info_text.insert(tk.END, f"Dosis promedio: {mean_global:.4f} Gy\n")
            self.subcircle_info_text.insert(tk.END, f"Desviación: {std_global:.4f} Gy\n")
            self.subcircle_info_text.insert(tk.END, f"CV: {cv_global:.4f}\n")
        
        # Botón para guardar resultados
        save_button = tk.Button(info_frame, text="Guardar Resultados", 
                               command=self.save_subcircles_to_file,
                               bg="#3E4A61", fg="white", font=("Segoe UI", 10),
                               relief="flat", bd=0, padx=10, pady=6)
        save_button.pack(pady=10)
        
        # Actualizar el canvas
        self.subcircles_fig.tight_layout()
        self.subcircles_canvas.draw()
        
        # Conectar evento de clic
        self.subcircles_canvas.mpl_connect('button_press_event', self.on_subcircle_click)

    def on_subcircle_click(self, event):
        """Maneja el clic en un subcírculo"""
        if not hasattr(self, 'subcircle_positions') or not self.subcircle_positions:
            return
            
        # Verificar si el clic fue dentro de algún subcírculo
        for i, (cx, cy) in enumerate(self.subcircle_positions):
            if i >= len(self.subcircles_data):
                continue
                
            dist = np.sqrt((event.xdata - cx)**2 + (event.ydata - cy)**2)
            if dist <= self.subcircles_data[i]["r"]:
                # Resaltar el subcírculo seleccionado
                for j, patch in enumerate(self.subcircle_patches):
                    if j == i:
                        patch.set_color('red')
                        patch.set_alpha(0.9)
                    else:
                        patch.set_color('#B2FF66')
                        patch.set_alpha(0.7)
                
                # Actualizar el canvas
                self.subcircles_canvas.draw()
                
                # Mostrar información del subcírculo seleccionado directamente en la pantalla
                data = self.subcircles_data[i]
                error = (data["std_dose"] / data["mean_dose"]) if data["mean_dose"] > 0 else 0
                
                # Resaltar la línea correspondiente en el texto
                self.subcircle_info_text.tag_remove("highlight", "1.0", tk.END)
                
                # Encontrar la línea correspondiente al subcírculo
                line_start = f"{i+3}.0"  # +3 porque hay dos líneas de encabezado
                line_end = f"{i+3}.end"
                
                self.subcircle_info_text.tag_add("highlight", line_start, line_end)
                self.subcircle_info_text.tag_config("highlight", background="yellow")
                self.subcircle_info_text.see(line_start)
                
                # Guardar el subcírculo seleccionado
                self.selected_subcircle = i
                
                break

    def procesar_circulo(self, img_rgb, x, y, r, step=1, factor_radio=0.9, graficar=False):
        h, w, _ = img_rgb.shape
        radio_seguro = int(r * factor_radio)

        # Máscara circular
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), radio_seguro, 255, -1)
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        # Recorte
        x1 = max(0, x - radio_seguro)
        y1 = max(0, y - radio_seguro)
        x2 = min(w, x + radio_seguro)
        y2 = min(h, y + radio_seguro)
        cut = masked_img[y1:y2, x1:x2]

        # Obtener valor de fondo
        area_idx = self.find_radiochromic_area(x, y)
        background = 0.0
        circle_id = "Desconocido"
        area_name = "Desconocida"
        
        if area_idx is not None:
            area = self.radiochromic_areas[area_idx]
            area_name = area["name"]
            background = self.get_area_background(area)
            
            # Identificar el círculo para obtener su ID
            for i, circle in enumerate(area["circles"]):
                if circle["x"] == x and circle["y"] == y:
                    # Ordenar círculos según el esquema proporcionado
                    circles_with_positions = []
                    for j, c in enumerate(area["circles"]):
                        circles_with_positions.append((c, j))
                    
                    sorted_circles = self.sort_circles_by_position(circles_with_positions, area["coords"])
                    
                    # Encontrar la posición del círculo actual en la lista ordenada
                    circle_idx = -1
                    for j, (c, _) in enumerate(sorted_circles):
                        if c["x"] == x and c["y"] == y:
                            circle_idx = j
                            break
                    
                    # Asignar letra según el esquema proporcionado
                    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
                    circle_id = f"{area['name']}_{letters[circle_idx % len(letters)]}"
                    
                    # Marcar círculos manuales
                    if "manual" in circle and circle["manual"]:
                        circle_id += " (M)"
                    
                    break

        # Mapa por bloques
        dose_map = []
        for j in range(0, cut.shape[0], step):
            row = []
            for i in range(0, cut.shape[1], step):
                block = cut[j:j+step, i:i+step]
                if block.size == 0 or np.all(block == 0):
                    row.append(0)
                    continue
                dose = self.calcular_dosis_promedio(block)
                # Restar fondo
                dose = max(0, dose - background)  # Asegurar que la dosis no sea negativa
                row.append(dose)
            if row:  # Asegurarse de que la fila no esté vacía
                dose_map.append(row)

        if not dose_map:
            return {
                "x": x,
                "y": y,
                "r": radio_seguro,
                "mean_dose": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "homo_std": 0,
                "homo_range": 0
            }

        dose_map = np.array(dose_map)

        # Estadísticas de homogeneidad
        valores_validos = dose_map[dose_map > 0]
        if len(valores_validos) == 0:
            mean_dose = 0
            std_dose = 0
            min_dose = 0
            max_dose = 0
            homogeneity_std = 0
            homogeneity_range = 0
        else:
            mean_dose = np.mean(valores_validos)
            std_dose = np.std(valores_validos, ddof=1)
            min_dose = np.min(valores_validos)
            max_dose = np.max(valores_validos)
            
            homogeneity_std = 100 * (1 - (std_dose / mean_dose)) if mean_dose > 0 else 0
            homogeneity_range = 100 * (1 - ((max_dose - min_dose) / mean_dose)) if mean_dose > 0 else 0

        if graficar and len(valores_validos) > 0:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.cm as cm

            # Guardar datos del círculo actual para uso posterior
            self.current_circle_data = {
                "id": circle_id,
                "area_name": area_name,
                "background": background,
                "mean_dose": mean_dose,
                "std_dose": std_dose,
                "min_dose": min_dose,
                "max_dose": max_dose
            }

            # Crear figura con un solo subplot
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            X = np.arange(dose_map.shape[1])
            Y = np.arange(dose_map.shape[0])
            X, Y = np.meshgrid(X, Y)
            
            # Establecer límites para mostrar desde cero hasta el máximo (para ver la "caída")
            zmin = 0  # Mostrar desde cero para ver la "caída"
            zmax = max_dose  # Usar el valor máximo real
            ax.set_zlim(zmin, zmax)
            
            # Graficar superficie con todos los valores para mostrar la caída
            surf = ax.plot_surface(X, Y, dose_map, cmap=cm.viridis)
            
            # Posición personalizada: [izquierda, abajo, ancho, alto]
            cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.5])  # mueve la barra más a la derecha
           
            fig.colorbar(surf, cax=cbar_ax, label='Dosis neta (Gy)')
            ax.set_title(f"Plot 3D de {circle_id}")
            ax.set_xlabel("X (bloques)")
            ax.set_ylabel("Y (bloques)")
            ax.set_zlabel("Dosis (Gy)")

            info_text = (
                f"Promedio: {mean_dose:.3f} Gy\n"
                f"Desviación estándar: {std_dose:.3f} Gy\n"
                f"Error: {(std_dose/mean_dose):.4f}\n"
                f"Mínimo: {min_dose:.3f} Gy\n"
                f"Máximo: {max_dose:.3f} Gy\n"
                f"Fondo: {background:.3f} Gy\n"
                f"Homo. σ/μ: {homogeneity_std:.1f}%\n"
                f"Homo. rango: {homogeneity_range:.1f}%"
            )

            # Ajustar layout del gráfico para dejar espacio
            plt.subplots_adjust(right=0.8, bottom=0.2)

            # Mostrar texto en una posición fija debajo de la colorbar
            fig.text(0.1, 0.05, info_text,
                ha='left', va='bottom',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round,pad=0.5'))
                
            # Añadir botón para mostrar análisis de pocillos
            button_ax = fig.add_axes([0.4, 0.05, 0.2, 0.05])
            button = plt.Button(button_ax, 'Dosis Pocillos', color='lightgoldenrodyellow', hovercolor='0.975')
            
            def on_button_click(event):
                # Crear ventana con subcírculos interactivos
                self.create_subcircles_window(self.current_circle_data, dose_map)
                
            button.on_clicked(on_button_click)
            
            plt.show()    

        # Calcular dosis bruta (sin restar fondo)
        raw_dose = max(0, mean_dose + background)  # Asegurar que la dosis no sea negativa

        return {
            "x": x,
            "y": y,
            "r": radio_seguro,
            "mean_dose": raw_dose,  # Dosis bruta
            "std": std_dose,
            "min": min_dose,
            "max": max_dose,
            "homo_std": homogeneity_std,
            "homo_range": homogeneity_range
        }


if __name__ == "__main__":
    root = tk.Tk()
    app = DoseApp(root)
    root.mainloop()
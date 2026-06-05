import os
import time
import cv2
import numpy as np
import json
import traceback

try:
    import supervision as sv
    from ultralytics import YOLO
    import customtkinter as ctk
    from PIL import Image, ImageTk
except ImportError as e:
    print(f" ERROR: {e}. Instala: pip install customtkinter Pillow")
    exit()

# 1. CONFIGURACIONES GLOBALES
PESOS_RIESGO = {
    "casco": 35,
    "chaleco": 35,
    "lentes": 10,
    "guantes": 10,
    "botas": 10
}
FRAMES_REQUERIDOS = 60

# 2. HERRAMIENTA DE CALIBRACIÓN INCORPORADA
def lanzar_calibrador(id_camara):
    """Abre la herramienta geométrica para que el usuario defina la zona con 4 clics."""
    puntos = []
    archivo_salida = f"zona_cam_{id_camara}.json"
    
    def capturar_clic(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append([x, y])
            print(f" Punto {len(puntos)}/4 en Cam {id_camara}: [{x}, {y}]")
            if len(puntos) == 4:
                with open(archivo_salida, "w") as f:
                    json.dump(puntos, f)
                print(f" ¡Zona guardada en {archivo_salida}!")

    cap = cv2.VideoCapture(id_camara)
    if not cap.isOpened():
        print(f" Error: No se detecta la cámara {id_camara}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    nombre_ventana = f"CALIBRACION - Camara {id_camara} (Haga 4 clics y presione ESC)"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 1280, 720)
    cv2.setMouseCallback(nombre_ventana, capturar_clic)

    print(f"\n[CALIBRACIÓN] -> Abriendo cámara {id_camara}. Haga 4 clics para enmarcar el área.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Efecto espejo en la calibración SOLO para la laptop (cam 0)
        if id_camara == 0:
            frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (1280, 720))
        
        for pt in puntos:
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
        if len(puntos) > 1:
            for i in range(len(puntos) - 1):
                cv2.line(frame, tuple(puntos[i]), tuple(puntos[i+1]), (0, 0, 255), 2)
        if len(puntos) == 4:
            cv2.line(frame, tuple(puntos[3]), tuple(puntos[0]), (0, 0, 255), 2)
            cv2.putText(frame, "ZONA GUARDADA - PRESIONE ESC PARA SALIR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow(nombre_ventana, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # Salir con ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# 3. MOTOR LÓGICO (IA y Geometría)
def cargar_zona(id_camara, width, height):
    archivo = f"zona_cam_{id_camara}.json"
    try:
        if os.path.exists(archivo):
            with open(archivo, "r") as f:
                puntos = np.array(json.load(f))
                escala_x = width / 1280.0
                escala_y = height / 720.0
                puntos[:, 0] = (puntos[:, 0] * escala_x).astype(int)
                puntos[:, 1] = (puntos[:, 1] * escala_y).astype(int)
                return puntos
    except Exception:
        pass
    return np.array([[5, 5], [width - 5, 5], [width - 5, height - 5], [5, height - 5]])

def hay_interseccion(box_persona, box_equipo):
    xA = max(box_persona[0], box_equipo[0])
    yA = max(box_persona[1], box_equipo[1])
    xB = min(box_persona[2], box_equipo[2])
    yB = min(box_persona[3], box_equipo[3])
    return max(0, xB - xA) * max(0, yB - yA) > 0

def procesar_frame(frame, model, zone, box_annotator, label_annotator, zone_annotator, nombre_camara, estado):
    results = model(frame, device=0, conf=0.45, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    if len(detections) > 0:
        is_inside = zone.trigger(detections=detections)
    else:
        is_inside = []

    personas_boxes = []
    equipos_boxes = []

    for is_in, class_id, xyxy in zip(is_inside, detections.class_id, detections.xyxy):
        if not is_in: continue
        nombre_clase = model.names[class_id].lower()
        if nombre_clase == "person":
            personas_boxes.append({"box": xyxy, "riesgo": 0})
        else:
            equipos_boxes.append((nombre_clase, xyxy))

    infracciones_actuales = 0
    riesgo_maximo_zona = 0

    for p_info in personas_boxes:
        epp_encontrado = {"helmet": False, "vest": False, "goggle": False, "gloves": False, "boots": False}
        faltas_explicitas = {"no_helmet": False, "no_goggle": False, "no_gloves": False, "no_boots": False, "none": False}

        for equipo_nombre, e_box in equipos_boxes:
            if hay_interseccion(p_info["box"], e_box):
                if equipo_nombre in epp_encontrado: epp_encontrado[equipo_nombre] = True
                elif equipo_nombre in faltas_explicitas: faltas_explicitas[equipo_nombre] = True

        riesgo = 100 if faltas_explicitas["none"] else sum([
            PESOS_RIESGO["casco"] if faltas_explicitas["no_helmet"] or not epp_encontrado["helmet"] else 0,
            PESOS_RIESGO["chaleco"] if not epp_encontrado["vest"] else 0,
            PESOS_RIESGO["lentes"] if faltas_explicitas["no_goggle"] or not epp_encontrado["goggle"] else 0,
            PESOS_RIESGO["guantes"] if faltas_explicitas["no_gloves"] or not epp_encontrado["gloves"] else 0,
            PESOS_RIESGO["botas"] if faltas_explicitas["no_boots"] or not epp_encontrado["boots"] else 0
        ])

        riesgo = min(riesgo, 100)
        p_info["riesgo"] = riesgo

        if riesgo > 0:
            infracciones_actuales += 1
            riesgo_maximo_zona = max(riesgo_maximo_zona, riesgo)

    if infracciones_actuales > 0:
        estado["consecutivas"] = min(estado["consecutivas"] + 1, FRAMES_REQUERIDOS)
    else:
        estado["consecutivas"] = max(estado["consecutivas"] - 2, 0)

    if estado["consecutivas"] == FRAMES_REQUERIDOS and not estado["cooldown"]:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        estado["pendiente_guardar"] = stamp
        estado["cooldown"] = True

    if estado["consecutivas"] == 0:
        estado["cooldown"] = False

    labels = [f"{model.names[c]} {conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    for p_info in personas_boxes:
        px1, py1, px2, py2 = map(int, p_info["box"])
        r = p_info["riesgo"]
        color = (0, 255, 0) if r == 0 else (0, 255, 255) if r <= 35 else (0, 0, 255)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 4)
        cv2.putText(frame, f"{r}% Riesgo", (px1, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    progreso = min(100, int((estado["consecutivas"] / FRAMES_REQUERIDOS) * 100))
    if riesgo_maximo_zona == 0:
        cv2.putText(frame, f"{nombre_camara} - SEGURO", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"{nombre_camara} - PELIGRO: {riesgo_maximo_zona}%", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Alerta Confirmando: {progreso}%", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if estado.get("pendiente_guardar"):
        stamp = estado.pop("pendiente_guardar")
        cv2.imwrite(f"Alertas/INFRACCION_{nombre_camara.replace(' ', '_')}_{stamp}.jpg", frame)

    return frame

# 4. INTERFAZ GRÁFICA UNIFICADA
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SentinelaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        self.cap0 = None
        self.cap1 = None
        self.model = None
        self.is_running = False 
        self.after_id = None    
        self.ancho_previo = 0 # Inicializar el detector de tamaño de ventana

        if not os.path.exists("Alertas"): 
            os.makedirs("Alertas")

        self.mostrar_menu_inicio()

    def limpiar_ventana(self):
        for widget in self.winfo_children():
            widget.destroy()

    def mostrar_menu_inicio(self):
        self.limpiar_ventana()
        self.is_running = False
        
        self.state('normal')
        self.title("Sentinela 360 - Menú Principal")
        self.geometry("750x550")
        
        self.marco_central = ctk.CTkFrame(self, corner_radius=20, fg_color="#1e1e24")
        self.marco_central.pack(expand=True, fill="both", padx=40, pady=40)

        ctk.CTkLabel(self.marco_central, text=" SENTINELA 360", font=("Roboto", 34, "bold"), text_color="#3498db").pack(pady=(30, 5))
        ctk.CTkLabel(self.marco_central, text="Sistema de Inteligencia Artificial para Seguridad Industrial", font=("Roboto", 14), text_color="gray").pack(pady=(0, 30))

        marco_botones = ctk.CTkFrame(self.marco_central, fg_color="transparent")
        marco_botones.pack(pady=10)

        btn_individual = ctk.CTkButton(marco_botones, text="▶ Monitoreo Individual (1 Cámara)", font=("Roboto", 15, "bold"), height=50, width=300, corner_radius=10, command=lambda: self.iniciar_sistema(modo="individual"))
        btn_individual.pack(pady=10)

        btn_dual = ctk.CTkButton(marco_botones, text="▶ Monitoreo Dual (2 Cámaras)", font=("Roboto", 15, "bold"), height=50, width=300, corner_radius=10, fg_color="#27ae60", hover_color="#219150", command=lambda: self.iniciar_sistema(modo="dual"))
        btn_dual.pack(pady=10)

        ctk.CTkLabel(self.marco_central, text="--- Herramientas Topográficas ---", font=("Roboto", 12), text_color="gray").pack(pady=(20, 10))

        frame_calibrar = ctk.CTkFrame(self.marco_central, fg_color="transparent")
        frame_calibrar.pack()
        
        btn_cal0 = ctk.CTkButton(frame_calibrar, text=" Calibrar Cam 0", width=140, height=35, corner_radius=8, fg_color="#f39c12", hover_color="#d68910", command=lambda: lanzar_calibrador(0))
        btn_cal0.pack(side="left", padx=10)

        btn_cal1 = ctk.CTkButton(frame_calibrar, text=" Calibrar Cam 1", width=140, height=35, corner_radius=8, fg_color="#f39c12", hover_color="#d68910", command=lambda: lanzar_calibrador(1))
        btn_cal1.pack(side="right", padx=10)

    def iniciar_sistema(self, modo):
        self.modo_actual = modo
        self.limpiar_ventana()

        lbl_carga = ctk.CTkLabel(self, text=" Inicializando Motor Neuronal y Cámaras...\nPor favor espere.", font=("Roboto", 18, "bold"), text_color="#f1c40f")
        lbl_carga.pack(expand=True)
        self.update()

        try:
            if self.model is None:
                self.model = YOLO("best.pt")
                
            self.cap0 = cv2.VideoCapture(0)
            self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if self.modo_actual == "dual":
                self.cap1 = cv2.VideoCapture(1)
                self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not self.cap0.isOpened():
                raise Exception("Cámara principal (0) desconectada.")
            if self.modo_actual == "dual" and not self.cap1.isOpened():
                raise Exception("Cámara secundaria (1) desconectada.")

        except Exception as e:
            lbl_carga.configure(text=f" ERROR: {e}\nRegresando al menú...", text_color="red")
            self.update()
            time.sleep(3)
            self.volver_al_menu()
            return

        self.is_running = True
        self.construir_dashboard()

    def volver_al_menu(self):
        print("[SISTEMA] -> Regresando al menú principal...")
        self.is_running = False 
        
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
            
        if self.cap0: self.cap0.release()
        if self.cap1: self.cap1.release()
        
        self.mostrar_menu_inicio()

    def construir_dashboard(self):
        self.limpiar_ventana()
        self.ancho_previo = 0 # Resetea el tracker de ventana
        
        if self.modo_actual == "individual":
            self.title("Sentinela 360 - Comando Individual")
            self.geometry("1100x750")
            self.res_w, self.res_h = 960, 540 
        else:
            self.title("Sentinela 360 - Comando DUAL")
            self.geometry("1350x700")
            self.res_w, self.res_h = 640, 360 

        # Intentar arrancar maximizado por comodidad
        try:
            self.state('zoomed')
        except:
            pass

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        zona0 = sv.PolygonZone(polygon=cargar_zona(0, self.res_w, self.res_h), triggering_anchors=[sv.Position.CENTER])
        self.annotator_cam0 = sv.PolygonZoneAnnotator(zone=zona0, color=sv.Color.BLUE, thickness=2)
        self.estado_cam0 = {"consecutivas": 0, "cooldown": False, "zona": zona0}

        if self.modo_actual == "dual":
            zona1 = sv.PolygonZone(polygon=cargar_zona(1, self.res_w, self.res_h), triggering_anchors=[sv.Position.CENTER])
            self.annotator_cam1 = sv.PolygonZoneAnnotator(zone=zona1, color=sv.Color.RED, thickness=2)
            self.estado_cam1 = {"consecutivas": 0, "cooldown": False, "zona": zona1}

        marco_header = ctk.CTkFrame(self, fg_color="#1e1e24", corner_radius=10)
        marco_header.pack(fill="x", padx=20, pady=(15, 5))
        
        titulo_texto = "SECTOR PRINCIPAL (CAM 0)" if self.modo_actual == "individual" else "SISTEMA MULTICÁMARA (DUAL)"
        ctk.CTkLabel(marco_header, text=f" MONITOREO ACTIVO: {titulo_texto}", font=("Roboto", 20, "bold")).pack(pady=10)

        # Contenedor vital para el Responsive
        self.marco_camaras = ctk.CTkFrame(self, fg_color="transparent")
        self.marco_camaras.pack(fill="both", expand=True, padx=20, pady=5)

        if self.modo_actual == "individual":
            self.lbl_cam0 = ctk.CTkLabel(self.marco_camaras, text="")
            self.lbl_cam0.pack(expand=True)
        else:
            self.lbl_cam0 = ctk.CTkLabel(self.marco_camaras, text="")
            self.lbl_cam0.pack(side="left", expand=True)
            self.lbl_cam1 = ctk.CTkLabel(self.marco_camaras, text="")
            self.lbl_cam1.pack(side="right", expand=True)

        marco_controles = ctk.CTkFrame(self, height=60, corner_radius=10, fg_color="#1e1e24")
        marco_controles.pack(fill="x", padx=20, pady=(5, 15))

        ctk.CTkLabel(marco_controles, text=" PROCESANDO IA EN TIEMPO REAL", font=("Roboto", 14, "bold"), text_color="#2ecc71").pack(side="left", padx=20, pady=15)
        ctk.CTkButton(marco_controles, text=" Volver al Menú", font=("Roboto", 14, "bold"), fg_color="#e74c3c", hover_color="#c0392b", corner_radius=8, command=self.volver_al_menu).pack(side="right", padx=20, pady=15)

        self.actualizar_video()

    def cv2_a_tkinter(self, frame):
        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_img)
        return ImageTk.PhotoImage(image=pil_img)

    def actualizar_video(self):
        if not self.is_running:
            return

        try:
            ancho_actual = self.marco_camaras.winfo_width()
            
            if ancho_actual > 100 and abs(self.ancho_previo - ancho_actual) > 10:
                self.ancho_previo = ancho_actual
                
                if self.modo_actual == "individual":
                    self.res_w = int(ancho_actual * 0.95) 
                    self.res_h = int(self.res_w * 9 / 16) 
                else:
                    self.res_w = int(ancho_actual * 0.48)
                    self.res_h = int(self.res_w * 9 / 16)
          
                zona0 = sv.PolygonZone(polygon=cargar_zona(0, self.res_w, self.res_h), triggering_anchors=[sv.Position.CENTER])
                self.annotator_cam0 = sv.PolygonZoneAnnotator(zone=zona0, color=sv.Color.BLUE, thickness=2)
                self.estado_cam0["zona"] = zona0

                if self.modo_actual == "dual":
                    zona1 = sv.PolygonZone(polygon=cargar_zona(1, self.res_w, self.res_h), triggering_anchors=[sv.Position.CENTER])
                    self.annotator_cam1 = sv.PolygonZoneAnnotator(zone=zona1, color=sv.Color.RED, thickness=2)
                    self.estado_cam1["zona"] = zona1


            if self.cap0 and self.cap0.isOpened():
                ret1, frame1 = self.cap0.read()
                if ret1:
                    frame1 = cv2.flip(frame1, 1)
                    frame1 = cv2.resize(frame1, (self.res_w, self.res_h))
                    frame1 = procesar_frame(frame1, self.model, self.estado_cam0["zona"], self.box_annotator, self.label_annotator, self.annotator_cam0, "CAM 0", self.estado_cam0)
                    img1 = self.cv2_a_tkinter(frame1)
                    self.lbl_cam0.configure(image=img1)
                    self.lbl_cam0.image = img1 

            if self.modo_actual == "dual" and self.cap1 and self.cap1.isOpened():
                ret2, frame2 = self.cap1.read()
                if ret2:
                    frame2 = cv2.resize(frame2, (self.res_w, self.res_h))
                    frame2 = procesar_frame(frame2, self.model, self.estado_cam1["zona"], self.box_annotator, self.label_annotator, self.annotator_cam1, "CAM 1", self.estado_cam1)
                    img2 = self.cv2_a_tkinter(frame2)
                    self.lbl_cam1.configure(image=img2)
                    self.lbl_cam1.image = img2

            self.after_id = self.after(15, self.actualizar_video)
        except Exception as e:
            print(f" Error en video: {e}")
            traceback.print_exc()

    def cerrar_aplicacion(self):
        print("[SENTINELA 360] -> Apagando sistema completamente...")
        self.is_running = False
        if self.cap0: self.cap0.release()
        if self.cap1: self.cap1.release()
        self.destroy()
        exit()

if __name__ == "__main__":
    app = SentinelaApp()
    app.mainloop()
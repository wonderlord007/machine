import os
import time
import cv2
import numpy as np
import json
import supervision as sv
from ultralytics import YOLO

def cargar_zona(width, height):
    """Carga el polígono guardado o usa pantalla completa por defecto."""
    if os.path.exists("zona_config.json"):
        with open("zona_config.json", "r") as f:
            return np.array(json.load(f))
    return np.array([[10, 10], [width - 10, 10], [width - 10, height - 10], [10, height - 10]])

def main():
    print("[SENTINELA 360] -> Inicializando Inteligencia Artificial...")
    model = YOLO("best.pt") 

    # 1 = Cámara Inalámbrica (Tu Xiaomi) | 0 = Webcam HP
    video_path = 1  
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error CRÍTICO: No se conectó la cámara {video_path}.")
        return

    # Solicitar resolución HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    polygon = cargar_zona(width, height)
    zone = sv.PolygonZone(polygon=polygon)
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2, text_thickness=2)

    consecutive_infractions = 0
    FRAMES_REQUERIDOS = 60  
    alert_cooldown = False  

    # Sistema de Auditoría
    if not os.path.exists("Alertas"): 
        os.makedirs("Alertas")
    
    archivo_log = "Alertas/registro_auditoria.csv"
    if not os.path.exists(archivo_log):
        with open(archivo_log, "w") as log:
            log.write("Fecha,Hora,Evento,Archivo_Evidencia\n")

    # 🛠️ FORZAR TAMAÑO DE VENTANA
    cv2.namedWindow("Sentinela 360 - Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sentinela 360 - Dashboard", 1280, 720)

    print("[SENTINELA 360] -> Sistema en ejecución. Presione ESC para finalizar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(10)
            continue  

        # Inferencia (conf=0.15 para no perder detalles por la transmisión WiFi)
        results = model(frame, device=0, conf=0.15)[0]
        detections = sv.Detections.from_ultralytics(results)
        is_inside = zone.trigger(detections=detections)

        # Segmentación Espacial
        personas_boxes = []
        equipos_boxes = []

        for is_in, class_id, xyxy in zip(is_inside, detections.class_id, detections.xyxy):
            if not is_in: continue
            
            nombre_clase = model.names[class_id].lower()
            if nombre_clase == "person":
                personas_boxes.append(xyxy)
            else:
                equipos_boxes.append((nombre_clase, xyxy))

        infracciones_actuales = 0

        # Auditoría Individual Geométrica
        for p_box in personas_boxes:
            px1, py1, px2, py2 = p_box
            tiene_casco = False
            tiene_chaleco = False
            falta_explicita = False

            for equipo_nombre, e_box in equipos_boxes:
                ex1, ey1, ex2, ey2 = e_box
                centro_x, centro_y = (ex1 + ex2) / 2, (ey1 + ey2) / 2

                if px1 <= centro_x <= px2 and py1 <= centro_y <= py2:
                    if equipo_nombre == "helmet": tiene_casco = True
                    elif equipo_nombre == "vest": tiene_chaleco = True
                    elif equipo_nombre in ["no_helmet", "no_goggle", "no_gloves", "no_boots", "none"]:
                        falta_explicita = True

            if falta_explicita or not tiene_casco or not tiene_chaleco:
                infracciones_actuales += 1

        # Filtro Temporal Antiruido
        if infracciones_actuales > 0:
            consecutive_infractions += 1
        else:
            if consecutive_infractions > 0: consecutive_infractions -= 1 

        # Renderizado Visual de Anotaciones
        labels = [f"{model.names[class_id]} {conf:.2f}" for class_id, conf in zip(detections.class_id, detections.confidence)]
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = zone_annotator.annotate(scene=frame)

        # Interfaz de Usuario (HUD)
        progreso = min(100, int((consecutive_infractions / FRAMES_REQUERIDOS) * 100))
        
        if infracciones_actuales == 0:
            estado_texto = "ESTADO: SEGURO"
            color_estado = (0, 255, 0) 
        else:
            estado_texto = "ESTADO: PELIGRO (Infractor detectado)"
            color_estado = (0, 0, 255) 

        cv2.putText(frame, estado_texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_estado, 2)
        cv2.putText(frame, f"Nivel de Riesgo: {progreso}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Guardado de Evidencia y Registro
        if consecutive_infractions >= FRAMES_REQUERIDOS and not alert_cooldown:
            fecha_actual = time.strftime("%Y-%m-%d")
            hora_actual = time.strftime("%H:%M:%S")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            ruta_archivo = f"Alertas/INFRACCION_{stamp}.jpg"
            
            cv2.imwrite(ruta_archivo, frame)
            
            with open(archivo_log, "a") as log:
                log.write(f"{fecha_actual},{hora_actual},Infraccion Detectada,{ruta_archivo}\n")
                
            print(f"⚠️ Evidencia asegurada y registrada: {stamp}")
            alert_cooldown = True  

        if consecutive_infractions == 0:
            alert_cooldown = False

        # Mostrar interfaz
        cv2.imshow("Sentinela 360 - Dashboard", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]: 
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("[SENTINELA 360] -> Cierre de procesos finalizado.")

if __name__ == "__main__":
    main()
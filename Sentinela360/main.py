"""Sentinela360 - main.py

Funcionalidad: Punto de entrada de la aplicación de detección y alertas.
Por qué existe: Orquestar la carga del modelo, la lectura de zonas, la captura de cámara y la lógica de detección/alerta.
Para qué sirve: Ejecutar inferencia sobre frames de cámara, anotar resultados y guardar evidencias en 'Alertas/'.
Cómo funciona: Carga 'best.pt', lee archivos JSON de zona, procesa cada frame con YOLO y la lógica geométrica, calcula riesgo y persiste alertas en disco.
"""

import os
import time
import cv2
import numpy as np
import json
import supervision as sv
from ultralytics import YOLO
# Importante: no modificar el orden de imports salvo que sepas las dependencias.

# REGLAS DE NEGOCIO DEL MODELO ESTABLE
PESOS_RIESGO = {
    "casco": 35,
    "chaleco": 35,
    "lentes": 10,
    "guantes": 10,
    "botas": 10
}

def cargar_zona(id_camara, width, height):
    """Carga el polígono buscando el nombre oficial de la calibración.

    Bloque: lectura de archivos de zona (prioriza zona_cam_<id>.json)
    - Si existe `zona_cam_<id>.json` lo carga y lo devuelve como numpy array.
    - Si no existe, intenta `zona_config.json` (formato alternativo).
    - Si falla, devuelve un polígono por defecto que encuadra casi todo el frame.

    Comentarios por línea importantes:
    "archivo": nombre del archivo esperado para la cámara (ej: zona_cam_0.json)
    "os.path.exists(archivo)": comprueba existencia antes de abrir
    "json.load(f)": parsea JSON con lista de puntos [[x,y],...]
    "np.array(...)": convierte la lista a array para operaciones geométricas
    "return default polygon": fallback seguro cuando no hay config
    """
    archivo = f"zona_cam_{id_camara}.json"  # nombre esperado para la calibración de la cámara
    try:
        # Si existe el archivo específico de la cámara, usarlo
        if os.path.exists(archivo):
            with open(archivo, "r") as f:
                return np.array(json.load(f))  # devuelve np.array([[x,y],...])
        # Si no, intentar archivo genérico de configuración
        elif os.path.exists("zona_config.json"):
            with open("zona_config.json", "r") as f:
                return np.array(json.load(f))
    except Exception:
        # Silencioso en caso de error de parseo; se usa el fallback
        pass
    # Fallback: polígono que cubre casi todo el frame, con margen de 5px
    return np.array([[5, 5], [width - 5, 5], [width - 5, height - 5], [5, height - 5]])

def hay_interseccion(box_persona, box_equipo):
    """Comprueba si dos cajas (xyxy) se superponen.

    Por línea:
    - xA,yA: coordenadas superior-izquierda del área de intersección
    - xB,yB: coordenadas inferior-derecha del área de intersección
    - El área se calcula como (xB-xA)*(yB-yA)
    - Si el área es mayor que 0, existe intersección
    """
    xA = max(box_persona[0], box_equipo[0])  # x izquierda del solapamiento
    yA = max(box_persona[1], box_equipo[1])  # y superior del solapamiento
    xB = min(box_persona[2], box_equipo[2])  # x derecha del solapamiento
    yB = min(box_persona[3], box_equipo[3])  # y inferior del solapamiento
    return max(0, xB - xA) * max(0, yB - yA) > 0  # True si hay área positiva

def main():
    print("[SENTINELA 360] -> Inicializando IA con Modelo Estable...")
    model = YOLO("best.pt") 

    video_path = 0  # 0 = Cámara Laptop Principal
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error CRÍTICO: No se conectó la cámara {video_path}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    polygon = cargar_zona(video_path, width, height)
    # PARCHE: triggering_anchors al centro para que detecte el cuerpo cortado por la cámara
    zone = sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.CENTER])
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=2, text_thickness=2)

    consecutive_infractions = 0
    FRAMES_REQUERIDOS = 60  
    alert_cooldown = False  

    if not os.path.exists("Alertas"): 
        os.makedirs("Alertas")
    
    archivo_log = "Alertas/registro_auditoria.csv"
    if not os.path.exists(archivo_log):
        with open(archivo_log, "w") as log:
            log.write("Fecha,Hora,Evento,Archivo_Evidencia\n")

    cv2.namedWindow("Sentinela 360 - Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sentinela 360 - Dashboard", 1280, 720)

    print("[SENTINELA 360] -> Sistema en ejecución. Presione ESC para finalizar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(10)
            continue  

        results = model(frame, device=0, conf=0.15)[0]
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
                # Almacenamos diccionario para inyectar el riesgo luego
                personas_boxes.append({"box": xyxy, "riesgo": 0})
            else:
                equipos_boxes.append((nombre_clase, xyxy))

        infracciones_actuales = 0

        # Lógica de cálculo original  Superposición
        for p_info in personas_boxes:
            px1, py1, px2, py2 = p_info["box"]
            
            epp_encontrado = {"helmet": False, "vest": False, "goggle": False, "gloves": False, "boots": False}
            faltas_explicitas = {"no_helmet": False, "no_goggle": False, "no_gloves": False, "no_boots": False, "none": False}

            for equipo_nombre, e_box in equipos_boxes:
                if hay_interseccion(p_info["box"], e_box):
                    if equipo_nombre in epp_encontrado:
                        epp_encontrado[equipo_nombre] = True
                    elif equipo_nombre in faltas_explicitas:
                        faltas_explicitas[equipo_nombre] = True

            riesgo_persona = 0

            if faltas_explicitas["none"]:
                riesgo_persona = 100
            else:
                if faltas_explicitas["no_helmet"] or not epp_encontrado["helmet"]: riesgo_persona += PESOS_RIESGO["casco"]
                if not epp_encontrado["vest"]: riesgo_persona += PESOS_RIESGO["chaleco"]
                if faltas_explicitas["no_goggle"] or not epp_encontrado["goggle"]: riesgo_persona += PESOS_RIESGO["lentes"]
                if faltas_explicitas["no_gloves"] or not epp_encontrado["gloves"]: riesgo_persona += PESOS_RIESGO["guantes"]
                if faltas_explicitas["no_boots"] or not epp_encontrado["boots"]: riesgo_persona += PESOS_RIESGO["botas"]

            riesgo_persona = min(riesgo_persona, 100)
            p_info["riesgo"] = riesgo_persona

            if riesgo_persona > 0:
                infracciones_actuales += 1

        if infracciones_actuales > 0:
            consecutive_infractions += 1
        else:
            if consecutive_infractions > 0: consecutive_infractions -= 1 

        labels = [f"{model.names[class_id]} {conf:.2f}" for class_id, conf in zip(detections.class_id, detections.confidence)]
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = zone_annotator.annotate(scene=frame)

        # Semáforo dinámico inyectado
        for p_info in personas_boxes:
            px1, py1, px2, py2 = map(int, p_info["box"])
            riesgo = p_info["riesgo"]
            
            if riesgo == 0: color_caja = (0, 255, 0)
            elif riesgo <= 35: color_caja = (0, 255, 255)
            else: color_caja = (0, 0, 255)

            cv2.rectangle(frame, (px1, py1), (px2, py2), color_caja, 4)
            cv2.putText(frame, f"{riesgo}% Riesgo", (px1, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_caja, 3)

        progreso = min(100, int((consecutive_infractions / FRAMES_REQUERIDOS) * 100))
        
        if infracciones_actuales == 0:
            estado_texto = "ESTADO: SEGURO"
            color_estado = (0, 255, 0) 
        else:
            estado_texto = "ESTADO: PELIGRO (Infractor detectado)"
            color_estado = (0, 0, 255) 

        cv2.putText(frame, estado_texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_estado, 2)
        cv2.putText(frame, f"Alerta Confirmando: {progreso}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if consecutive_infractions >= FRAMES_REQUERIDOS and not alert_cooldown:
            fecha_actual = time.strftime("%Y-%m-%d")
            hora_actual = time.strftime("%H:%M:%S")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            ruta_archivo = f"Alertas/INFRACCION_{stamp}.jpg"
            
            cv2.imwrite(ruta_archivo, frame)
            
            with open(archivo_log, "a") as log:
                log.write(f"{fecha_actual},{hora_actual},Infraccion Detectada,{ruta_archivo}\n")
                
            print(f" Evidencia asegurada y registrada: {stamp}")
            alert_cooldown = True  

        if consecutive_infractions == 0:
            alert_cooldown = False

        cv2.imshow("Sentinela 360 - Dashboard", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]: 
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("[SENTINELA 360] -> Cierre de procesos finalizado.")

if __name__ == "__main__":
    main()
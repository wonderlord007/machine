import os
import time
import cv2
import numpy as np
import json
import supervision as sv
from ultralytics import YOLO

# Constantes de arquitectura
WIDTH = 1280
HEIGHT = 720
FRAMES_REQUERIDOS = 60

def cargar_zona(id_camara):
    """Carga el polígono en resolución 1280x720."""
    archivo = f"zona_cam_{id_camara}.json"
    if os.path.exists(archivo):
        with open(archivo, "r") as f:
            return np.array(json.load(f))
    # Margen de 5 píxeles aplicado a la resolución HD
    return np.array([[5, 5], [WIDTH - 5, 5], [WIDTH - 5, HEIGHT - 5], [5, HEIGHT - 5]])

def procesar_frame(frame, model, zone, box_annotator, label_annotator, zone_annotator, nombre_camara, estado):
    """Ejecuta el análisis lógico, la auditoría y gestiona las capturas de cada sector."""
    results = model(frame, device=0, conf=0.15)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    is_inside = zone.trigger(detections=detections)

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

    # Lógica espacial de asociación
    for p_box in personas_boxes:
        px1, py1, px2, py2 = p_box
        tiene_casco, tiene_chaleco, falta_explicita = False, False, False

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

    # Filtro temporal estructurado en diccionario de estados
    if infracciones_actuales > 0:
        estado["consecutivas"] += 1
    else:
        if estado["consecutivas"] > 0:
            estado["consecutivas"] -= 1

    if estado["consecutivas"] >= FRAMES_REQUERIDOS and not estado["cooldown"]:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        estado["pendiente_guardar"] = stamp
        estado["cooldown"] = True

    if estado["consecutivas"] == 0:
        estado["cooldown"] = False

    # Renderizado
    labels = [f"{model.names[c]} {conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    # Indicadores visuales en pantalla
    progreso = min(100, int((estado["consecutivas"] / FRAMES_REQUERIDOS) * 100))

    if infracciones_actuales == 0:
        cv2.putText(frame, f"{nombre_camara} - SEGURO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"{nombre_camara} - PELIGRO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(frame, f"Nivel de Riesgo: {progreso}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Persistencia de evidencia fotográfica
    if estado.get("pendiente_guardar"):
        stamp = estado.pop("pendiente_guardar")
        # Formatear nombre del sector para el archivo
        sector_limpio = nombre_camara.replace(" ", "_")
        cv2.imwrite(f"Alertas/INFRACCION_{sector_limpio}_{stamp}.jpg", frame)
        print(f"⚠️ Evidencia física registrada en {nombre_camara}: {stamp}")

    return frame

def main():
    print("[SENTINELA 360] -> Inicializando Centro de Comando DUAL...")
    model = YOLO("best.pt") 

    cap_laptop = cv2.VideoCapture(0) 
    cap_movil = cv2.VideoCapture(1)  

    if not cap_laptop.isOpened() or not cap_movil.isOpened():
        print("❌ Error: Se requieren ambas cámaras operativas para el modo dual.")
        return

    cap_laptop.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap_laptop.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap_movil.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap_movil.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # IMPORTANTE: Eliminado el 'frame_resolution_wh' para compatibilidad con tu versión
    zona_cam0 = sv.PolygonZone(polygon=cargar_zona(0))
    zona_cam1 = sv.PolygonZone(polygon=cargar_zona(1))
    
    annotator_cam0 = sv.PolygonZoneAnnotator(zone=zona_cam0, color=sv.Color.BLUE, thickness=2)
    annotator_cam1 = sv.PolygonZoneAnnotator(zone=zona_cam1, color=sv.Color.RED, thickness=2)

    # Estado temporal independiente para evitar interferencias cruzadas
    estado_cam0 = {"consecutivas": 0, "cooldown": False}
    estado_cam1 = {"consecutivas": 0, "cooldown": False}

    if not os.path.exists("Alertas"): 
        os.makedirs("Alertas")

    cv2.namedWindow("Sentinela 360 - Comando Central", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sentinela 360 - Comando Central", 2560, 720)

    print("✅ [SISTEMA DUAL ACTIVO] -> Transmitiendo a máxima resolución...")

    while True:
        ret1, frame1 = cap_laptop.read()
        ret2, frame2 = cap_movil.read()

        if not ret1 or not ret2:
            cv2.waitKey(10)
            continue

        frame1 = cv2.resize(frame1, (WIDTH, HEIGHT))
        frame2 = cv2.resize(frame2, (WIDTH, HEIGHT))

        frame1_procesado = procesar_frame(
            frame1, model, zona_cam0, box_annotator, label_annotator, annotator_cam0, "SECTOR 0", estado_cam0
        )
        
        frame2_procesado = procesar_frame(
            frame2, model, zona_cam1, box_annotator, label_annotator, annotator_cam1, "SECTOR 1", estado_cam1
        )

        # Fusión horizontal (Ancho total: 2560px)
        pantalla_dividida = np.hstack((frame1_procesado, frame2_procesado))
        cv2.imshow("Sentinela 360 - Comando Central", pantalla_dividida)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap_laptop.release()
    cap_movil.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Sistema apagado de forma segura.")

if __name__ == "__main__":
    main()
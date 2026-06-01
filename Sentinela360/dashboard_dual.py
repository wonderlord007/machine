import os
import time
import cv2
import numpy as np
import json
import supervision as sv
from ultralytics import YOLO

WIDTH = 1280
HEIGHT = 720
FRAMES_REQUERIDOS = 60

PESOS_RIESGO = {
    "casco": 35,
    "chaleco": 35,
    "lentes": 10,
    "guantes": 10,
    "botas": 10
}

def cargar_zona(id_camara):
    archivo = f"zona_cam_{id_camara}.json"
    try:
        if os.path.exists(archivo):
            with open(archivo, "r") as f:
                return np.array(json.load(f))
    except Exception:
        pass
    return np.array([[5, 5], [WIDTH - 5, 5], [WIDTH - 5, HEIGHT - 5], [5, HEIGHT - 5]])

def hay_interseccion(box_persona, box_equipo):
    xA = max(box_persona[0], box_equipo[0])
    yA = max(box_persona[1], box_equipo[1])
    xB = min(box_persona[2], box_equipo[2])
    yB = min(box_persona[3], box_equipo[3])
    return max(0, xB - xA) * max(0, yB - yA) > 0

def procesar_frame(frame, model, zone, box_annotator, label_annotator, zone_annotator, nombre_camara, estado):
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
            if riesgo_persona > riesgo_maximo_zona:
                riesgo_maximo_zona = riesgo_persona

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
        riesgo = p_info["riesgo"]
        
        if riesgo == 0: color = (0, 255, 0)
        elif riesgo <= 35: color = (0, 255, 255)
        else: color = (0, 0, 255)

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 4)
        cv2.putText(frame, f"{riesgo}% Riesgo", (px1, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    progreso_tiempo = min(100, int((estado["consecutivas"] / FRAMES_REQUERIDOS) * 100))

    if riesgo_maximo_zona == 0:
        cv2.putText(frame, f"{nombre_camara} - ESTADO: SEGURO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Gravedad de Riesgo: 0%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif riesgo_maximo_zona <= 40:
        cv2.putText(frame, f"{nombre_camara} - PELIGRO LEVE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Gravedad de Riesgo: {riesgo_maximo_zona}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif riesgo_maximo_zona <= 70:
        cv2.putText(frame, f"{nombre_camara} - PELIGRO MODERADO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Gravedad de Riesgo: {riesgo_maximo_zona}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(frame, f"{nombre_camara} - PELIGRO CRITICO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Gravedad de Riesgo: {riesgo_maximo_zona}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if riesgo_maximo_zona > 0:
        cv2.putText(frame, f"Confirmando Alerta: {progreso_tiempo}%", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if estado.get("pendiente_guardar"):
        stamp = estado.pop("pendiente_guardar")
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
    
    # PARCHE: triggering_anchors al centro
    zona_cam0 = sv.PolygonZone(polygon=cargar_zona(0), triggering_anchors=[sv.Position.CENTER])
    zona_cam1 = sv.PolygonZone(polygon=cargar_zona(1), triggering_anchors=[sv.Position.CENTER])
    
    annotator_cam0 = sv.PolygonZoneAnnotator(zone=zona_cam0, color=sv.Color.BLUE, thickness=2)
    annotator_cam1 = sv.PolygonZoneAnnotator(zone=zona_cam1, color=sv.Color.RED, thickness=2)

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
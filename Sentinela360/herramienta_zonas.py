"""Herramienta de calibración de zonas - herramienta_zonas.py

Funcionalidad: Interfaz para definir y guardar zonas de interés por cámara.
Por qué existe: Permitir al usuario delimitar áreas donde se evaluarán detecciones (sin tocar el código).
Para qué sirve: Generar archivos 'zona_cam_<id>.json' con las coordenadas de los polígonos.
Cómo funciona: Abre la cámara, permite capturar 4 clics del usuario y guarda las coordenadas en un JSON.
"""

import cv2
import numpy as np
import json
import sys

# Definición estándar de resolución para todo el sistema
WIDTH = 1280
HEIGHT = 720

ID_CAMARA = 0
puntos = []
archivo_salida = ""

def capturar_clic(event, x, y, flags, param):
    # Evento: clic izquierdo del ratón
    if event == cv2.EVENT_LBUTTONDOWN:
        # Guardar coordenada (x,y) en la lista global 'puntos'
        puntos.append([x, y])
        print(f"📍 Coordenada registrada en Cámara {ID_CAMARA}: [{x}, {y}]")
        
        # Cuando tengamos 4 puntos, serializar a JSON y guardar
        if len(puntos) == 4:
            with open(archivo_salida, "w") as archivo_json:
                json.dump(puntos, archivo_json)
            
            print(f"\n✅ Zona guardada exitosamente en '{archivo_salida}'.")
            print("Presione la tecla ESC en la ventana de video para finalizar.")

def main():
    global ID_CAMARA, archivo_salida
    
    print("=======================================")
    print("  CALIBRACIÓN DE ZONAS - RESOLUCIÓN HD ")
    print("=======================================")
    print("[0] Cámara Principal (Laptop)")
    print("[1] Cámara Secundaria (Celular)")
    
    try:
        # Solicitar al usuario la cámara a configurar (0 o 1)
        seleccion = input("\nIngrese el número de la cámara a configurar (0 o 1): ")
        ID_CAMARA = int(seleccion)
    except ValueError:
        # Manejo simple de entrada inválida
        print("Entrada no válida. El programa se cerrará.")
        sys.exit()

    archivo_salida = f"zona_cam_{ID_CAMARA}.json"
    
    # Abrir la cámara seleccionada
    print(f"\n[HERRAMIENTA] -> Estableciendo conexión con la cámara {ID_CAMARA}...")
    cap = cv2.VideoCapture(ID_CAMARA)

    # Validar que el dispositivo esté disponible
    if not cap.isOpened():
        print(f"❌ Error CRÍTICO: No se detectó señal en el índice {ID_CAMARA}.")
        sys.exit()

    # Forzar resolución HD para que las coordenadas sean consistentes
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    nombre_ventana = f"Calibracion - Camara {ID_CAMARA}"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, WIDTH, HEIGHT)
    cv2.setMouseCallback(nombre_ventana, capturar_clic)

    print("\nINSTRUCCIONES: Haga clic izquierdo en 4 puntos para delimitar el área de riesgo.")

    while True:
        ret, frame = cap.read()
        # Leer frame de la cámara
        if not ret:
            # Si no hay frame, esperar y reintentar
            cv2.waitKey(10)
            continue

        # Redimensionado de seguridad para garantizar coincidencia de píxeles
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Dibujar puntos registrados
        for pt in puntos:
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
        
        if len(puntos) > 1:
            for i in range(len(puntos) - 1):
                cv2.line(frame, tuple(puntos[i]), tuple(puntos[i+1]), (0, 0, 255), 2)
        if len(puntos) == 4:
            cv2.line(frame, tuple(puntos[3]), tuple(puntos[0]), (0, 0, 255), 2)

        cv2.imshow(nombre_ventana, frame)
        
        # Lectura de teclas: si la zona está completa o se presiona ESC, salir
        key = cv2.waitKey(1) & 0xFF
        if len(puntos) == 4 and key != 255:
            break
        if key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
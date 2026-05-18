# 🛡️ Sentinela 360
**Sistema de Monitoreo de Seguridad Industrial (HSE) con Inteligencia Artificial**

Sentinela 360 es un sistema de visión computacional de grado de producción diseñado para auditar en tiempo real el cumplimiento del Equipo de Protección Personal (EPP) en zonas de riesgo. Desarrollado como proyecto para la carrera de Ingeniería de Software con Inteligencia Artificial, este sistema utiliza una arquitectura YOLO para identificar trabajadores y verificar la portación de cascos, chalecos, guantes, botas y lentes.

---

## 🚀 Características Principales
* **Arquitectura Dual-Cam (Split-Screen):** Monitoreo simultáneo en HD (1280x720) mediante una cámara CCTV/Webcam y una secundaria inalámbrica.
* **Asociación Espacial Geométrica:** Algoritmo que calcula el centroide de cada equipo detectado para vincularlo matemáticamente a un trabajador específico, evitando falsos positivos.
* **Filtro Temporal Antiruido:** Sistema de persistencia que requiere confirmación visual continua (por defecto, 60 fotogramas) antes de disparar una alerta.
* **Trazabilidad:** Generación de bitácoras `.csv` y evidencia fotográfica con HUD renderizado.

---

## 📊 Sobre el Dataset y el Modelo
El cerebro de Sentinela 360 es un modelo YOLOv10 entrenado específicamente para entornos industriales. 
* **Plataformas de entrenamiento:** Roboflow (etiquetado y preprocesamiento) y Google Colab (entrenamiento en GPU).
* **Clases Detectadas (11 en total):** * *Elementos base:* `Person` (Persona), `helmet` (Casco), `vest` (Chaleco).
  * *Faltas explícitas:* `no_helmet`, `no_vest`, `no_goggle`, `no_gloves`, `no_boots`, `none`.
* **Peso del modelo:** El archivo `best.pt` contiene los pesos finales optimizados.

---

## ⚙️ Instalación y Configuración

### 1. Clonar el Repositorio
Para descargar el proyecto en tu máquina local, ejecuta en tu terminal:
```bash
git clone [https://github.com/tu-usuario/Sentinela360.git](https://github.com/tu-usuario/Sentinela360.git)
cd Sentinela360

```

### 2. Entorno Virtual y Dependencias

Se recomienda utilizar un entorno virtual (venv o conda). El sistema requiere aceleración gráfica por hardware (NVIDIA CUDA) para procesar múltiples cámaras sin latencia.

**Instalar PyTorch (Soporte GPU - CUDA 11.8):**

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

```

**Instalar dependencias de visión:**

```bash
pip install opencv-python numpy supervision ultralytics roboflow

```

---

## 🛠️ Modificación y Personalización (Ajustes para Desarrolladores)

Si deseas adaptar el código para otros entornos o modificar su sensibilidad, puedes alterar las siguientes variables directamente en `dashboard_dual.py`:

* **Modificar el tiempo de alerta (Filtro Antiruido):**
Busca la variable `FRAMES_REQUERIDOS`. Por defecto está en `60` (equivale a 2 segundos a 30 FPS).
* *Para una alerta más rápida (1 segundo):* Cambiar a `30`.
* *Para una alerta más tolerante (5 segundos):* Cambiar a `150`.


* **Modificar la sensibilidad de la IA (Confidence):**
Busca la línea de inferencia: `model(frame, device=0, conf=0.15)`.
* Si la IA genera falsos positivos (detecta cosas que no son), sube el umbral a `conf=0.40`.
* Si la IA ignora objetos lejanos, bájalo a `conf=0.10`.


* **Cambiar el origen de las cámaras:**
Las variables `cap_laptop = cv2.VideoCapture(0)` y `cap_movil = cv2.VideoCapture(1)` definen los puertos. Modifica los índices (0, 1, 2) según tu configuración de hardware o puertos RTSP para cámaras IP.

---

## 🚦 Guía de Uso Rápido

### 1. Calibración Topográfica (Mapping)

Debe delimitar las "Zonas de Riesgo" para enseñar a la IA dónde mirar.

```bash
python herramienta_zonas.py

```

* Ingresa `0` para mapear la cámara principal y marca 4 puntos.
* Ingresa `1` para mapear la cámara secundaria (ej. cámara inalámbrica Redmi 14C u otro móvil) y marca 4 puntos.

### 2. Despliegue en Producción

Una vez generados los archivos `zona_cam_0.json` y `zona_cam_1.json`, inicia el Centro de Comando:

```bash
python dashboard_dual.py

```

* **Cierre seguro:** Presiona la tecla `ESC` para detener la inferencia, purgar la VRAM y cerrar el sistema. Las evidencias se guardarán automáticamente en la carpeta `/Alertas`.

```

Con este README cubres exactamente lo que un repositorio bien hecho necesita: origen de los datos, instalación paso a paso, configuración de parámetros y uso del sistema. ¡Listo para impactar!

```

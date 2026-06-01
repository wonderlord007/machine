# 🛡️ Sentinela 360

**Sistema de Monitoreo de Seguridad Industrial (HSE) con Inteligencia Artificial**

Sentinela 360 es un software de visión computacional de grado de producción diseñado para auditar en tiempo real el cumplimiento del Equipo de Protección Personal (EPP) en zonas de riesgo. Desarrollado como proyecto para la carrera de Ingeniería de Software con Inteligencia Artificial, este sistema utiliza una arquitectura YOLO integrada en una interfaz gráfica moderna para identificar trabajadores y verificar la portación de cascos, chalecos, guantes, botas y lentes.

---

## 🚀 Características Principales

* **Interfaz Gráfica Unificada (GUI):** Aplicación de escritorio nativa en modo oscuro (construida con CustomTkinter) que elimina la necesidad de usar la consola. Incluye un *Launcher* para seleccionar el modo de operación.
* **Escalabilidad de Monitoreo:** * *Modo Individual:* Monitoreo enfocado a pantalla completa mediante una cámara web principal.
* *Modo Dual:* Monitoreo simultáneo en pantalla dividida mediante una cámara principal y una secundaria (ej. inalámbrica/RTSP).


* **Calibración Geométrica Integrada:** Herramienta visual dentro de la app para delimitar áreas de peligro mediante 4 clics, adaptando automáticamente la resolución (Aspect Ratio) sin distorsionar la imagen.
* **Asociación Espacial Avanzada:** Algoritmo de superposición (Overlap) anclado al centro de masa, que permite vincular equipos de protección al trabajador incluso si la cámara corta parte del cuerpo (planos medios).
* **Filtro Temporal Antiruido:** Sistema de persistencia que requiere confirmación visual continua (60 fotogramas por defecto) antes de disparar una alerta, evitando falsos positivos.
* **Trazabilidad de Evidencia:** Generación automática de bitácoras y evidencia fotográfica con HUD renderizado.

---

## 📊 Sobre el Dataset y el Modelo

El cerebro de Sentinela 360 es un modelo YOLO entrenado y optimizado específicamente para entornos industriales.

* **Plataformas de entrenamiento:** Roboflow (etiquetado y preprocesamiento) y Google Colab (entrenamiento en GPU).
* **Versión Estable (Inglés):** El sistema utiliza un modelo de alta confiabilidad con clases base como `Person` (Persona), `helmet` (Casco), `vest` (Chaleco), junto con detecciones de faltas explícitas (`no_helmet`, `no_goggle`, `none`, etc.).
* **Peso del modelo:** El archivo `best.pt` contiene los pesos finales optimizados para inferencia en tiempo real.

---

## ⚙️ Instalación y Configuración

### 1. Clonar el Repositorio

Para descargar el proyecto en tu máquina local, ejecuta en tu terminal:

```bash
git clone https://github.com/wonderlord007/Sentinela360.git
cd Sentinela360

```

### 2. Entorno Virtual y Dependencias

Se recomienda utilizar un entorno virtual (venv o conda). El sistema aprovecha la aceleración gráfica por hardware (NVIDIA CUDA) para procesar los flujos de video sin latencia.

**Instalar PyTorch (Soporte GPU - CUDA 11.8):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

**Instalar dependencias de visión e Interfaz Gráfica:**

```bash
pip install opencv-python numpy supervision ultralytics roboflow customtkinter Pillow

```

---

## 🛠️ Ajustes para Desarrolladores (Fine-Tuning)

Si deseas adaptar el código para otros entornos o modificar su sensibilidad, puedes alterar las siguientes variables directamente en `sentinela_app.py`:

* **Modificar el tiempo de alerta (Filtro Antiruido):**
Busca la variable `FRAMES_REQUERIDOS`. Por defecto está en `60` (aprox. 2 segundos).
* *Para alertas inmediatas:* Cambiar a `30`.
* *Para mayor tolerancia:* Cambiar a `120`.


* **Modificar la sensibilidad de la IA (Confidence):**
Busca la línea de inferencia: `model(frame, device=0, conf=0.45, verbose=False)`.
* El valor `0.45` está optimizado para evitar falsos positivos con uniformes civiles. Si la IA ignora objetos lejanos, bájalo a `0.25`.



---

## 🚦 Guía de Uso

Toda la operación se realiza ahora desde un único archivo maestro.

### 1. Iniciar el Sistema

Ejecuta el siguiente comando en tu terminal para abrir el Launcher:

```bash
python sentinela_app.py

```

### 2. Flujo de Operación

1. **Calibración:** En el menú principal, haz clic en **"⚙️ Calibrar Zona"** (Cam 0 o Cam 1). Haz 4 clics en la ventana de video para delimitar la zona de riesgo y presiona `ESC` para guardar. (Nota: La Cam 0 cuenta con efecto espejo nativo para facilitar la orientación).
2. **Monitoreo:** Selecciona **"Iniciar Modo Individual"** o **"Iniciar Modo Dual"**.
3. **Gestión:** El sistema detectará las infracciones automáticamente, mostrando el riesgo en pantalla y guardando capturas en la carpeta `/Alertas`.
4. **Cierre Seguro:** Utiliza el botón rojo **"Finalizar Turno"** en la interfaz gráfica para liberar las cámaras y apagar los procesos correctamente.

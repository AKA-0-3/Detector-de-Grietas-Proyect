# 🧱 Detector de Grietas (Concrete Crack Detection)

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Gradio-App-blue?style=for-the-badge&logo=gradio" alt="Gradio">
  <img src="https://img.shields.io/badge/FastAPI-API-green?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge&logo=python" alt="Python">
</p>

> ⚠️ **AVISO IMPORTANTE SOBRE PYTHON Y TENSORFLOW:** 
> Debido a los ciclos de actualización de TensorFlow, las versiones más recientes de Python (como 3.12 o 3.13) **no son compatibles** actualmente y generarán errores al intentar instalar las dependencias. Se exige y recomienda encarecidamente utilizar **Python 3.10 o 3.11** (este proyecto ha sido probado y validado exitosamente en **Python 3.11.x**).

Proyecto completo y modular para la **detección y clasificación de grietas en superficies de concreto** utilizando Redes Neuronales Convolucionales (CNN) con TensorFlow/Keras. El sistema incluye una interfaz web interactiva con Gradio (soporte para cámara web y aprendizaje incremental), una API REST en FastAPI y un cuaderno Jupyter completo para entrenamiento y evaluación.

---

## 🚀 Características Principales

- **Modelo CNN Personalizado:** Arquitectura optimizada con capas convolucionales, *Max Pooling*, *Dropout* y aumento de datos (*Data Augmentation*).
- **Interfaz Web Interactiva (Gradio):** Permite predecir imágenes estáticas o capturadas por la cámara web, corregir etiquetas en tiempo real y reentrenar de forma incremental.
- **API Backend (FastAPI):** Endpoints listos para producción (`/predecir` y `/entrenar`) para integrar el modelo en otras aplicaciones.
- **Cuaderno Jupyter Completo:** Incluye flujos para entrenamiento con validación, métricas de rendimiento, matriz de confusión y estimación de inclinación mediante la Transformada de Hough.

---

## 📂 Estructura del Repositorio

```text
Detector-de-Grietas-Proyect/
│
├── app.py                      # Aplicación web interactiva principal (Gradio)
├── app_gradio.py               # Versión adaptada de la app (compatible con Google Colab / Local)
├── server_api.py               # Servidor API REST con FastAPI
├── deteccion_grietas_limpio.ipynb  # Notebook de Jupyter con el flujo completo de entrenamiento
└── README.md                   # Documentación del proyecto

import os
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

IMG_HEIGHT = 128
IMG_WIDTH = 128
DATA_DIR = './dataset'
MODEL_PATH = './modelo_grietas.keras'
ENTRENAR_CADA = 5
BATCH_SIZE = 32
SEED = 123

modelo = tf.keras.models.load_model(MODEL_PATH)
CLASES = sorted([d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d))])

app = FastAPI(title="Detector de Grietas API")


@app.get("/")
def raiz():
    return {"clases": CLASES, "modelo": MODEL_PATH, "entrenar_cada": ENTRENAR_CADA}


def _preparar(imagen_bgr):
    img = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)).astype("float32")
    return np.expand_dims(img, axis=0)


@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    datos = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "No se pudo leer la imagen"}, status_code=400)
    prob = float(modelo.predict(_preparar(img), verbose=0)[0][0])
    idx = int(prob > 0.5)
    return {
        "clase": CLASES[idx],
        "certeza": round(float(max(prob, 1 - prob)), 4),
        "prob_positive": round(prob, 4),
    }


@app.post("/entrenar")
async def entrenar(file: UploadFile = File(...), etiqueta: str = Form(...)):
    if etiqueta not in CLASES:
        return JSONResponse(
            {"error": f"etiqueta debe ser una de {CLASES}"}, status_code=400)

    datos = await file.read()
    carpeta = os.path.join(DATA_DIR, etiqueta)
    os.makedirs(carpeta, exist_ok=True)
    nombre = f"camara_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    with open(os.path.join(carpeta, nombre), "wb") as f:
        f.write(datos)

    total = sum(len(os.listdir(os.path.join(DATA_DIR, c))) for c in CLASES)
    reentrenado = False
    if total % ENTRENAR_CADA == 0 and min(
            len(os.listdir(os.path.join(DATA_DIR, c))) for c in CLASES) >= 2:
        ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE, seed=SEED)
        modelo.fit(ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE), epochs=3)
        modelo.save(MODEL_PATH)
        reentrenado = True

    return {"guardada": nombre, "clase": etiqueta, "reentrenado": reentrenado}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

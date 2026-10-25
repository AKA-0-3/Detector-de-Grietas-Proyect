import os
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import gradio as gr

IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
SEED = 123
EPOCHS_REENTRENO = 5

EN_COLAB = os.path.isdir('/content')
DATA_DIR = '/content/archive_extracted' if EN_COLAB else './archive_extracted'
MODEL_PATH = '/content/modelo_grietas.keras' if EN_COLAB else './modelo_grietas.keras'

ENTRENAR_CADA = 5


def construir_modelo():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if os.path.isdir(DATA_DIR):
    CLASES = sorted([d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d))]) or ["Negative", "Positive"]
else:
    CLASES = ["Negative", "Positive"]
    print(f"AVISO: no existe {DATA_DIR}. Crea las carpetas Positive/ y Negative/ con imagenes.")

CONTADOR_NUEVAS = 0

if os.path.exists(MODEL_PATH):
    modelo = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado:", MODEL_PATH)
else:
    modelo = construir_modelo()
    print("No se encontro modelo guardado; se creo uno nuevo (entrena primero para buenos resultados).")
print("Clases:", CLASES)


def predecir(imagen):
    if imagen is None:
        return "Sube una imagen o usa la camara."
    img = cv2.resize(imagen, (IMG_WIDTH, IMG_HEIGHT)).astype("float32")
    prob = float(modelo.predict(np.expand_dims(img, axis=0), verbose=0)[0][0])
    idx = int(prob > 0.5)
    certeza = prob if idx == 1 else 1 - prob
    return f"{CLASES[idx]}  (certeza: {certeza:.2%})"


def reentrenar(epochs=EPOCHS_REENTRENO):
    cuentas = {c: len(os.listdir(os.path.join(DATA_DIR, c))) for c in CLASES}
    print("Tamano del dataset:", cuentas)
    if min(cuentas.values()) < 2:
        print("Cada clase necesita al menos 2 imagenes para reentrenar.")
        return
    kw = dict(seed=SEED, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
    if min(cuentas.values()) >= 10:
        tr = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, validation_split=0.2, subset="training", **kw)
        va = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, validation_split=0.2, subset="validation", **kw)
        tr = tr.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
        va = va.cache().prefetch(tf.data.AUTOTUNE)
        modelo.fit(tr, validation_data=va, epochs=epochs)
    else:
        ds = tf.keras.utils.image_dataset_from_directory(DATA_DIR, **kw)
        ds = ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
        modelo.fit(ds, epochs=epochs)
    modelo.save(MODEL_PATH)
    print("Modelo reentrenado y guardado en:", MODEL_PATH)


def guardar_y_entrenar(imagen, etiqueta, entrenar_ahora=False):
    global CONTADOR_NUEVAS
    if imagen is None:
        return "Toma o sube una imagen primero."
    carpeta = os.path.join(DATA_DIR, etiqueta)
    os.makedirs(carpeta, exist_ok=True)
    nombre = f"camara_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    ruta = os.path.join(carpeta, nombre)
    cv2.imwrite(ruta, cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
    CONTADOR_NUEVAS += 1
    if entrenar_ahora or CONTADOR_NUEVAS >= ENTRENAR_CADA:
        reentrenar()
        CONTADOR_NUEVAS = 0
        return f"Guardada '{nombre}' en {etiqueta}. Modelo reentrenado."
    faltan = ENTRENAR_CADA - CONTADOR_NUEVAS
    return (f"Guardada '{nombre}' en {etiqueta}. "
            f"Reentrenare en {faltan} imagen(es) mas (o pulsa 'Reentrenar ahora').")


with gr.Blocks(title="Detector de Grietas") as demo:
    gr.Markdown(
        "# 🧱 Detector de Grietas - entrena con tu camara\n"
        "Toma fotos, corrige la etiqueta si el modelo se equivoca y el modelo mejora solo."
    )
    with gr.Tab("🔍 Predecir"):
        with gr.Row():
            img_in = gr.Image(sources=["webcam", "upload"], type="numpy",
                              label="Cámara o imagen subida")
            btn_pred = gr.Button("🔎 Predecir", variant="primary")
        res = gr.Textbox(label="Resultado")
        btn_pred.click(predecir, inputs=[img_in], outputs=[res])

    with gr.Tab("📷 Entrenar"):
        gr.Markdown(
            "Toma la foto, marca la **etiqueta correcta** "
            "(corrige si el modelo se equivocó) y guarda. "
            f"El modelo se reentrena automáticamente cada {ENTRENAR_CADA} imágenes."
        )
        img_tr = gr.Image(sources=["webcam", "upload"], type="numpy", label="Toma la foto")
        etq = gr.Radio(CLASES, label="Etiqueta correcta")
        with gr.Row():
            btn_guardar = gr.Button("💾 Guardar imagen")
            btn_reent = gr.Button("🔄 Reentrenar ahora")
        msg = gr.Textbox(label="Estado")
        btn_guardar.click(guardar_y_entrenar, inputs=[img_tr, etq], outputs=[msg])
        btn_reent.click(lambda i, e: guardar_y_entrenar(i, e, True),
                        inputs=[img_tr, etq], outputs=[msg])

if __name__ == "__main__":
    demo.launch(share=True)

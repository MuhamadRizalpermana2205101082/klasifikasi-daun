import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model_daun.h5")

# Label kelas daun (ganti sesuai datasetmu)
label_kelas = ["Daun Hijau", "Daun Merah Hijau", "Daun Lainnya"]

def prediksi_daun(img):
    image = img.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    kelas = np.argmax(pred, axis=1)[0]
    return f"Hasil Prediksi: {label_kelas[kelas]}"

# Interface Gradio
demo = gr.Interface(fn=prediksi_daun, inputs=gr.Image(type="pil"), outputs="text", title="🌿 Deteksi Daun Tanaman Hias")
demo.launch()


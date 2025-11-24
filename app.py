import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import traceback

# Enable memory growth (optional but recommended for large models)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# ---------------------------
# LOAD MODELS
# ---------------------------
try:
    brain_model = tf.keras.models.load_model("brain_tumor_detector.h5", compile=False)
except:
    print("Error loading brain model:")
    traceback.print_exc()

try:
    skin_model = tf.keras.models.load_model("skin_cancer_cnn.h5", compile=False)
except:
    print("Error loading skin model:")
    traceback.print_exc()

# Correct input sizes
BRAIN_SIZE = 240
SKIN_SIZE = 224


# ---------------------------
# PREPROCESS FUNCTION
# ---------------------------
def preprocess(img, size):
    img = img.convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------------------------
# PREDICTION FUNCTIONS
# ---------------------------
def predict_brain(image):
    try:
        x = preprocess(image, BRAIN_SIZE)
        pred = brain_model.predict(x)[0][0]  # sigmoid output

        if pred > 0.5:
            return "🧠 **Brain Tumor Detected** ⚠️"
        else:
            return "🧠 **No Brain Tumor Detected** ✅"
    except Exception:
        return traceback.format_exc()


def predict_skin(image):
    try:
        x = preprocess(image, SKIN_SIZE)
        pred = skin_model.predict(x)[0][0]  # sigmoid output

        if pred > 0.5:
            return "🩺 **Skin Cancer Detected** ⚠️"
        else:
            return "🩺 **No Skin Cancer Detected** ✅"
    except Exception:
        return traceback.format_exc()


# ---------------------------
# BUILD SIMPLE TEST UI
# (We test detection before integrating the fancy UI)
# ---------------------------

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Test Brain Model")
    img1 = gr.Image(type="pil")
    out1 = gr.Textbox()
    btn1 = gr.Button("Predict Brain Tumor")
    btn1.click(predict_brain, inputs=img1, outputs=out1)

    gr.Markdown("## 🩺 Test Skin Model")
    img2 = gr.Image(type="pil")
    out2 = gr.Textbox()
    btn2 = gr.Button("Predict Skin Cancer")
    btn2.click(predict_skin, inputs=img2, outputs=out2)

demo.launch(server_name="0.0.0.0", server_port=7861)



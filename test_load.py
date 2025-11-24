from tensorflow.keras.models import load_model
import traceback

for p in ["brain_tumor_detector.h5", "skin_cancer_cnn.h5"]:
    print("=== Loading", p)
    try:
        m = load_model(p, compile=False)
        m.summary()
    except Exception as e:
        print("ERROR loading", p)
        traceback.print_exc()

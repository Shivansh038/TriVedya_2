# Multi-Disease Detection System

This project is a Multi-Disease Detection System capable of identifying Brain Tumors from MRI scans and Skin Cancer from dermoscopic images. It uses deep-learning classification models trained using TensorFlow/Keras. The system includes a GUI, preprocessing utilities, visualization scripts, and standalone prediction modules for both diseases.

---

## 📁 Project Structure

```
app.py                     # Main application runner
brain_tumor_detector.h5    # Trained deep learning model
brain_tumor_imd.py         # Brain Tumor file
displayTumor.py            # Visualization of tumor images
frames.py                  # GUI frame components
gui.py                     # Complete GUI interface
predictTumor.py            # Script to predict tumor from an image
skinvision (2).ipynb       # Skin cancer file
test_load.py               # Test script to load the model
notumor.jpg                # Sample image (no tumor)
tumor.jpg                  # Sample image (tumor)
tumordetection.jpg         # Result demo image
viewtumor.jpg              # For GUI display
```

---

## 🚀 Features

* **Deep Learning Model** for brain tumor detection using MRI scans
* **GUI-based interface** for interactive predictions
* **Image preprocessing** and visualization utilities
* **Prediction script** for direct command-line use
* **Includes sample images** for quick testing

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install tensorflow keras numpy opencv-python pillow
```

Add any other required libraries depending on your scripts.

### 2. Run the GUI

```bash
python gui.py
```

### 3. Run Prediction from Command Line

```bash
python predictTumor.py --image path/to/image.jpg
```

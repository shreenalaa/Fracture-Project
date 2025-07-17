

# 🦴 Fracture Detection API

This project provides a RESTful API for detecting bone fractures from medical images using a deep learning model built with PyTorch. The model classifies input images as either **fractured** or **not fractured**. The API is built using **Flask**, making it lightweight and easy to deploy.

---

## 🔍 Overview

The goal of this project is to help automate the preliminary diagnosis of bone fractures from medical images (e.g., X-rays). This system accepts image files through a POST request and returns a prediction based on a pretrained PyTorch model.

---

## 🚀 Features

* ✅ REST API endpoint for real-time image classification
* 🧠 Deep learning model trained on fracture detection data
* 🖼️ Preprocessing with torchvision transforms
* 📦 Easy integration into medical tools or frontend apps
* ⚙️ Torch-based inference with no need for GPU

---

## 📁 Project Structure

```
fracture_detection_api/
│
├── saved_models/
│   └── fracture_best_model.pth     # Pretrained PyTorch model
├── app.py                          # Main Flask application
└── README.md                       # Project documentation
```

---

## 📦 Requirements

Install the required Python packages with:

```bash
pip install flask torch torchvision pillow
```

---

## 🧠 Model

* **Architecture**: Custom or standard CNN (e.g., ResNet)
* **Input size**: 224x224 RGB images
* **Classes**:

  * `0`: Fractured
  * `1`: Not Fractured

The model is stored in `saved_models/fracture_best_model.pth`.

---

## 🛠️ Usage

### ▶️ Running the API

Start the Flask server:

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000/`.

---

### 📤 Sending a Prediction Request

Use tools like `curl`, Postman, or a frontend interface to send an image file.

#### Example using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@/path/to/image.jpg"
```

#### Response:

```json
{
  "prediction": "fractured"
}
```

---

## 🧪 Image Preprocessing

* Resize to 224×224 pixels
* Normalize using ImageNet statistics:

  * Mean: `[0.485, 0.456, 0.406]`
  * Std: `[0.229, 0.224, 0.225]`

---

## 📌 Notes

* Ensure the uploaded file is a valid image (e.g., `.jpg`, `.png`)
* This app is configured for development (`debug=True`); remove in production
* Extendable to include confidence scores or Grad-CAM visualization

---

## 📄 License

This project is licensed under the MIT License.

---

## 👷‍♀️ Author

**Shereen Alaa**
Machine Learning Engineer
[LinkedIn](https://www.linkedin.com/in/shreen-alaa/) | [GitHub](https://github.com/shreenalaa)


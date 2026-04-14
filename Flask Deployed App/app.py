import os
import gdown
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

# -------------------- LOAD CSV --------------------
disease_info = pd.read_csv(r'D:\plant\Plant-Disease-Detection-main\Plant-Disease-Detection-main\Flask Deployed App\disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv(r'D:\plant\Plant-Disease-Detection-main\Plant-Disease-Detection-main\Flask Deployed App\supplement_info.csv', encoding='cp1252')

# -------------------- MODEL --------------------
model_path = "plant_disease_model_1_latest.pt"
GDRIVE_FILE_ID = "13o3rNbawnA8ZSgUDdu7Y3LFGvXHEYG3F"

# Download model if not exists
if not os.path.exists(model_path):
    print("Downloading model...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, model_path, quiet=False)

# Load model
model = CNN.CNN(39)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print("Model loaded successfully")

# -------------------- PREDICTION FUNCTION --------------------
def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))

    with torch.no_grad():
        output = model(input_data)

    probs = F.softmax(output, dim=1).numpy()[0]

    # Top 3 predictions
    top3 = probs.argsort()[-3:][::-1]
    confidence = float(np.max(probs)) * 100

    return top3, confidence, probs


# -------------------- FLASK --------------------
app = Flask(__name__)

# HOME
@app.route('/')
def home_page():
    return render_template('home.html')

# CONTACT
@app.route('/contact')
def contact():
    return render_template('contact-us.html')

# AI PAGE
@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

# MOBILE (optional)
@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

# -------------------- SUBMIT --------------------
@app.route('/submit', methods=['POST'])
def submit():

    image = request.files['image']

    # Save image
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, image.filename)
    image.save(file_path)

    # Prediction
    top3, confidence, probs = prediction(file_path)
    pred = top3[0]

    # Main disease info
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]

    # Supplement info
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    # -------- NEW FEATURES --------
    top3_names = [disease_info['disease_name'][i] for i in top3]
    top3_scores = [round(probs[i]*100, 2) for i in top3]

    return render_template(
        'submit.html',
        title=title,
        desc=description,
        prevent=prevent,
        image_url=image_url,
        pred=pred,
        sname=supplement_name,
        simage=supplement_image_url,
        buy_link=supplement_buy_link,

        # NEW
        confidence=round(confidence, 2),
        top3_names=top3_names,
        top3_scores=top3_scores
    )


# -------------------- MARKET --------------------
@app.route('/market')
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )


# -------------------- RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
import os
import json
import gdown
import torch
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN

# -------------------- PATH SETUP --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- FILE PATHS --------------------
disease_path = os.path.join(BASE_DIR, "disease_info.csv")
supplement_path = os.path.join(BASE_DIR, "supplement_info.csv")
model_path = os.path.join(BASE_DIR, "plant_disease_model_1_latest.pt")

# -------------------- LOAD CSV --------------------
disease_info = pd.read_csv(disease_path, encoding='cp1252')
supplement_info = pd.read_csv(supplement_path, encoding='cp1252')

# -------------------- DOWNLOAD MODEL IF NOT EXISTS --------------------
if not os.path.exists(model_path):
    print("⬇️ Downloading model...")
    url = "https://drive.google.com/uc?export=download&id=13o3rNbawnA8ZSgUDdu7Y3LFGvXHEYG3F"
    gdown.download(url, model_path, quiet=False)

# -------------------- LOAD MODEL --------------------
model = CNN.CNN(39)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# -------------------- PREDICTION FUNCTION --------------------
def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.unsqueeze(0)

    with torch.no_grad():
        output = model(input_data)
        pred = torch.argmax(output, dim=1).item()

    return pred

# -------------------- FLASK APP --------------------
app = Flask(__name__)

# -------------------- ROUTES --------------------
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']

        upload_folder = os.path.join(BASE_DIR, "static/uploads")
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, image.filename)
        image.save(file_path)

        pred = prediction(file_path)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]

        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
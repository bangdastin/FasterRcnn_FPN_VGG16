from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import random

app = Flask(__name__, static_folder="static")
CORS(app)

model_path = os.path.expanduser("C:/Users/Lenovo/Documents/TA-2025-01/Website/model/fasterrcnn_vgg16_fpn.pth")
model = fasterrcnn_resnet50_fpn(weights=None) 
model.load_state_dict(torch.load(model_path))  
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    draw = ImageDraw.Draw(image)
    cars_detected = 0

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        print(f"Label: {label.item()}, Score: {score.item()}")
        if score > 0.5 and label.item() == 3:
            cars_detected += 1
            box = [int(coord) for coord in box]
            draw.rectangle(box, outline='red', width=3)
            draw.text((box[0], box[1] - 10), f"Car {score:.2f}", fill='red')

    result_path = os.path.join("static", "result.jpg")
    image.save(result_path)

    cache_buster = random.randint(0, 100000)

    return jsonify({
        'cars_detected': cars_detected,
        'image_url': f'/static/result.jpg?cb={cache_buster}'
    })

if __name__ == '__main__':
    app.run(debug=True)

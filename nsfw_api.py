from flask import Flask, request, jsonify
from PIL import Image
import torch
import timm
import io

app = Flask(__name__)

# Load Marqo NSFW Detection Model
model = timm.create_model('hf_hub:Marqo/nsfw-image-detection-384', pretrained=True)
model.eval()

# Get model label names (usually ['SFW', 'NSFW'])
class_names = model.pretrained_cfg['label_names']

# Image preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

@app.route('/detect', methods=['POST'])
def detect_nsfw():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    img_bytes = request.files['image'].read()
    input_tensor = preprocess_image(img_bytes)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0].tolist()

    result = {label: float(prob) for label, prob in zip(class_names, probabilities)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import torch

app = Flask(__name__)

# Model cache
processor = None
model = None
device = "cpu"


def load_model():
    global processor, model

    if processor is None or model is None:
        print("Loading BLIP model...")

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device) # type: ignore
        model.eval()

        print("Model loaded successfully.")


@app.route('/analyze', methods=['POST'])
def analyze_image():

    load_model()

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        image_file = request.files['image']
        image = Image.open(
            io.BytesIO(image_file.read())
        ).convert('RGB')

        inputs = processor(image, return_tensors="pt") # type: ignore
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_length=50) # type: ignore

        caption = processor.decode( # type: ignore
            out[0],
            skip_special_tokens=True
        )

        return jsonify({'description': caption})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)
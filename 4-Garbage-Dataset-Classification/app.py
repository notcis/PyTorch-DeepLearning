import json
from pathlib import Path

import gradio as gr
from PIL import Image
import torch
from torchvision import transforms

from model import create_model

DEPLOY_DIR = Path(__file__).parent

# Load class names
with open(DEPLOY_DIR / "class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Device & Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=len(class_names))
weights_path = DEPLOY_DIR / "weights" / "model_weights.pt"

if not weights_path.exists():
    raise FileNotFoundError(f"Model weights not found at {weights_path}. Verify Git LFS pull.")

model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# Transforms matching EfficientNet-B2 default evaluation preset (size 288)
transform = transforms.Compose([
    transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image: Image.Image):
    if image is None:
        return {}
    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    return {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

# Sample examples
example_images = [[str(p)] for p in sorted((DEPLOY_DIR / "assets").glob("*.jpg"))]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Trash Image"),
    outputs=gr.Label(num_top_classes=len(class_names), label="Predictions"),
    examples=example_images if example_images else None,
    title="🗑️ Garbage Dataset Classification",
    description="Classify waste into cardboard, glass, metal, paper, plastic, or trash using EfficientNet-B2.",
)

if __name__ == "__main__":
    demo.launch()
---
name: gradio-interface-builder
description: Builds interactive Gradio web interfaces for PyTorch/Deep Learning models, including image classification, object detection, text inference, and custom data processing with modern Blocks/Interface APIs.
---

# Gradio Interface Builder Guidelines

When generating web demo interfaces with `gradio` for machine learning models (PyTorch/Lightning, Hugging Face, or Computer Vision pipelines), follow these standard practices:

1. **Modern Layout Architecture**: Use `gr.Blocks()` for full control over layouts, columns, and event listeners (prefer `gr.Blocks()` over basic `gr.Interface` when building custom apps).
2. **Inference Function Encapsulation**:
   - Keep preprocessing, forward pass (`torch.no_grad()`), and postprocessing clearly separated.
   - Return clean outputs suitable for Gradio components (e.g., `Dict[str, float]` for `gr.Label`, `PIL.Image` or `np.ndarray` for `gr.Image`).
3. **Component Standards**:
   - Inputs: Set `type="pil"` or `type="numpy"` for `gr.Image`.
   - Outputs: Use `gr.Label(num_top_classes=K)` for classification confidence scores.
   - UI Controls: Include clear examples (`gr.Examples`), clear buttons, and submit buttons.
4. **Device Agnostic Execution**: Ensure input tensors are transferred to the correct device (`cuda`, `mps`, or `cpu`) and moved back to `cpu` for display.

---

## Standard Reference Implementation (Computer Vision / Image Classification)

```python
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional

# 1. Device Setup & Transform Definition
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_gradio_app(
    model: nn.Module,
    class_names: List[str],
    title: str = "Deep Learning Model Demo",
    description: str = "Upload an image to get top class predictions.",
    examples: Optional[List[str]] = None
) -> gr.Blocks:
    """
    Builds a complete Gradio Blocks application for model inference.
    """
    model.to(DEVICE)
    model.eval()

    def predict(image: Optional[Image.Image]) -> Dict[str, float]:
        if image is None:
            return {}

        # Preprocessing
        img_rgb = image.convert("RGB")
        tensor = transform_pipeline(img_rgb).unsqueeze(0).to(DEVICE) # [1, C, H, W]

        # Model Inference
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        # Build dictionary of {class_name: probability}
        confidences = {
            class_names[i]: float(probabilities[i].item())
            for i in range(len(class_names))
        }
        return confidences

    # Build Gradio UI with gr.Blocks
    with gr.Blocks(theme=gr.themes.Soft(), title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                with gr.Row():
                    btn_clear = gr.ClearButton(components=[input_image], value="Clear")
                    btn_submit = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                output_label = gr.Label(num_top_classes=5, label="Top Predictions")

        # Event handling
        btn_submit.click(fn=predict, inputs=input_image, outputs=output_label)
        input_image.change(fn=predict, inputs=input_image, outputs=output_label)

        if examples:
            gr.Examples(examples=examples, inputs=input_image)

    return demo

# Example execution entrypoint:
# if __name__ == "__main__":
#     # app = create_gradio_app(model, class_names=['Cat', 'Dog'])
#     # app.launch(share=False)
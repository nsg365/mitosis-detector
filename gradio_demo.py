#!/usr/bin/env python3
"""
Gradio Demo: Upload a 512x512 patch image → See mitosis detections with bounding boxes live
"""
import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from src.stage2_detector import build_detector, DEVICE

# Load model
print("Loading Stage 2 detector...")
model = build_detector()
ckpt = torch.load("checkpoints/stage2_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE).eval()

SCORE_THRESH = 0.4

def detect_mitosis(image_pil):
    """
    Input: PIL Image (512x512 patch)
    Output: Annotated image with bounding boxes
    """
    if image_pil is None:
        return None, "No image uploaded"
    
    # Convert to tensor
    img_np = np.array(image_pil).astype(np.float32) / 255.0
    if len(img_np.shape) == 2:  # Grayscale
        img_np = np.stack([img_np] * 3, axis=-1)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get predictions
    boxes = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    
    # Filter by threshold
    mask = scores >= SCORE_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    
    # Draw on image
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)
    
    mitosis_count = 0
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{score:.3f}', color='red', fontsize=10, weight='bold')
        mitosis_count += 1
    
    ax.set_title(f"Mitosis Detection: {mitosis_count} detected", fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to PIL
    fig.canvas.draw()
    img_out = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    
    info_text = f"✓ Detections: {mitosis_count}\n✓ Score threshold: {SCORE_THRESH}\n✓ Model: Faster R-CNN ResNet50-FPN (epoch 65, mAP@0.5=0.7598)"
    
    return img_out, info_text

# Gradio interface
title = "🔬 Mitosis Detection Demo"
description = """
**Upload a 512×512 histopathology patch** → The model detects mitotic figures with bounding boxes.

**Model Details:**
- Architecture: Faster R-CNN with ResNet50-FPN backbone
- Training: 100 epochs on TUPAC16 data
- Best checkpoint: Epoch 65, mAP@0.5 = 0.7598
- F1 Score: 0.7742 (77.42%)
- Validation set: 11 patches with 16 ground-truth mitoses

**How to use:**
1. Upload or paste a patch image (PNG/JPG)
2. Model will detect mitoses and show confidence scores
3. Red boxes = detected mitotic figures
"""

demo = gr.Interface(
    fn=detect_mitosis,
    inputs=gr.Image(type="pil", label="Upload 512×512 Histopathology Patch"),
    outputs=[
        gr.Image(type="pil", label="Detection Output"),
        gr.Textbox(label="Detection Summary", lines=4)
    ],
    title=title,
    description=description,
    examples=[],
    allow_flagging="never"
)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRADIO DEMO: Mitosis Detection")
    print("="*70)
    print("\n✓ Model loaded successfully")
    print("✓ Launching Gradio interface...")
    print("\nAccess the demo at: http://localhost:7860")
    print("\n" + "="*70 + "\n")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

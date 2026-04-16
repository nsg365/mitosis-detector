import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from src.stage2_detector import build_detector, DEVICE
from robust_inference import preprocess_flexible_size, unmap_boxes

# Load model
print("Loading Stage 2 detector...")
model = build_detector()
ckpt = torch.load("checkpoints/stage2_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE).eval()

SCORE_THRESH = 0.4

def detect_mitosis(image_pil):
    if image_pil is None:
        return None, "Please upload an image"
    
    original_size = image_pil.size
    
    img_tensor, metadata = preprocess_flexible_size(image_pil, return_metadata=True)
    img_tensor = img_tensor.to(DEVICE)
    img_np = img_tensor.cpu().numpy()
    
    with torch.no_grad():
        outputs = model([img_tensor])
    
    boxes_padded = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    
    mask = scores >= SCORE_THRESH
    boxes_padded = boxes_padded[mask]
    scores = scores[mask]
    
    if len(boxes_padded) > 0:
        boxes_original = unmap_boxes(boxes_padded, metadata)
    else:
        boxes_original = np.empty((0, 4))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np.transpose(1, 2, 0)) 
    
    mitosis_count = 0
    for box, score in zip(boxes_padded, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{score:.3f}', color='red', fontsize=10, weight='bold')
        mitosis_count += 1
    
    ax.set_title(f"Mitosis Detection: {mitosis_count} detected", fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_out = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
    img_out = img_out.convert('RGB')
    plt.close(fig)
    
    info_text = f"""✓ Detections: {mitosis_count}
 Score threshold: {SCORE_THRESH}
 Model: Faster R-CNN ResNet50-FPN (epoch 65, mAP@0.5=0.7598)
 Input: {original_size[0]}×{original_size[1]} → 512×512 (padded)
 Robustness: ANY image size without retraining"""
    
    return img_out, info_text


# Gradio interface
title = "Mitosis Detection Demo"
description = """
**Upload an histopathology image (preferably from TUPAC16 dataset)** → The model detects mitotic figures with bounding boxes.

**Model Details:**
- Architecture: Faster R-CNN with ResNet50-FPN backbone
- Training: 100 epochs on TUPAC16 data
- Best checkpoint: Epoch 65, mAP@0.5 = 0.7598
- F1 Score: 0.7742 (77.42%)
"""

demo = gr.Interface(
    fn=detect_mitosis,
    inputs=gr.Image(type="pil", label="Upload Image (ANY size)"),
    outputs=[
        gr.Image(type="pil", label="Detection Output"),
        gr.Textbox(label="Detection Summary", lines=5)
    ],
    title=title,
    description=description,
    examples=[]
)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRADIO DEMO: Mitosis Detection (Robust to Any Size)")
    print("="*70)
    print("\n✓ Model loaded successfully")
    print("✓ Launching Gradio interface...")
    print("\nAccess the demo at: http://localhost:7861")
    print("\nTest with:")
    print("  - 64×64 image (tiny)")
    print("  - 512×512 image (standard)")
    print("  - 1024×1024 image (large)")
    print("  - 800×600 image (non-square)")
    print("  - Any other size!")
    print("\n" + "="*70 + "\n")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)


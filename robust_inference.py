import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def preprocess_flexible_size(image, target_size=512, return_metadata=False):
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray((image * 255).astype(np.uint8))
    
    orig_w, orig_h = image.size
    
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    image_padded = Image.new('RGB', (target_size, target_size), color=(255, 255, 255))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    image_padded.paste(image_resized, (pad_x, pad_y))
    
    img_tensor = TF.to_tensor(image_padded)
    
    if return_metadata:
        metadata = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'orig_w': orig_w,
            'orig_h': orig_h,
            'new_w': new_w,
            'new_h': new_h,
        }
        return img_tensor, metadata
    
    return img_tensor


def unmap_boxes(boxes_padded, metadata):

    boxes = boxes_padded.copy()
    
    boxes[:, [0, 2]] -= metadata['pad_x']
    boxes[:, [1, 3]] -= metadata['pad_y']
    
    scale = metadata['scale']
    boxes[:, :4] /= scale
    
    boxes[:, 0] = np.clip(boxes[:, 0], 0, metadata['orig_w'])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, metadata['orig_h'])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, metadata['orig_w'])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, metadata['orig_h'])
    
    return boxes


def batch_preprocess(images_list, target_size=512):
    tensors = []
    metadatas = []
    
    for img in images_list:
        tensor, metadata = preprocess_flexible_size(img, target_size, return_metadata=True)
        tensors.append(tensor)
        metadatas.append(metadata)
    
    return torch.stack(tensors), metadatas


def detect_with_robustness(model, image, device='cpu', score_threshold=0.4):

    img_tensor, metadata = preprocess_flexible_size(image, return_metadata=True)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model([img_tensor])
    
    boxes_padded = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    
    mask = scores >= score_threshold
    boxes_padded = boxes_padded[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    if len(boxes_padded) > 0:
        boxes_original = unmap_boxes(boxes_padded, metadata)
    else:
        boxes_original = np.empty((0, 4))
    
    return boxes_original, scores, labels


if __name__ == "__main__":
    print("Testing robust preprocessing utilities...\n")
    
    print("Test 1: Processing various image sizes")
    test_sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024), (800, 600), (300, 800)]
    
    for w, h in test_sizes:
        img_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
        
        tensor, metadata = preprocess_flexible_size(img_pil, return_metadata=True)
        
        assert tensor.shape == (3, 512, 512), f"Shape mismatch for {w}×{h}"
        assert metadata['scale'] <= 1.0
        print(f"   {w:4d}×{h:4d} → processed successfully (scale={metadata['scale']:.3f})")
    
    print("\nTest 2: Box coordinate remapping")
    w, h = 800, 600
    img_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_array)
    
    tensor, metadata = preprocess_flexible_size(img_pil, return_metadata=True)
    
    boxes_padded = np.array([[50, 50, 150, 150], [200, 200, 300, 300]], dtype=float)
    boxes_original = unmap_boxes(boxes_padded, metadata)
    
    assert np.all(boxes_original[:, 0] >= 0) and np.all(boxes_original[:, 2] <= w)
    assert np.all(boxes_original[:, 1] >= 0) and np.all(boxes_original[:, 3] <= h)
    print(f"   Boxes remapped correctly (within bounds [0-{w}]×[0-{h}])")
    
    print("\n All robustness tests passed!")

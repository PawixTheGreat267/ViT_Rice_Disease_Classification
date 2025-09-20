from PIL import Image
from transformers import ViTImageProcessor, ViTForMaskedImageModeling
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Load pre-trained processor and model
# -----------------------------
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
model = ViTForMaskedImageModeling.from_pretrained("facebook/vit-mae-base")

# -----------------------------
# 2. Load sample image (cat)
# -----------------------------
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
print(f"Original image size: {image.size}")

# -----------------------------
# 3. Make an incomplete version (black rectangle)
# -----------------------------
incomplete_image = image.resize((224, 224))
incomplete_np = np.array(incomplete_image).copy()

# Black out a region (simulate missing info)
incomplete_np[60:180, 100:200, :] = 0  
incomplete_image = Image.fromarray(incomplete_np)

# -----------------------------
# 4. Function to mask + reconstruct
# -----------------------------
def reconstruct(img, mask_ratio=0.4):
    # Preprocess
    inputs = processor(images=img, return_tensors="pt")

    # Random patch mask
    num_patches = 14 * 14
    num_masked = int(mask_ratio * num_patches)
    mask_flat = np.hstack([
        np.ones(num_masked, dtype=bool),
        np.zeros(num_patches - num_masked, dtype=bool)
    ])
    np.random.shuffle(mask_flat)
    mask = mask_flat.reshape(14, 14)

    inputs["bool_masked_pos"] = torch.tensor(mask.flatten()[None, :])

    # Forward pass
    outputs = model(**inputs)
    reconstructed = outputs.reconstruction

    # Denormalize for visualization
    recon = reconstructed[0].detach().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    recon = recon * std + mean
    recon = np.clip(recon, 0, 1)
    recon = np.transpose(recon, (1, 2, 0))
    return mask, recon

# -----------------------------
# 5. Run reconstruction
# -----------------------------
mask1, recon1 = reconstruct(image)
mask2, recon2 = reconstruct(incomplete_image)

# -----------------------------
# 6. Visualization
# -----------------------------
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Row 1 → Original image case
axs[0,0].imshow(image.resize((224,224)))
axs[0,0].set_title("Original Image")
axs[0,0].axis("off")

masked_img1 = np.array(image.resize((224,224))).copy()
for i in range(14):
    for j in range(14):
        if mask1[i,j]:
            y1, y2 = i*16, (i+1)*16
            x1, x2 = j*16, (j+1)*16
            masked_img1[y1:y2, x1:x2, :] = 0
axs[0,1].imshow(masked_img1)
axs[0,1].set_title("Masked (Original)")
axs[0,1].axis("off")

axs[0,2].imshow(recon1)
axs[0,2].set_title("Reconstructed (Original)")
axs[0,2].axis("off")

# Row 2 → Incomplete image case
axs[1,0].imshow(incomplete_image)
axs[1,0].set_title("Incomplete Image")
axs[1,0].axis("off")

masked_img2 = np.array(incomplete_image).copy()
for i in range(14):
    for j in range(14):
        if mask2[i,j]:
            y1, y2 = i*16, (i+1)*16
            x1, x2 = j*16, (j+1)*16
            masked_img2[y1:y2, x1:x2, :] = 0
axs[1,1].imshow(masked_img2)
axs[1,1].set_title("Masked (Incomplete)")
axs[1,1].axis("off")

axs[1,2].imshow(recon2)
axs[1,2].set_title("Reconstructed (Incomplete)")
axs[1,2].axis("off")

plt.tight_layout()
plt.show()

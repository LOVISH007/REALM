import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load CLIP model and processor
CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
CLIP_MODEL.eval()

def preprocess_image(image) -> torch.Tensor:
    """Preprocess image for CLIP input."""
    return CLIP_PROCESSOR(images=image, return_tensors="pt")["pixel_values"]


def get_text_embedding(text: str) -> torch.Tensor:
    """Get text embedding using CLIP."""
    with torch.no_grad():
        inputs = CLIP_PROCESSOR(text=[text], return_tensors="pt", padding=True, truncation=True)
        text_features = CLIP_MODEL.get_text_features(**inputs)
        return F.normalize(text_features, p=2, dim=-1)


def get_patch_embedding(patch):
    """Get normalized embedding of an image patch using CLIP.
    Args:
        patch (PIL.Image): Image patch.
    
    Returns:
        torch.Tensor: Normalized embedding of the image patch.
    """
    pixel_values = preprocess_image(patch)
    with torch.no_grad():
        image_features = CLIP_MODEL.get_image_features(pixel_values)
        return F.normalize(image_features, p=2, dim=-1)


def generate_heat_map(image: Image, text: str, window_size:int = 64, stride:int = 32) -> np.ndarray:
    """
    Generate heatmap of similarity between image patches and a text description.

    Args:
        image (PIL.Image): Input image.
        text (str): Text to compare with image patches.
        window_size (int): Size of the sliding window.
        stride (int): Stride of the sliding window.
    
    Returns:
        heatmap (numpy.ndarray): Heatmap of similarity scores.
    """
    img_w, img_h = image.size
    heatmap = np.zeros((img_h, img_w), dtype=np.float32)
    count_map = np.zeros((img_h, img_w), dtype=np.float32)

    text_embedding = get_text_embedding(text)

    # Slide window over image
    for top in range(0, img_h - window_size + 1, stride):
        for left in range(0, img_w - window_size + 1, stride):
            patch = image.crop((left, top, left + window_size, top + window_size))
            patch_embedding = get_patch_embedding(patch)
            
            # Cosine similarity
            score = torch.matmul(patch_embedding, text_embedding.T).item()

            # Add score to heatmap region
            heatmap[top:top+window_size, left:left+window_size] += score
            count_map[top:top+window_size, left:left+window_size] += 1

    # Normalize by count to get average scores
    heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map!=0)

    return heatmap


def to_pil_image(np_img: np.ndarray) -> Image.Image:
    """Convert a numpy array to a PIL Image.
    Args:
        np_img (numpy.ndarray): Input image as a numpy array.
    
    Returns:
        PIL.Image: Converted PIL Image.
    """
    if np_img.dtype != np.uint8:
        np_img = (np_img * 255).astype(np.uint8)
    if np_img.shape[-1] == 4:
        np_img = np_img[..., :3]  # Drop alpha channel if exists
    return Image.fromarray(np_img)


def plot_and_save_heatmap(image: Image, text: str, save_path: str, window_size:int = 64, stride:int = 8) -> np.ndarray:
    """Generate and save heatmap overlayed on the original image.
    
    Args:
        image (PIL.Image): Input image.
        text (str): Text to compare with image patches.
        save_path (str): Path to save the heatmap image.
        window_size (int): Size of the sliding window.
        stride (int): Stride of the sliding window.

    Returns:
        np.ndarray: Generated heatmap.
    """
    heatmap = generate_heat_map(image, text, window_size, stride)
    
    # Normalize heatmap to [0, 1]
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() if heatmap.max() != 0 else 1

    # Convert heatmap to PIL Image
    heatmap_img = to_pil_image(heatmap)

    # Overlay heatmap on original image
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='hot', alpha=0.5)
    plt.title("Text Similarity Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    plt.close()

    print(f"Heatmap saved to {save_path}")
    return heatmap

def main():
    # Check if correct number of arguments provided
    if len(sys.argv) != 4:
        print("Usage: python get_unrealism_heatmaps.py <image_path> <text_description> <save_path>")
        sys.exit(1)
    
    # Get arguments from command line
    image_path = sys.argv[1]
    text_description = sys.argv[2]
    save_path = sys.argv[3]

    image = Image.open(image_path).convert("RGB")
    plot_and_save_heatmap(image, text_description, save_path)

if __name__ == "__main__":
    main()

# To predict the output run the following command:
              
# python predict.py --image_dir test --model_path Trained_Model.pth --csv_output predictions.csv --json_output predictions.json


import os
import csv
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from scipy.optimize import linear_sum_assignment


class TileTransformer(nn.Module):
    """
    Defines the model architecture. Must be identical to the
    class used during training to load the state_dict.
    """
    def __init__(self, tile_encoder='resnet18', embed_dim=512, nhead=8, nlayers=3, dropout=0.1):
        super().__init__()
        res = models.resnet18(pretrained=False) # Set pretrained=False for inference if not needed
        res.fc = nn.Identity()
        self.backbone = res  # outputs 512-d for resnet18
        self.embed_dim = 512
        if self.embed_dim != embed_dim:
            self.proj = nn.Linear(self.embed_dim, embed_dim)
            final_dim = embed_dim
        else:
            self.proj = None
            final_dim = self.embed_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=final_dim, nhead=nhead, dropout=dropout, dim_feedforward=final_dim*4, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Linear(final_dim, 9)

    def forward(self, tiles):
        B = tiles.shape[0]
        n = tiles.shape[1]  # 9
        tiles_flat = tiles.view(B*n, tiles.shape[2], tiles.shape[3], tiles.shape[4])
        x = self.backbone(tiles_flat)  # (B*n, 512)
        if self.proj is not None:
            x = self.proj(x)  # (B*n, embed_dim)
        x = x.view(B, n, -1)  # (B, 9, D)
        # transformer expects seq_len, batch, dim
        x_t = x.permute(1, 0, 2)
        x_out = self.transformer(x_t)  
        x_out = x_out.permute(1, 0, 2)  
        logits = self.head(x_out)  #  per tile logits over positions
        return logits

def hungarian_from_logits(logits):

    if isinstance(logits, torch.Tensor):
        arr = logits.detach().cpu().numpy()
    else:
        arr = np.array(logits)
    cost = -arr
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = [-1]*arr.shape[0]
    for r,c in zip(row_ind, col_ind):
        perm[r] = int(c)
    return perm

def infer_single_image(model, image_path, image_size=201, device='cpu'):

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img = np.array(img).astype(np.uint8)
    except Exception as e:
        print(f"Error opening or resizing image {image_path}: {e}")
        return None

    cut = image_size // 3
    tiles = []
    
    for i in range(3):
        for j in range(3):
            tile = img[i*cut:(i+1)*cut, j*cut:(j+1)*cut, :]
            tile_img = Image.fromarray(tile)
            tiles.append(to_tensor(tile_img))
            
    tiles = torch.stack(tiles).unsqueeze(0).to(device)  # (1, 9, C, H, W)
    
    model.eval()
    with torch.no_grad():
        logits = model(tiles)[0]  # (9, 9)
        perm = hungarian_from_logits(logits)
        
    return perm


def run_inference(model_path, image_dir, csv_output_path, json_output_path, tile_size, force_cpu):
    """
    Main function to run inference on a directory of images.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    print(f"Using device: {device}")

    image_size = tile_size * 3
    print(f"Using tile size: {tile_size}, full image size: {image_size}x{image_size}")

    print("Loading model...")
    model = TileTransformer(
        tile_encoder='resnet18', 
        embed_dim=512, 
        nhead=8, 
        nlayers=3
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture in this script matches the one used for training.")
        return
        
    model.eval()
    print("Model loaded successfully.")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(image_dir) 
        if os.path.splitext(f)[1].lower() in image_extensions
    ])
    
    if not image_files:
        print(f"No images found in directory: {image_dir}")
        return

    print(f"Found {len(image_files)} images to process...")

    all_results = []
    for filename in tqdm(image_files, desc="Predicting"):
        image_path = os.path.join(image_dir, filename)
        
        perm_list = infer_single_image(model, image_path, image_size=image_size, device=device)
        
        if perm_list:
            perm_string = " ".join(str(p) for p in perm_list)
            
            all_results.append({
                "filename": filename,
                "sequence_list": perm_list,
                "sequence_string": perm_string
            })
        else:
            print(f"Skipping corrupt or unreadable file: {filename}")

    print(f"Saving CSV results to {csv_output_path}...")
    try:
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "sequence"])
            for res in all_results:
                writer.writerow([res["filename"], res["sequence_string"]])
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    print(f"Saving JSON results to {json_output_path}...")
    json_data = {
        "images": [
            {
                "filename": res["filename"],
                "sequence": res["sequence_list"]
            } for res in all_results
        ]
    }
    
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        
    print("\nInference complete.")
    print(f"CSV saved to: {csv_output_path}")
    print(f"JSON saved to: {json_output_path}")

# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jigsaw Puzzle Inference Script")
    
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing images for inference."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the saved model file (e.g., 'best_jigsaw.pth')."
    )
    parser.add_argument(
        "--csv_output", 
        type=str, 
        default="predictions.csv", 
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--json_output", 
        type=str, 
        default="predictions.json", 
        help="Path to save the output JSON file."
    )
    parser.add_argument(
        "--tile_size", 
        type=int, 
        default=67, 
        help="Tile size used during training. (Default: 67, as in your script)"
    )
    parser.add_argument(
        "--force_cpu", 
        action='store_true', 
        help="Force use of CPU even if CUDA is available."
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        image_dir=args.image_dir,
        csv_output_path=args.csv_output,
        json_output_path=args.json_output,
        tile_size=args.tile_size,
        force_cpu=args.force_cpu
    )
"""
train.py — Lightweight few-shot fine-tuning of Grounding DINO on CPU.
Updates the model to better detect NEU defect classes.
"""

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import config

# Configuration
# ---------------------------------------------------------------------
NUM_SHOTS = 20           
EPOCHS = 5               
BATCH_SIZE = 1           
GRAD_ACCUM_STEPS = 4     
LEARNING_RATE = 5e-5     
MAX_SIZE = 480           # Very small for CPU speed
OUTPUT_DIR = Path("fine_tuned_model")
DATASET_DIR = Path("NEU-Surface-Defect-Dataset-1/valid")
ANNO_FILE = DATASET_DIR / "_annotations.coco.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical Class List (Must match config.PROMPT_TEXT order/content)
CLASSES = ["crazing", "inclusion", "patches", "pitted surface", "rolled-in scale", "scratches"]
PROMPT = config.PROMPT_TEXT  # "crazing. inclusion. patches. ..."

def get_token_ids_for_classes(processor, prompt, classes):
    """
    Finds the token ID(s) for each class name in the prompt.
    Returns a dict: { "crazing": [id], "inclusion": [id], ... }
    """
    inputs = processor(text=[prompt], return_tensors="pt")
    input_ids = inputs.input_ids[0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    
    # Simple heuristic: find the sequence of tokens matching the class name
    # This keeps it simple. Grounding DINO tokenizer usually preserves whole words 
    # or splits them.
    
    class_token_ids = {}
    for cls_name in classes:
        # We assume the class name appears exactly once in the prompt
        # Tokenize the class name alone to see what parts it has
        cls_tokens = processor.tokenizer(cls_name, add_special_tokens=False).input_ids
        
        # Find this sequence in the full prompt input_ids
        found = False
        for i in range(len(input_ids) - len(cls_tokens) + 1):
            if input_ids[i : i + len(cls_tokens)] == cls_tokens:
                # Found match. 
                # Grounding DINO expects the index of the token in input_ids
                # We can verify which token indices correspond to this class.
                # Just take the first token index for now (often sufficient for single-token words)
                # or the list of all indices for that word? 
                # The model expects a single class label per box usually if doing Softmax, 
                # but Grounding DINO does sigmoid per token. 
                # It likely expects simply the index of the token that is "positive".
                class_token_ids[cls_name] = i 
                found = True
                break
        
        if not found:
            logger.warning(f"Could not find tokens for class '{cls_name}' in prompt!")
            class_token_ids[cls_name] = -1
            
    return class_token_ids

class FewShotDataset(Dataset):
    def __init__(self, images, coco_data):
        self.images = images
        self.coco = coco_data
        self.img_to_anns = defaultdict(list)
        for ann in coco_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
        self.cat_id_to_name = {c['id']: c['name'] for c in coco_data['categories']}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = DATASET_DIR / img_info['file_name']
        image = Image.open(img_path).convert("RGB")
        
        # Resize logic is handled by processor usually, but we force small size here
        # to save loading time/memory, resizing PIL before converting to tensor
        w, h = image.size
        scale = MAX_SIZE / max(w, h)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h))
            
        return img_info, image

def run_training():
    logger.info("Initializing Few-Shot Training...")
    
    # 1. Prepare Data
    with open(ANNO_FILE) as f:
        coco = json.load(f)
        
    cat_map_name = {c['id']: c['name'] for c in coco['categories']} # Original names "crazing", "rolled-in_scale"
    
    # Group useful images
    valid_images = []
    # Only keep images that have annotations of interest
    img_ids_with_anns = set()
    for ann in coco['annotations']:
        if cat_map_name[ann['category_id']] != 'surface-defects':
            img_ids_with_anns.add(ann['image_id'])
            
    images_by_cat = defaultdict(list)
    for img in coco['images']:
        if img['id'] not in img_ids_with_anns: continue
        
        # Find dominant cat
        anns = [a for a in coco['annotations'] if a['image_id'] == img['id']]
        if not anns: continue
        c_name = cat_map_name[anns[0]['category_id']]
        if c_name == 'surface-defects': continue
        
        images_by_cat[c_name].append(img)
        
    # Sample 20
    train_images = []
    logger.info(f"Sampling {NUM_SHOTS} images per class:")
    for cat in CLASSES: # Iterate our canonical list
        # Map canonical "rolled-in scale" to COCO "rolled-in_scale" or "rolled-in scale"
        # In COCO it is "rolled-in_scale" (with underscore)
        # In PROMPT it is "rolled-in scale" (with space)
        # We need to robustly find it.
        
        # Search for key in images_by_cat
        found_key = None
        for k in images_by_cat.keys():
            if k.replace("_", " ") == cat.replace("_", " "):
                found_key = k
                break
        
        if found_key:
            pool = images_by_cat[found_key]
            sample = random.sample(pool, min(len(pool), NUM_SHOTS))
            train_images.extend(sample)
            logger.info(f"  - {cat} (found as {found_key}): {len(sample)} images")
        else:
            logger.warning(f"  - {cat}: No images found?")
            
    random.shuffle(train_images)
    logger.info(f"Total training samples: {len(train_images)}")
    
    # 2. Load Model
    model = AutoModelForZeroShotObjectDetection.from_pretrained(config.MODEL_ID)
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    
    # Freeze
    for name, param in model.named_parameters():
        if "bbox_predictor" in name or "class_embed" in name:
            param.requires_grad = True # Train heads
        elif "enc_output" in name:
             param.requires_grad = True # Train intermediate projection
        else:
            param.requires_grad = False # Freeze backbone & bert
            
    logger.info("Model loaded. Backbones frozen.")
    
    # 3. Training Setup
    dataset = FewShotDataset(train_images, coco)
    
    # We will use simple class indices 0..5 corresponding to CLASSES list
    # The model might interpret these aligned with the phrases if we construct input correctly?
    # Actually, fine-tuning GroundingDino often requires 'class_labels' to be
    # the index into the provided text queries if we provided a list of phrases?
    # Or if we provided a single string, it expects token indices.
    # The previous error "index 17 out of bounds for dim 0 with size 6" suggests
    # the model MIGHT be seeing 6 classes (maybe from config.num_labels? or it parsed the prompt?).
    # Let's try 0..5.
    
    def collate_fn(batch):
        img_infos, pil_images = zip(*batch)
        
        # We pass the full prompt to every image
        texts = [PROMPT] * len(pil_images)
        inputs = processor(images=list(pil_images), text=texts, return_tensors="pt", padding=True)
        
        labels = []
        for i, info in enumerate(img_infos):
            anns = [a for a in coco['annotations'] if a['image_id'] == info['id']]
            w, h = pil_images[i].size
            
            boxes = []
            class_ids = []
            
            for ann in anns:
                cat_name = cat_map_name[ann['category_id']]
                # Normalized mapping
                # "rolled-in_scale" (COCO) -> "rolled-in scale" (CLASSES)
                c_norm = cat_name.replace("_", " ") 
                if c_norm == "pitted surface on steel": c_norm = "pitted surface" # Handle variants if any
                
                # Check for "crazing", "inclusion", ... in the name
                # Robust matching against CLASSES
                match_id = -1
                for idx, cls in enumerate(CLASSES):
                    if cls in c_norm:
                        match_id = idx
                        break
                
                if match_id != -1:
                    x, y, bw, bh = ann['bbox']
                    cx = (x + bw/2) / w
                    cy = (y + bh/2) / h
                    nw = bw / w
                    nh = bh / h
                    boxes.append([cx, cy, nw, nh])
                    class_ids.append(match_id)
            
            if boxes:
                # Ensure we cast to correct device/type later, but here create CPU tensors
                labels.append({
                    "class_labels": torch.tensor(class_ids, dtype=torch.long),
                    "boxes": torch.tensor(boxes, dtype=torch.float32)
                })
            else:
                labels.append({
                    "class_labels": torch.tensor([], dtype=torch.long),
                    "boxes": torch.tensor([], size=(0, 4), dtype=torch.float32)
                })
                
        return inputs, labels

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 4. Train
    # Only train heads for lightweight fine-tuning
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LEARNING_RATE)
    
    model.train()
    
    print("\nStarting training loop...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, (inputs, labels) in enumerate(progress):
            
            # Forward
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            if loss is None:
                # Some versions might return None if no loss computed?
                # Usually AutoModel returns loss if labels provided.
                logger.error("Model returned None loss!")
                continue
                
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            loss_val = loss.item() * GRAD_ACCUM_STEPS
            epoch_loss += loss_val
            progress.set_postfix({"loss": f"{loss_val:.4f}"})
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
    # 5. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info(f"Fine-tuned model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_training()

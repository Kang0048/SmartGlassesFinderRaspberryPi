import torch
import numpy as np
import shutil
import os
import cv2
from datetime import datetime

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def save_match_image(frame, folder='detected_matches', max_images=10):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"match_{timestamp}.jpg")
    cv2.imwrite(path, frame)

    # 삭제 로직
    images = sorted(os.listdir(folder))
    if len(images) > max_images:
        oldest = images[0]
        os.remove(os.path.join(folder, oldest))

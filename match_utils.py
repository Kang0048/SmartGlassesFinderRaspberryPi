import torch
import numpy as np
import shutil
import os
import cv2
from datetime import datetime
from captureThread import save_capture_and_vector
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

def make_vector_folder(objects_root : str, yolo, embedder, transform) :
     for obj_name in os.listdir(objects_root):
        label_dir = os.path.join(objects_root, obj_name)
        img_dir = os.path.join(label_dir,"images")
        number = 0
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir,img_file)
            img = cv2.imread(img_path)
            number +=1
            if img is None:
               print("cannot read picture")
               continue

            save_capture_and_vector(img, yolo, embedder, transform, label_dir, "test", number)

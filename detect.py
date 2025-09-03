import cv2
import torch
import time
import os
import numpy as np
from match_utils import cosine_similarity
from torchvision.models import resnet18
import torchvision.transforms as transforms
from firebase_uploader import upload_latest_image
import shutil
from PIL import Image
from collections import defaultdict

source_root = '/home/pi/yolo-object-matcher/objects'
target_root = '/home/pi/yolo-object-matcher/detected_matches'
REPO_DIR    = '/home/pi/yolo-object-matcher/yolov5'
WEIGHTS     = '/home/pi/yolo-object-matcher/yolov5n.pt'
IMG_SIZE    = 640
SIM_THR     = 0.75
cooldown_seconds = 10.0
upload_pic       = 60.0 


last_saved_time_by_cls    = defaultdict(float)  
last_detected_time_by_cls = defaultdict(float)  


def make_vec_list(vec_dir: str) -> list:
    new_ref_vecs = []
    for f in sorted(os.listdir(vec_dir)):
        if f.endswith(".pt"):
            v = torch.load(os.path.join(vec_dir, f), map_location="cpu")
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            new_ref_vecs.append(v.float())
    print(f"vector {len(new_ref_vecs)} load finished from {vec_dir}")
    return new_ref_vecs

def save_image_to_dir(frame_bgr, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    path = os.path.join(out_dir, f"{ts}.jpg")
    cv2.imwrite(path, frame_bgr)
    return path


embedder = resnet18(pretrained=True)
embedder.fc = torch.nn.Identity()
embedder.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# YOLOv5 (local)
yolo = torch.hub.load(REPO_DIR, 'custom', path=WEIGHTS, source='local')
yolo.conf = 0.25
yolo.iou  = 0.5
yolo.to('cpu').eval()


ref_by_class = {}  # {class_name: [Tensor, ...]}
for class_name in sorted(os.listdir(source_root)):
    
    class_dir = os.path.join(source_root, class_name)
    if not os.path.isdir(class_dir):
        continue
    class_vec_dir = os.path.join(class_dir, "vectors")
    if not os.path.isdir(class_vec_dir):
        continue
    vecs = make_vec_list(class_vec_dir)
    if vecs:
        ref_by_class[class_name] = vecs

print("loaded classes:", list(ref_by_class.keys()))


if os.path.exists(target_root):
    shutil.rmtree(target_root)
os.makedirs(target_root, exist_ok=True)
for class_name in ref_by_class.keys():
    os.makedirs(os.path.join(target_root, class_name), exist_ok=True)


cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_count += 1

    if frame_count % 10 == 0:
        with torch.inference_mode():
            res = yolo(frame, size=IMG_SIZE)
            det = res.xyxy[0].cpu().numpy() if hasattr(res, 'xyxy') else res.pred[0].cpu().numpy()

        if det is None or len(det) == 0:
            print("no box captured")
        else:
            for box in det:
                x1, y1, x2, y2, conf, cls = box.astype(int)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    q = embedder(transform(crop_rgb).unsqueeze(0)).squeeze().float()

              
                now_mono = time.monotonic()
                for cls_name, cls_ref_vecs in ref_by_class.items():
                    if not cls_ref_vecs:
                        continue
                    sims = [cosine_similarity(q, ref) for ref in cls_ref_vecs]
                    best_sim = max(sims) if sims else -1.0

                    print(f"[DEBUG] {cls_name} similarity: {best_sim:.2f}")

                    if best_sim >= SIM_THR and (now_mono - last_saved_time_by_cls[cls_name] > cooldown_seconds):
                        out_dir = os.path.join(target_root, cls_name)
                        save_image_to_dir(frame, out_dir)
                        last_saved_time_by_cls[cls_name]    = now_mono
                        last_detected_time_by_cls[cls_name] = now_mono
                        print(f"[SAVE] image saved -> {cls_name}")

   
    now_mono = time.monotonic()
    for cls_name, t_last in list(last_detected_time_by_cls.items()):
        if t_last > 0 and (now_mono - t_last) > upload_pic:
            cls_dir = os.path.join(target_root, cls_name)
            print(f"[{cls_name}] 1min over -> upload")

            try:
                upload_latest_image(cls_dir)  
               
                if os.path.isdir(cls_dir):
                    for f in os.listdir(cls_dir):
                        fp = os.path.join(cls_dir, f)
                        if os.path.isfile(fp):
                            os.remove(fp)
                print(f"[{cls_name}] uploaded & cleared")
            except Exception as e:
                print(f"[WARN] upload failed for {cls_name}: {e}")
            finally:
                last_detected_time_by_cls[cls_name] = 0.0

    #cv2.imshow("YOLO Matching", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

cap.release()
cv2.destroyAllWindows()

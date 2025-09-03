import os
import cv2
import torch
import time
import shutil
from collections import defaultdict
from match_utils import cosine_similarity
from firebase_uploader import upload_latest_image
from torchvision.models import resnet18
import torchvision.transforms as transforms

def make_vec_list(vec_dir: str) -> list:
    new_ref_vecs = []
    for f in sorted(os.listdir(vec_dir)):
        if f.endswith(".pt"):
            v = torch.load(os.path.join(vec_dir, f), map_location="cpu")
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            new_ref_vecs.append(v.float())
    print(f"[INFO] vector {len(new_ref_vecs)} loaded from {vec_dir}")
    return new_ref_vecs

def make_ref_by_class(source_root: str) -> dict:
    ref_by_class = {}
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
    return ref_by_class

def save_image_to_dir(frame_bgr, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    path = os.path.join(out_dir, f"{ts}.jpg")
    cv2.imwrite(path, frame_bgr)
    return path

# -------------------------
# Detection Loop (Thread)
# -------------------------
def detection_loop(yolo, embedder, transform, target_root, source_root, cap, camera_lock, pause_event):
  
    IMG_SIZE = 640
    SIM_THR = 0.75
    cooldown_seconds = 10.0
    upload_pic = 60.0

    last_saved_time_by_cls = defaultdict(float)
    last_detected_time_by_cls = defaultdict(float)
    folder_initialized = False 

    print("[THREAD] detection loop started")
    frame_count = 0
    while True:
        if pause_event.is_set():
            folder_initialized = False
            time.sleep(0.05)
            continue

        if not folder_initialized:
            if os.path.exists(target_root):
                shutil.rmtree(target_root)
            os.makedirs(target_root, exist_ok=True)
            local_ref_by_class = make_ref_by_class(source_root)
            for cls_name in local_ref_by_class.keys():
                os.makedirs(os.path.join(target_root, cls_name), exist_ok=True)
            
            folder_initialized = True
            print("[YOLO Thread] Target folders initialized")

        with camera_lock:
            ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % 10 == 0:
            with torch.inference_mode():
                res = yolo(frame, size=IMG_SIZE)
                det = res.xyxy[0].cpu().numpy() if hasattr(res, "xyxy") else res.pred[0].cpu().numpy()

            if det is None or len(det) == 0:
                print("no box captured",flush=True)
                continue

            for box in det:
                x1, y1, x2, y2, conf, cls = box.astype(int)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    q = embedder(transform(crop_rgb).unsqueeze(0)).squeeze().float()

                now_mono = time.monotonic()
                for cls_name, cls_ref_vecs in local_ref_by_class.items():
                    if not cls_ref_vecs:
                        continue
                    sims = [cosine_similarity(q, ref) for ref in cls_ref_vecs]
                    best_sim = max(sims) if sims else -1.0
                    print(f"[DEBUG] {cls_name} similarity: {best_sim:.2f}",flush=True)
                    if best_sim >= SIM_THR and (now_mono - last_saved_time_by_cls[cls_name] > cooldown_seconds):
                        out_dir = os.path.join(target_root, cls_name)
                        save_image_to_dir(frame, out_dir)
                        last_saved_time_by_cls[cls_name] = now_mono
                        last_detected_time_by_cls[cls_name] = now_mono
                        print(f"[YOLO Thread] Saved: {cls_name}, similarity={best_sim:.2f}",flush=True)

        now_mono = time.monotonic()
        for cls_name, t_last in list(last_detected_time_by_cls.items()):
            if t_last > 0 and (now_mono - t_last) > upload_pic:
                cls_dir = os.path.join(target_root, cls_name)
                try:
                    from firebase_uploader import upload_latest_image
                    upload_latest_image(cls_dir)
                    for f in os.listdir(cls_dir):
                        fp = os.path.join(cls_dir, f)
                        if os.path.isfile(fp):
                            os.remove(fp)
                    print(f"[YOLO Thread] Uploaded & cleared: {cls_name}")
                except Exception as e:
                    print(f"[YOLO Thread] Upload failed: {cls_name} -> {e}")
                finally:
                    last_detected_time_by_cls[cls_name] = 0.0
   

    cap.release()
    cv2.destroyAllWindows()

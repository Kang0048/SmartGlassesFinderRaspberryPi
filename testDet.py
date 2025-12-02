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
import threading
#from bytetrack import BYTETracker
from bytetrack.tracker.byte_tracker import BYTETracker
class Args:
    track_thresh = 0.4
    track_buffer = 30
    match_thresh = 0.6
    mot20 = False

def maintain_last_images(folder, max_count=10):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if len(files) <= max_count:
        return
    files.sort(key=lambda x: os.path.getmtime(x))
    for f in files[:-max_count]:
        os.remove(f)
        print(f"[Cleanup] Deleted old file: {f}")

def make_vec_list(vec_dir: str, label:str) -> list:
    new_ref_vecs = []
    for f in sorted(os.listdir(vec_dir)):
        if f.endswith(".pt"):
          pt_path = os.path.join(vec_dir, f)
          txt_path = pt_path.replace(".pt", ".txt")
          v = torch.load(pt_path, map_location="cpu")
          if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
          v = v.float()
          if os.path.exists(txt_path):
            with open(txt_path,"r") as t:
              yolo_cls = int(t.read().strip())
          else:
            yolo_cls = -1

          new_ref_vecs.append((v,yolo_cls))
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
        vecs = make_vec_list(class_vec_dir, class_name)
        if vecs:
            ref_by_class[class_name] = vecs
    return ref_by_class

def save_image_to_dir(frame_bgr, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    path = os.path.join(out_dir, f"{ts}.jpg")
    cv2.imwrite(path, frame_bgr)
    return path

def async_upload(cls_dir, cls_name):
    try:
        upload_latest_image(cls_dir)
        for f in os.listdir(cls_dir):
            fp = os.path.join(cls_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
        print(f"[UPLOAD] Uploaded & cleared: {cls_name}")
    except Exception as e:
        print(f"[UPLOAD] Upload failed: {cls_name} -> {e}")
# -------------------------
# Detection Loop (Thread)
# -------------------------
def detection_loop(yolo, embedder, transform, target_root, source_root, cap, camera_lock, pause_event, frame_queue,annotated_queue):
    boxes_to_draw = []  # frame에 그릴 정보 담을 리스트
    LAST_ROOT = "/home/pi/SmartGlassesFinderRaspberryPi/last"
    last_yolo_time = 0
    YOLO_INTERVAL = 0  # 1초마다 실행  
    IMG_SIZE = 640
    SIM_THR = 0.75
    cooldown_seconds = 10.0
    upload_pic = 10.0
    LCD_WIDTH = 480
    LCD_HEIGHT = 320
    embedding_cache = {}  # track_id -> embedding
    last_saved_time_by_cls = defaultdict(float)
    last_detected_time_by_cls = defaultdict(float)
    folder_initialized = False 

    args = Args()
    tracker = BYTETracker(args)
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
            print(f"[DEBUG] local_ref_by_class length: {len(local_ref_by_class)}")
            print(f"[DEBUG] keys: {list(local_ref_by_class.keys())}")
            for cls_name in local_ref_by_class.keys():
                os.makedirs(os.path.join(target_root, cls_name), exist_ok=True)
            folder_initialized = True
            print("[YOLO Thread] Target folders initialized")

        frame_to_show = frame_queue.get()
        now = time.monotonic()
        fps_time = time.monotonic()
        if now - last_yolo_time >= YOLO_INTERVAL:
            last_yolo_time = now
            frame_to_save = frame_to_show.copy()
            frame_to_box = frame_to_save.copy()
            with torch.inference_mode():
                res = yolo(frame_to_save, size=640)
                det = res.xyxy[0].cpu().numpy() if hasattr(res, "xyxy") else res.pred[0].cpu().numpy()
                
            if det is None or len(det) == 0:
                print("no box captured",flush=True)
                continue
            online_targets = tracker.update(det, [frame_to_save.shape[1], frame_to_save.shape[0]], [frame_to_save.shape[1], frame_to_save.shape[0]])
            crops, track_ids = [], []
            for t in online_targets:
                tid = t.track_id
                x1, y1, x2, y2 = map(int, t.tlbr)
                if tid not in embedding_cache:
                    crop = frame_to_save[y1:y2, x1:x2]
                    crops.append(transform(crop))
                    track_ids.append(tid)
            if crops:
                batch = torch.stack(crops, dim=0)  # (N,3,H,W)
                with torch.no_grad():
                    embeddings = embedder(batch).float()  # (N, 512)
                for tid, emb in zip(track_ids, embeddings):
                    embedding_cache[tid] = emb.squeeze()  # 1차원 벡터로 캐시
                 
            boxes_to_draw = []
            for t in online_targets:
                tid = t.track_id
                x1, y1, x2, y2 = map(int, t.tlbr)
                q = embedding_cache[tid]
                #boxes_to_draw.append((x1,y1,x2,y2))
                print(f"{x1}, {y1}, {x2}, {y2} : 좌표값")
                now_mono = time.monotonic()
                check = False
                for label_name, ref_vecs in local_ref_by_class.items():
                    for ref_vec, ref_cls in ref_vecs:
                        sims = cosine_similarity(q, ref_vec)
                        if sims >= SIM_THR and (now_mono - last_saved_time_by_cls[label_name] > cooldown_seconds):
                            boxes_to_draw.append((x1, y1, x2, y2))
                            out_dir = os.path.join(target_root, label_name)
                            last_dir = os.path.join(LAST_ROOT,label_name)
                            save_image_to_dir(frame_to_save, out_dir)
                            save_image_to_dir(frame_to_save, last_dir)
                            maintain_last_images(last_dir, max_count=10)
                            last_saved_time_by_cls[label_name] = now_mono
                            last_detected_time_by_cls[label_name] = now_mono
                            check = True
                            print('식별됨')
                            break
                now_time = time.monotonic()
                print(f"{now_time - fps_time}")
                if annotated_queue.full():
                   _ = annotated_queue.get()
            annotated_queue.put(boxes_to_draw)
                
        # --------------------------
        # 일정 시간 지나면 업로드
        # --------------------------
        for cls_name, t_last in list(last_detected_time_by_cls.items()):
            if t_last > 0 and (now - t_last) > upload_pic:
                cls_dir = os.path.join(target_root, cls_name)
                threading.Thread(target=async_upload, args=(cls_dir, cls_name), daemon=True).start()
                last_detected_time_by_cls[cls_name] = 0.0
                

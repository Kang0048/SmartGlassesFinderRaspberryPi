#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, unicodedata, sys, json, queue
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models import resnet18
import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer

OBJECTS_ROOT   = "/home/pi/yolo-object-matcher/objects"
REPO_DIR       = "/home/pi/yolo-object-matcher/yolov5"
WEIGHTS        = "/home/pi/yolo-object-matcher/yolov5n.pt"
IMG_SIZE       = 640
PAD            = 0.05
SR             = 16000
VOSK_MODEL_DIR = "/home/pi/models/vosk-model-small-en-us-0.15"

def ensure_label_dir(label: str) -> str:
    base = os.path.join(OBJECTS_ROOT, label)
    os.makedirs(base, exist_ok=True)
    return base

def pad_clip(x1, y1, x2, y2, W, H, pad=0.05):
    w, h = x2 - x1, y2 - y1
    wpad, hpad = int(w * pad), int(h * pad)
    x1 = max(0, int(x1 - wpad))
    y1 = max(0, int(y1 - hpad))
    x2 = min(W - 1, int(x2 + wpad))
    y2 = min(H - 1, int(y2 + hpad))
    return x1, y1, x2, y2

def choose_center_idx(det, W, H):
    cx = (det[:, 0] + det[:, 2]) / 2.0
    cy = (det[:, 1] + det[:, 3]) / 2.0
    d2 = (cx - W / 2.0) ** 2 + (cy - H / 2.0) ** 2
    return int(np.argmin(d2))

def sanitize_label(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z0-9\-_]+", "", text)
    return text[:32] if text else "item"

def next_index(label_dir: str, label: str) -> int:
    pat = re.compile(rf"^{re.escape(label)}_(\d+)\.jpg$")
    max_idx = 0
    if os.path.isdir(label_dir):
        for fn in os.listdir(label_dir):
            m = pat.match(fn)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except:
                    pass
    return max_idx + 1

def save_capture_and_vector(frame, yolo, embedder, transform, label_dir, label, idx):
    img_name = f"{label}_{idx}.jpg"
    img_path = os.path.join(label_dir, img_name)
    cv2.imwrite(img_path, frame)
    print(f"saved image: {img_path}")

    H, W = frame.shape[:2]
    with torch.inference_mode():
        res = yolo(frame, size=IMG_SIZE)
        det = res.xyxy[0].cpu().numpy() if hasattr(res, "xyxy") else res.pred[0].cpu().numpy()

    if det is None or len(det) == 0:
        print("no detection box; skip vector")
        return

    idx_box = 0 if len(det) == 1 else choose_center_idx(det, W, H)
    x1, y1, x2, y2 = det[idx_box, :4].astype(int)
    x1, y1, x2, y2 = pad_clip(x1, y1, x2, y2, W, H, PAD)

    if x2 > x1 and y2 > y1:
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            vec = embedder(transform(crop_rgb).unsqueeze(0)).squeeze().cpu()
        vec_path = os.path.join(label_dir, f"{label}_{idx}.pt")
        torch.save(vec, vec_path)
        print(f"vector saved: {vec_path}")
    else:
        print("invalid crop box; skip vector")

def init_audio():
    if not os.path.isdir(VOSK_MODEL_DIR):
        print(f"Vosk model not found: {VOSK_MODEL_DIR}")
        sys.exit(1)
    model_vosk = VoskModel(VOSK_MODEL_DIR)
    recognizer = KaldiRecognizer(model_vosk, SR)
    recognizer.SetWords(True)
    audio_q = queue.Queue()
    def audio_callback(indata, frames, t, status):
        audio_q.put(bytes(indata))
    stream = sd.RawInputStream(
        samplerate=SR, blocksize=8000, dtype="int16",
        channels=1, callback=audio_callback
    )
    stream.start()
    return recognizer, audio_q, stream

def main():
    print("Loading YOLO...")
    yolo = torch.hub.load(REPO_DIR, "custom", path=WEIGHTS, source="local")
    yolo.conf = 0.1
    yolo.iou  = 0.5
    yolo.to("cpu").eval()

    print("Loading ResNet18 embedder...")
    embedder = resnet18(weights="IMAGENET1K_V1")
    embedder.fc = torch.nn.Identity()
    embedder.eval()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    print("Initializing Vosk...")
    recognizer, audio_q, stream = init_audio()

    state = "idle"
    pending_label = None

    print("Voice control ready.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            if not audio_q.empty():
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    if not text:
                        continue

                    print(f"[voice] {text}")

                    if "quit" in text or "exit" in text:
                        print("Exit by voice command.")
                        break

                    if state == "idle":
                        if re.search(r"\bhi\b", text):
                            state = "await_label"
                            print("Armed. Say one word for label.")

                    elif state == "await_label":
                        tokens = re.findall(r"[a-z0-9\-_]+", text)
                        label = sanitize_label(tokens[0]) if tokens else None
                        if label:
                            pending_label = label
                            state = "confirm"
                            print(f"Did you say '{pending_label}'? Say yes or no.")

                    elif state == "confirm":
                        if re.search(r"\byes\b", text):
                            label = pending_label
                            pending_label = None
                            label_dir = ensure_label_dir(label)
                            idx = next_index(label_dir, label)
                            print(f"Confirmed. Label '{label}' → {label_dir}")
                            save_capture_and_vector(frame, yolo, embedder, transform, label_dir, label, idx)
                            state = "idle"
                            print("Back to idle. Say 'hi' again.")
                        elif re.search(r"\bno\b", text):
                            print("Okay, say the label again.")
                            pending_label = None
                            state = "await_label"
                        else:
                            print("Please say yes or no.")

            time.sleep(0.01)

    finally:
        cap.release()
        try:
            stream.stop()
        except:
            pass

if __name__ == "__main__":
    main()

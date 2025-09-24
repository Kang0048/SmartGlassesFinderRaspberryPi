# voiceThread.py
import threading, queue, time, os, sys, re, json, unicodedata, cv2, torch
from vosk import Model as VoskModel, KaldiRecognizer
import sounddevice as sd
import numpy as np
import os
from gtts import gTTS
from firebase_uploader import upload_object_image
VOSK_MODEL_DIR = "/home/pi/models/vosk-model-small-en-us-0.15"
SR = 16000
PAD = 0.05
OBJECTS_ROOT = "/home/pi/SmartGlassesFinderRaspberryPi/objects"
SOURCE_ROOT = "/home/pi/SmartGlassesFinderRaspberryPi/last"
IMG_SIZE = 640

def ensure_label_dir(label: str) -> str:
    base = os.path.join(OBJECTS_ROOT, label)
    os.makedirs(base, exist_ok=True)
    return base
    
def speak(text, stream,lang="en"):
    filename = "temp_tts.mp3"
    stream.stop()
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        os.system(f"mpg321 {filename}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)
    stream.start()

def sanitize_label(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^a-z0-9\-_]+", "", text)
    return text[:32] if text else "item"

def next_index(label_dir: str, label: str) -> int:
    import re
    img_dir = os.path.join(label_dir, "images") 
    pat = re.compile(rf"^{re.escape(label)}_(\d+)\.jpg$")
    max_idx = 0
    if os.path.isdir(img_dir):
        for fn in os.listdir(img_dir):
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
    img_dir = os.path.join(label_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    img_name = f"{label}_{idx}.jpg"
    img_path = os.path.join(img_dir, img_name)
    cv2.imwrite(img_path, frame)
    print(f"[voiceThread] saved image: {img_path}")

    # YOLO detection & vector
    H, W = frame.shape[:2]
    cx_frame, cy_frame = W//2, H//2
    with torch.inference_mode():
        res = yolo(frame, size=IMG_SIZE)
        det = res.xyxy[0].cpu().numpy() if hasattr(res, "xyxy") else res.pred[0].cpu().numpy()

    if det is None or len(det) == 0:
        print("[voiceThread] no detection box; skip vector")
        return

    min_dist = float('inf')
    central_box_idx = -1

    for idx_box, box in enumerate(det):
        x1, y1, x2, y2, conf, yolo_cls = box
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2
        dist = (x_c - cx_frame)**2 + (y_c - cy_frame)**2  # 거리 제곱
        if dist < min_dist:
           min_dist = dist
           central_box_idx = idx_box

    box = det[central_box_idx]
    x1, y1, x2, y2, conf, yolo_cls = box
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    wpad, hpad = int((x2-x1)*PAD), int((y2-y1)*PAD)
    x1, y1 = max(0, x1-wpad), max(0, y1-hpad)
    x2, y2 = min(W-1, x2+wpad), min(H-1, y2+hpad)
    
    if x2 > x1 and y2 > y1:
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            vec = embedder(transform(crop_rgb).unsqueeze(0)).squeeze().cpu()

        vec_dir = os.path.join(label_dir, "vectors")
        os.makedirs(vec_dir, exist_ok=True)
        vec_path = os.path.join(vec_dir, f"{label}_{idx}.pt")
        torch.save(vec, vec_path)
        
        label_path = os.path.join(vec_dir, f"{label}_{idx}.txt")
        with open (label_path, "w") as f:
             f.write(str(yolo_cls))
        print(f"[voiceThread] vector saved: {vec_path}")
    else:
        print("[voiceThread] invalid crop box; skip vector")

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
        channels=1,callback=audio_callback
    )
    stream.start()
    print(stream)
    print(stream.active)
    return recognizer, audio_q, stream

def voice_label_thread(yolo, embedder, transform, cap, camera_lock, pause_event,recognizer, audio_q, stream, show_queue):
    sd.default.device = "USB Audio Device"
    pending_label = None
    state = "idle"
    print("[voiceThread] Voice control ready.")
    try:
        while True:
            if not audio_q.empty():
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    if not text:
                        continue
                    print(f"[voiceThread][voice] {text}")

                    if "out" in text or "exit" in text:
                        state = "idle"
                        speak("tracking system start",stream)
                        print("[voiceThread] Exit by voice command.")
                        pause_event.clear()

                    # --- 시작 ---
                    if state == "idle" and re.search(r"\bhi\b", text):
                        print("[voiceThread] 'hi' detected. Pausing YOLO...")
                        pause_event.set()
                        state = "decision"
                        speak("May I help you?",stream)
                        print("[voiceThread] Do you want to 'make' or 'delete'?")

                    # --- decision ---
                    elif state == "decision":
                        if "make" in text:
                            state = "await_label"
                            print("[voiceThread] Say one word to make label's name.")
                            speak("say one word to make label name",stream)
                        elif "delete" in text:
                            state = "delete_await_label"
                            print("[voiceThread] Say the folder(label) name you want to delete.")
                            speak("say folder name you want to delete",stream)
                        elif "see" in text:
                            speak("say folder name you want to see",stream)
                            state = "see_picture"
                        else:
                            speak("please say again",stream)        

                    # --- make flow ---
                    elif state == "await_label":
                        tokens = re.findall(r"[a-z0-9\-_]+", text)
                        label = sanitize_label(tokens[0]) if tokens else None
                        if label:
                            pending_label = label
                            state = "confirm"
                            print(f"[voiceThread] Did you say '{pending_label}'? Say yes or no.")
                            speak(f"Did you say {pending_label}? say yes or no",stream)
                    elif state == "see_picture":
                        tokens = re.findall(r"[a-z0-9\-_]+",text)
                        label = sanitize_label(tokens[0]) if tokens else None
                        if label:
                           pending_label = label
                           label_dir = os.path.join(SOURCE_ROOT, pending_label)
                           if os.path.isdir(label_dir):
                                state = "show_pircure"
                                print(f"[voiceThread] Folder '{pending_label}' found. say out to quit")
                                speak(f"{pending_label} folder found. say out to quit",stream)
                                image_files = [f for f in os.listdir(label_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                                if image_files:
                                   print("image founded")
                                   img_path = os.path.join(label_dir, image_files[0])
                                   show_queue.put(img_path)
                                else:
                                   print(f"[voiceThread] No images found in {label_dir}")
                           else:
                               print(f"[voiceThread] Folder '{pending_label}' not found. Say another name.")
                               speak("folder not found please say again",stream)
                    elif state == "show_picture":
                        folder_path = os.path.join()             
                    elif state == "confirm":
                        if re.search(r"\byes\b", text):
                            state = "capturing"
                            print(f"[voiceThread] Capturing started for label '{pending_label}'. Say 'good' to take a photo, 'done' to finish.")
                            speak("say good to take a photo",stream)
              
                        elif re.search(r"\bno\b", text):
                            print("[voiceThread] Okay, say the label again.")
                            speak("okay say the label again",stream)
                            pending_label = None
                            state = "await_label"
                        else:
                            print("[voiceThread] Please say yes or no.")
                            speak("please say yes or no again",stream)

                    elif state == "capturing":
                        if re.search(r"\bdone\b", text):
                            print(f"[voiceThread] Done capturing for label '{pending_label}'.")
                            speak("finshed tracking system start",stream)
                            state = "idle"
                            label = pending_label
                            img_path = os.path.join(OBJECTS_ROOT,label)
                            threading.Thread(target = upload_object_image, args = (img_path, label), daemon=True).start()
                            pending_label = None
                            pause_event.clear()
                        elif re.search(r"\bgood\b", text):
                            label_dir = ensure_label_dir(pending_label)
                            idx = next_index(label_dir, pending_label)
                            with camera_lock:
                                cap.read()  
                                ret, frame = cap.read()
                                if ret:
                                    save_capture_and_vector(frame, yolo, embedder, transform, label_dir, pending_label, idx)
                                    print(f"[voiceThread] Captured #{idx} for label '{pending_label}'")
                            speak("say good to take more picture",stream)

                    # --- delete flow ---
                    elif state == "delete_await_label":
                        tokens = re.findall(r"[a-z0-9\-_]+", text)
                        label = sanitize_label(tokens[0]) if tokens else None
                        if label:
                            pending_label = label
                            label_dir = os.path.join(OBJECTS_ROOT, pending_label)
                            if os.path.isdir(label_dir):
                                state = "delete_confirm"
                                print(f"[voiceThread] Folder '{pending_label}' found. Say yes to delete, no to cancel.")
                                speak(f"{pending_label} folder found. say yes or no to delete",stream)
                            else:
                                print(f"[voiceThread] Folder '{pending_label}' not found. Say another name.")
                                speak("folder not found please say again",stream)

                    elif state == "delete_confirm":
                        if re.search(r"\byes\b", text):
                            label_dir = os.path.join(OBJECTS_ROOT, pending_label)
                            try:
                                import shutil
                                shutil.rmtree(label_dir)
                                print(f"[voiceThread] Folder '{pending_label}' deleted.")
                                speak(f"{pending_label} folder deleted",stream)
                            except Exception as e:
                                print(f"[voiceThread] Error deleting '{pending_label}': {e}")
                            pending_label = None
                            state = "idle"  
                            pause_event.clear()
                        elif re.search(r"\bno\b", text):
                            print("[voiceThread] Delete cancelled. Back to decision.")
                            speak("delete cancelled",stream)
                            pending_label = None
                            state = "decision"
                            speak("back to system",stream)
                            pause_event.clear()
                   
            time.sleep(0.1)
    finally:
        try:
            stream.stop()
        except:
            pass
        try:
            stream.close()
        except:
            pass

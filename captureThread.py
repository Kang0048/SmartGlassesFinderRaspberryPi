# voiceThread.py
import threading, queue, time, os, sys, re, json, unicodedata, cv2, torch
from vosk import Model as VoskModel, KaldiRecognizer
import sounddevice as sd

VOSK_MODEL_DIR = "/home/pi/models/vosk-model-small-en-us-0.15"
SR = 16000
PAD = 0.05
OBJECTS_ROOT = "/home/pi/yolo-object-matcher/objects"
IMG_SIZE = 640

def ensure_label_dir(label: str) -> str:
    base = os.path.join(OBJECTS_ROOT, label)
    os.makedirs(base, exist_ok=True)
    return base

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
    with torch.inference_mode():
        res = yolo(frame, size=IMG_SIZE)
        det = res.xyxy[0].cpu().numpy() if hasattr(res, "xyxy") else res.pred[0].cpu().numpy()

    if det is None or len(det) == 0:
        print("[voiceThread] no detection box; skip vector")
        return

    idx_box = 0
    x1, y1, x2, y2 = det[idx_box, :4].astype(int)
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
        channels=1, callback=audio_callback
    )
    stream.start()
    return recognizer, audio_q, stream

def voice_label_thread(yolo, embedder, transform, cap, camera_lock, pause_event):
    recognizer, audio_q, stream = init_audio()
    state = "idle"
    pending_label = None

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

                    if "quit" in text or "exit" in text:
                        print("[voiceThread] Exit by voice command.")
                        break

                    # --- 시작 ---
                    if state == "idle" and re.search(r"\bhi\b", text):
                        print("[voiceThread] 'hi' detected. Pausing YOLO...")
                        pause_event.set()
                        state = "decision"
                        print("[voiceThread] Do you want to 'make' or 'delete'?")

                    # --- decision ---
                    elif state == "decision":
                        if "make" in text:
                            state = "await_label"
                            print("[voiceThread] Say one word to make label's name.")
                        elif "delete" in text:
                            state = "delete_await_label"
                            print("[voiceThread] Say the folder(label) name you want to delete.")

                    # --- make flow ---
                    elif state == "await_label":
                        tokens = re.findall(r"[a-z0-9\-_]+", text)
                        label = sanitize_label(tokens[0]) if tokens else None
                        if label:
                            pending_label = label
                            state = "confirm"
                            print(f"[voiceThread] Did you say '{pending_label}'? Say yes or no.")

                    elif state == "confirm":
                        if re.search(r"\byes\b", text):
                            state = "capturing"
                            print(f"[voiceThread] Capturing started for label '{pending_label}'. Say 'good' to take a photo, 'done' to finish.")
                        elif re.search(r"\bno\b", text):
                            print("[voiceThread] Okay, say the label again.")
                            pending_label = None
                            state = "await_label"
                        else:
                            print("[voiceThread] Please say yes or no.")

                    elif state == "capturing":
                        if re.search(r"\bdone\b", text):
                            print(f"[voiceThread] Done capturing for label '{pending_label}'.")
                            pending_label = None
                            state = "idle"
                            pause_event.clear()
                        elif re.search(r"\bgood\b", text):
                            label_dir = ensure_label_dir(pending_label)
                            idx = next_index(label_dir, pending_label)
                            with camera_lock:
                                cap.read()  # 버퍼 비우기
                                ret, frame = cap.read()
                                if ret:
                                    save_capture_and_vector(frame, yolo, embedder, transform, label_dir, pending_label, idx)
                                    print(f"[voiceThread] Captured #{idx} for label '{pending_label}'")

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
                            else:
                                print(f"[voiceThread] Folder '{pending_label}' not found. Say another name.")

                    elif state == "delete_confirm":
                        if re.search(r"\byes\b", text):
                            label_dir = os.path.join(OBJECTS_ROOT, pending_label)
                            try:
                                import shutil
                                shutil.rmtree(label_dir)
                                print(f"[voiceThread] Folder '{pending_label}' deleted.")
                            except Exception as e:
                                print(f"[voiceThread] Error deleting '{pending_label}': {e}")
                            pending_label = None
                            state = "idle"
                            pause_event.clear()
                        elif re.search(r"\bno\b", text):
                            print("[voiceThread] Delete cancelled. Back to decision.")
                            pending_label = None
                            state = "decision"
                        else:
                            print("[voiceThread] Please say yes or no.")
            time.sleep(0.1)
    finally:
        try:
            stream.stop()
        except:
            pass

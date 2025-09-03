# main.py
import threading, time, cv2
from detectThread import detection_loop
from captureThread import voice_label_thread
import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms

source_root = '/home/pi/yolo-object-matcher/objects'
target_root = '/home/pi/yolo-object-matcher/detected_matches'
REPO_DIR    = '/home/pi/yolo-object-matcher/yolov5'
WEIGHTS     = '/home/pi/yolo-object-matcher/yolov5n.pt'

def main():

    print("[MAIN] Loading embedder...")
    embedder = resnet18(pretrained=True)
    embedder.fc = torch.nn.Identity()
    embedder.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("[MAIN] Loading YOLOv5 model...")
    yolo = torch.hub.load(REPO_DIR, 'custom', path=WEIGHTS, source='local')
    yolo.conf = 0.25
    yolo.iou  = 0.5
    yolo.to('cpu').eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[MAIN] Cannot open camera. Exiting.")
        return
    camera_lock = threading.Lock()
    pause_event = threading.Event()

    t_yolo = threading.Thread(
        target=detection_loop,
        args=(yolo, embedder, transform, target_root, source_root, cap, camera_lock, pause_event),
        daemon=True
    )
    t_yolo.start()
    print("[MAIN] Detection thread started")

    t_voice = threading.Thread(
        target=voice_label_thread,
        args=(yolo, embedder, transform, cap, camera_lock, pause_event),
        daemon=True
    )
    t_voice.start()
    print("[MAIN] Voice label thread started")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[MAIN] 종료")
        t_yolo.join()
        t_voice.join()
        cap.release()


if __name__ == "__main__":
    main()

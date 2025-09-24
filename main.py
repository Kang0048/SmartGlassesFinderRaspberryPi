# main.py
import threading, time, cv2
from detectThread import detection_loop
from captureThread import voice_label_thread
import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
import os
from queue import Queue
from captureThread import init_audio
from firebase_utils import init_firebase, download_detected_matches
from match_utils import make_vector_folder, find_working_camera 
source_root = '/home/pi/SmartGlassesFinderRaspberryPi/objects'
target_root = '/home/pi/SmartGlassesFinderRaspberryPi/detected_matches'
REPO_DIR    = '/home/pi/SmartGlassesFinderRaspberryPi/yolov5'
WEIGHTS     = '/home/pi/SmartGlassesFinderRaspberryPi/yolov5n.pt'
FIREBASE_CRED_PATH = '/home/pi/SmartGlassesFinderRaspberryPi/smartglassesfinder-firebase-adminsdk-fbsvc-6b2ce3da61.json'
BUCKET_NAME = 'smartglassesfinder.appspot.com'
shared_state = {"mode": "idle"}
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

    init_firebase(FIREBASE_CRED_PATH, BUCKET_NAME)
    download_detected_matches(source_root)
    print("firebase picture downloaded")
    
    make_vector_folder(source_root, yolo, embedder, transform)
    cap = cv2.VideoCapture(0)

# 해상도 지정 (카메라 드라이버가 지원하는 값이어야 함)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)   # 가로 해상도
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)  # 세로 해상도
    cap.set(cv2.CAP_PROP_FPS, 15)            # 초당 프레임 수

    if not cap.isOpened():
        print("[MAIN] Cannot open camera. Exiting.")
        return
    camera_lock = threading.Lock()
    pause_event = threading.Event()

    frame_queue = Queue()      # 실시간 카메라 영상
    show_queue  = Queue()      # 사진 보기 요청
    capture_queue = Queue()
    t_yolo = threading.Thread(
        target=detection_loop,
        args=(yolo, embedder, transform, target_root, source_root, cap, camera_lock, pause_event,frame_queue),
        daemon=True
    )
    t_yolo.start()
    print("[MAIN] Detection thread started")
    recognizer, audio_q, stream = init_audio()
    t_voice = threading.Thread(
        target=voice_label_thread,
        args=(yolo, embedder, transform, cap, camera_lock, pause_event,recognizer, audio_q, stream,show_queue),
        daemon=True
    )
    t_voice.start()
    print("[MAIN] Voice label thread started")

    try:
      while True:
           if not frame_queue.empty():
              frame = frame_queue.get()
              cv2.imshow("Camera", frame)

            # 2. 사진 보여주기 처리
           if not show_queue.empty():
              img_path = show_queue.get()
              img = cv2.imread(img_path)
              if img is not None:
                 cv2.imshow("Label Preview", img)

           key = cv2.waitKey(30) & 0xFF
           if key == ord('q'):
              break

    except KeyboardInterrupt:
        print("[MAIN] 종료")
        t_yolo.join()
        t_voice.join()
        cap.release()
    finally:
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
        main()

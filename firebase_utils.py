import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
import shutil
import urllib.parse
import requests

def init_firebase(cred_path: str, bucket_name: str):
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': bucket_name
        })
    print("[Firebase] Initialized")

def extract_storage_path(image_url: str) -> str:
    if "firebasestorage.googleapis.com" in image_url:
        path = image_url.split("/o/")[1].split("?")[0]
        return urllib.parse.unquote(path)   # objects/picture/xxx.jpg
    elif "storage.googleapis.com" in image_url:
        # 라즈베리파이 업로드 URL 형식
        path = image_url.split("/", 4)[-1]  # objects/cell/xxx.jpg
        return path
    else:
        raise ValueError(f"Unknown URL format: {image_url}")


def download_detected_matches(source_root: str):
    db = firestore.client()

    # 기존 폴더 초기화
    if os.path.exists(source_root):
        shutil.rmtree(source_root)
    os.makedirs(source_root, exist_ok=True)

    docs = db.collection("detected_objects").stream()
    for doc in docs:
        data = doc.to_dict()
        name = data.get("name")
        image_url = data.get("imageUrl")   # Firestore에서 꺼낸 URL
        if not name or not image_url:
            continue

        # label별 디렉토리 생성
        label_dir = os.path.join(source_root, name, "images")
        os.makedirs(label_dir, exist_ok=True)

        # 파일 이름 생성 (token 제거)
        filename = os.path.basename(image_url.split("?")[0])
        local_path = os.path.join(label_dir, filename)

        # 다운로드
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"[OK] {local_path} 저장 완료")
            else:
                print(f"[ERROR] 다운로드 실패: {image_url}")
        except Exception as e:
            print(f"[EXCEPTION] {image_url} -> {e}")

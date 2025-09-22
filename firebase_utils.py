import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
import shutil
import urllib.parse
def init_firebase(cred_path: str, bucket_name: str):
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': bucket_name
        })
    print("[Firebase] Initialized")

def download_detected_matches(source_root: str):
    db = firestore.client()
    bucket = storage.bucket()

    if os.path.exists(source_root):
        shutil.rmtree(source_root)
    os.makedirs(source_root, exist_ok=True)

    docs = db.collection("detected_objects").stream()
    for doc in docs:	
        data = doc.to_dict()
        name = data.get("name")
        image_uri = data.get("imageUrl")  # Storage 경로

        if not name or not image_uri:
            continue
        print(f"image uri : {image_uri}")
        # label 폴더 + images 폴더 생성
        label_dir = os.path.join(source_root, name, "images")
        os.makedirs(label_dir, exist_ok=True)

        # Storage blob 다운로드
        filename = os.path.basename(image_uri)
        local_path = os.path.join(label_dir, filename)
        parseUrl = urllib.parse.unquote(image_uri.split("/o/")[1].split("?")[0])
        blob = bucket.blob(parseUrl)
        blob.download_to_filename(local_path)
        print(f"[Firebase] Downloaded: {local_path}")

    print("[Firebase] All files downloaded.")

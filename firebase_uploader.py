import os
import firebase_admin
from firebase_admin import credentials, storage

if not firebase_admin._apps:
    cred = credentials.Certificate("smartglassesfinder-firebase-adminsdk-fbsvc-b822b9f07c.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'smartglassesfinder.firebasestorage.app'
    })

def upload_latest_image(folder_path : str):
    files = []
    for f in os.listdir(folder_path):
        fp = os.path.join(folder_path, f)
        if os.path.isfile(fp) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            files.append(fp)
    if not files:
        print("no file exists.")
        return

    files = [os.path.join(folder_path, f) for f in files]
    latest_file = max(files, key=os.path.getmtime)

    object_name = os.path.basename(os.path.normpath(folder_path))
    filename = os.path.basename(latest_file)
    dest_path = f"{object_name}/{filename}"


    bucket = storage.bucket()
    blob = bucket.blob(dest_path)
    
    blob.upload_from_filename(latest_file)
    blob.make_public()

    print(f"Firebase uploaded : {latest_file}")
    print(f"URL: {blob.public_url}")
    return blob.public_url

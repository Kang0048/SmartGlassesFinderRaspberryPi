import os
import firebase_admin
from firebase_admin import credentials, storage, firestore
import time
if not firebase_admin._apps:
    cred = credentials.Certificate("smartglassesfinder-firebase-adminsdk-fbsvc-4e79e15856.json")
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

def upload_object_image(local_path:str, label:str) :
    try:
       label_path = os.path.join(local_path, "images")
       if not os.path.exists(label_path):
            print(f"[firebase] no images folder: {label_path}")
            return
       
       bucket = storage.bucket()
       db = firestore.client()

       for img_file in os.listdir(label_path) :
           if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
               continue 
           img_path = os.path.join(label_path,img_file)
  
           ts = int(time.time()*1000)
           blob_path = f"objects/{label}/{ts}_{img_file}.jpg"
           blob = bucket.blob(blob_path)
           blob.upload_from_filename(img_path)
           blob.make_public()
           url = blob.public_url
 
           doc = {
               "name": label,
               "imageUrl": url,
               "timestamp":ts
           }
           db.collection("detected_objects").document(label).set(doc)
           print(f"[firebase] upload: {label}, url = {url}")
    except Exception as e:
           print(f"[firebase] upload failed: {e}")

  

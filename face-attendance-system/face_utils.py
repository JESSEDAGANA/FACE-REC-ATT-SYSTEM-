import cv2
import numpy as np
import os
from deepface import DeepFace

# Initialize face detector - use the correct cascade path
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_faces(img, min_neighbors=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(  # Fixed the parenthesis here
        gray,
        scaleFactor=1.1,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )  # Added the missing closing parenthesis
    return faces

def get_face_embedding(img):
    try:
        # Using Facenet for embeddings
        embedding = DeepFace.represent(img, model_name='Facenet')[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def save_face_image(user_id, img, face_coords):
    x, y, w, h = face_coords
    face_img = img[y:y+h, x:x+w]
    
    user_folder = f"static/faces/{user_id}"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    img_count = len(os.listdir(user_folder))
    img_path = f"{user_folder}/{img_count+1}.jpg"
    cv2.imwrite(img_path, face_img)
    return img_path

def load_face_embeddings():
    embeddings = []
    labels = []
    
    users = os.listdir("static/faces")
    for user_id in users:
        user_folder = f"static/faces/{user_id}"
        if not os.path.isdir(user_folder):
            continue
            
        for imgname in os.listdir(user_folder):
            img_path = f"{user_folder}/{imgname}"
            img = cv2.imread(img_path)
            embedding = get_face_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(int(user_id))
    
    return embeddings, labels
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import cv2
import numpy as np
import joblib
from database import *
from face_utils import *
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app)

# Initialize database
init_db()

# Global variables
DATETODAY = datetime.now().strftime("%d-%B-%Y")
N_IMAGES = 10  # Number of images to capture for each user

@app.route('/')
def home():
    attendance = get_today_attendance()
    users = get_all_users()
    return render_template('home.html', 
                         attendance=attendance,
                         users=users,
                         datetoday2=DATETODAY,
                         totalreg=len(users))

@app.route('/add', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        try:
            username = secure_filename(request.form['newusername'].strip())
            user_id = int(request.form['newuserid'])
            
            # Validate inputs
            if not re.match(r'^[A-Za-z\s]{2,50}$', username):
                return render_template('home.html', 
                                     mess='Name must be 2-50 letters/spaces',
                                     attendance=get_today_attendance(),
                                     users=get_all_users(),
                                     datetoday2=DATETODAY,
                                     totalreg=len(get_all_users()))
            
            if user_exists(user_id):
                return render_template('home.html', 
                                     mess='User ID already exists',
                                     attendance=get_today_attendance(),
                                     users=get_all_users(),
                                     datetoday2=DATETODAY,
                                     totalreg=len(get_all_users()))
            
            # Start face capture process
            return redirect(url_for('capture_faces', 
                                 username=username, 
                                 user_id=user_id))
            
        except ValueError as e:
            return render_template('home.html', 
                                 mess=str(e),
                                 attendance=get_today_attendance(),
                                 users=get_all_users(),
                                 datetoday2=DATETODAY,
                                 totalreg=len(get_all_users()))
    
    return redirect(url_for('home'))

@app.route('/capture/<username>/<int:user_id>')
def capture_faces(username, user_id):
    return render_template('capture.html', 
                         username=username,
                         user_id=user_id,
                         n_images=N_IMAGES)

@app.route('/start_capture/<username>/<int:user_id>')
def start_capture(username, user_id):
    # Add user to database
    db_user_id = add_user(username, user_id)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Could not open webcam", 500
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    images_captured = 0
    while images_captured < N_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save every 5th frame to get varied images
            if images_captured % 5 == 0:
                save_face_image(user_id, frame, (x, y, w, h))
                images_captured += 1
                
            cv2.putText(frame, f"Captured: {images_captured}/{N_IMAGES}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    cap.release()
    
    # Train model after capturing
    train_model()
    
    # Send completion signal
    yield "data: done\n\n"

@app.route('/video_feed/<username>/<user_id>')
def video_feed(username, user_id):
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield "data: error:Camera not found\n\n"
            return

        images_captured = 0
        try:
            while images_captured < 10:  # 10 images
                ret, frame = cap.read()
                if not ret:
                    yield "data: error:Frame read failed\n\n"
                    break

                faces = face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
                if len(faces) > 0:
                    # Save face and increment counter
                    images_captured += 1
                    yield f"data: progress:{images_captured*10}\n\n"
                    
                # Send frame
                ret, jpeg = cv2.imencode('.jpg', frame)
                yield b'data: --frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                
            yield "data: done\n\n"
        finally:
            cap.release()
            cv2.destroyAllWindows()
    return Response(generate(), mimetype='text/event-stream')

@app.route('/take_attendance')
def take_attendance():
    # Load the trained model
    try:
        model = joblib.load('static/face_recognition_model.pkl')
    except:
        return render_template('home.html', 
                             mess='Model not trained yet',
                             attendance=get_today_attendance(),
                             users=get_all_users(),
                             datetoday2=DATETODAY,
                             totalreg=len(get_all_users()))
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Could not open webcam", 500
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    attendance_taken = False
    start_time = datetime.now()
    timeout = 30  # seconds
    
    while (datetime.now() - start_time).seconds < timeout:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            # Get embedding and predict
            embedding = get_face_embedding(face_img)
            if embedding is not None:
                user_id = model.predict([embedding])[0]
                user = get_user_by_id(user_id)
                
                if user:
                    add_attendance(user['id'])
                    attendance_taken = True
                    cv2.putText(frame, f"{user['username']}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    break
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    cap.release()
    
    # Update attendance list via SocketIO
    if attendance_taken:
        attendance = get_today_attendance()
        socketio.emit('attendance_update', {
            'attendance': [dict(row) for row in attendance]
        })
    
    yield "data: done\n\n"

@app.route('/start_attendance')
def start_attendance():
    return Response(take_attendance(),
                   mimetype='text/event-stream')

def train_model():
    embeddings, labels = load_face_embeddings()
    if embeddings and labels:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(embeddings, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        return True
    return False

if __name__ == '__main__':
    if not os.path.exists('static/faces'):
        os.makedirs('static/faces')
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

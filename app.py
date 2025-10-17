"""
Flask API Backend cho Face Recognition với DeepFace
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from datetime import datetime, timedelta
import cv2

app = Flask(__name__)
CORS(app)

# Khởi tạo Firebase
cred_path = os.environ.get('FIREBASE_CREDENTIALS', 'firebase-key.json')
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'face-timekeeping-31820.firebasestorage.app'
})

db = firestore.client()
bucket = storage.bucket()

# Cache embeddings
registered_faces = {}

def base64_to_image(base64_string):
    """Chuyển base64 sang numpy array"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image)

def get_face_embedding(image):
    """Lấy embedding từ ảnh"""
    try:
        # DeepFace trả về embedding
        embedding_objs = DeepFace.represent(
            img_path=image,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )
        
        if embedding_objs and len(embedding_objs) > 0:
            return np.array(embedding_objs[0]["embedding"])
        return None
    except Exception as e:
        print(f"Lỗi get_face_embedding: {e}")
        return None

def cosine_similarity(embedding1, embedding2):
    """Tính độ tương đồng cosine"""
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

def load_registered_faces():
    """Tải embeddings từ Firebase"""
    global registered_faces
    registered_faces = {}
    
    employees = db.collection('employees').stream()
    for emp in employees:
        data = emp.to_dict()
        name = data['name']
        embedding = np.array(data['embedding'])
        registered_faces[name] = embedding
    
    print(f"✅ Đã tải {len(registered_faces)} nhân viên")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'registered_employees': len(registered_faces)
    })

@app.route('/api/detect', methods=['POST'])
def detect_face():
    """Phát hiện khuôn mặt"""
    try:
        data = request.json
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'Thiếu ảnh'}), 400
        
        image = base64_to_image(image_base64)
        
        # Detect faces
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend="opencv",
            enforce_detection=False
        )
        
        locations = []
        for face in faces:
            area = face['facial_area']
            # Chuyển sang format [top, right, bottom, left]
            locations.append([
                area['y'],
                area['x'] + area['w'],
                area['y'] + area['h'],
                area['x']
            ])
        
        return jsonify({
            'faces_detected': len(locations),
            'locations': locations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Nhận diện khuôn mặt"""
    try:
        data = request.json
        image_base64 = data.get('image')
        threshold = data.get('threshold', 0.6)
        
        if not image_base64:
            return jsonify({'error': 'Thiếu ảnh'}), 400
        
        image = base64_to_image(image_base64)
        
        # Lấy embedding
        embedding = get_face_embedding(image)
        
        if embedding is None:
            return jsonify({'name': None, 'message': 'Không phát hiện khuôn mặt'})
        
        # So sánh với database
        best_match = None
        max_similarity = threshold
        
        for name, known_embedding in registered_faces.items():
            similarity = cosine_similarity(embedding, known_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name
        
        if best_match:
            # Lấy vị trí khuôn mặt
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend="opencv",
                enforce_detection=False
            )
            
            location = None
            if faces:
                area = faces[0]['facial_area']
                location = [
                    area['y'],
                    area['x'] + area['w'],
                    area['y'] + area['h'],
                    area['x']
                ]
            
            return jsonify({
                'name': best_match,
                'confidence': round(max_similarity, 3),
                'distance': round(1 - max_similarity, 3),
                'location': location
            })
        else:
            return jsonify({
                'name': None,
                'message': 'Không nhận diện được',
                'max_similarity': round(max_similarity, 3)
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_face():
    """Đăng ký khuôn mặt mới"""
    try:
        data = request.json
        name = data.get('name')
        image_base64 = data.get('image')
        
        if not name or not image_base64:
            return jsonify({'error': 'Thiếu tên hoặc ảnh'}), 400
        
        image = base64_to_image(image_base64)
        
        # Detect faces
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend="opencv",
            enforce_detection=True
        )
        
        if len(faces) == 0:
            return jsonify({'error': 'Không phát hiện khuôn mặt trong ảnh'}), 400
        
        if len(faces) > 1:
            return jsonify({'error': 'Phát hiện nhiều hơn 1 khuôn mặt'}), 400
        
        # Lấy embedding
        embedding = get_face_embedding(image)
        
        if embedding is None:
            return jsonify({'error': 'Không thể mã hóa khuôn mặt'}), 400
        
        # Lưu vào Firebase
        db.collection('employees').document(name).set({
            'name': name,
            'embedding': embedding.tolist(),
            'createdAt': firestore.SERVER_TIMESTAMP
        })
        
        # Cập nhật cache
        registered_faces[name] = embedding
        
        return jsonify({
            'success': True,
            'message': f'Đã đăng ký thành công: {name}',
            'total_employees': len(registered_faces)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Lấy danh sách nhân viên"""
    return jsonify({
        'employees': list(registered_faces.keys()),
        'total': len(registered_faces)
    })

@app.route('/api/employees/<name>', methods=['DELETE'])
def delete_employee(name):
    """Xóa nhân viên"""
    try:
        if name not in registered_faces:
            return jsonify({'error': 'Nhân viên không tồn tại'}), 404
        
        db.collection('employees').document(name).delete()
        del registered_faces[name]
        
        return jsonify({
            'success': True,
            'message': f'Đã xóa: {name}',
            'remaining': len(registered_faces)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance', methods=['POST'])
def check_attendance():
    """Chấm công"""
    try:
        data = request.json
        name = data.get('name')
        is_auto = data.get('is_auto', False)
        
        if not name:
            return jsonify({'error': 'Thiếu tên nhân viên'}), 400
        
        # Kiểm tra cooldown
        settings = db.collection('settings').document('attendance').get()
        cooldown = 30
        if settings.exists:
            cooldown = settings.to_dict().get('cooldownMinutes', 30)
        
        recent = db.collection('attendance')\
            .where('name', '==', name)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(1)\
            .get()
        
        if recent:
            last_time = recent[0].to_dict()['timestamp']
            time_diff = datetime.now() - last_time
            
            if time_diff < timedelta(minutes=cooldown):
                remaining = cooldown - int(time_diff.total_seconds() / 60)
                return jsonify({
                    'success': False,
                    'message': f'Đã chấm công rồi! Vui lòng chờ {remaining} phút nữa'
                }), 400
        
        # Lưu chấm công
        db.collection('attendance').add({
            'name': name,
            'type': 'checkin',
            'isAuto': is_auto,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            'success': True,
            'message': f'Chấm công thành công: {name}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reload', methods=['POST'])
def reload_faces():
    """Reload embeddings"""
    try:
        load_registered_faces()
        return jsonify({
            'success': True,
            'total': len(registered_faces)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Đang khởi động Face Recognition API (DeepFace)...")
    print("📦 Đang tải dữ liệu từ Firebase...")
    load_registered_faces()
    print("✅ Sẵn sàng!")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
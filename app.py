"""
Flask API Backend cho Face Recognition
Cài đặt:
pip install flask flask-cors face_recognition opencv-python pillow firebase-admin
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Cho phép frontend gọi API

# Khởi tạo Firebase Admin
import os
cred_path = os.environ.get('FIREBASE_CREDENTIALS', 'firebase-key.json')
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'face-timekeeping-31820.firebasestorage.app'
})

db = firestore.client()
bucket = storage.bucket()

# Cache khuôn mặt đã đăng ký trong memory
registered_faces = {}

def base64_to_image(base64_string):
    """Chuyển đổi base64 string sang numpy array"""
    # Loại bỏ header nếu có
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image)

def load_registered_faces():
    """Tải tất cả khuôn mặt đã đăng ký từ Firebase"""
    global registered_faces
    registered_faces = {}
    
    employees = db.collection('employees').stream()
    for emp in employees:
        data = emp.to_dict()
        name = data['name']
        descriptor = np.array(data['descriptor'])
        registered_faces[name] = descriptor
    
    print(f"✅ Đã tải {len(registered_faces)} nhân viên")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Kiểm tra API hoạt động"""
    return jsonify({
        'status': 'ok',
        'registered_employees': len(registered_faces)
    })

@app.route('/api/detect', methods=['POST'])
def detect_face():
    """
    Phát hiện khuôn mặt trong ảnh
    Input: { "image": "base64_string" }
    Output: { "faces_detected": int, "locations": [...] }
    """
    try:
        data = request.json
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'Thiếu ảnh'}), 400
        
        # Chuyển đổi và phát hiện
        image = base64_to_image(image_base64)
        face_locations = face_recognition.face_locations(image)
        
        return jsonify({
            'faces_detected': len(face_locations),
            'locations': face_locations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """
    Nhận diện khuôn mặt
    Input: { "image": "base64_string" }
    Output: { "name": "...", "confidence": 0.95, "distance": 0.4 }
    """
    try:
        data = request.json
        image_base64 = data.get('image')
        threshold = data.get('threshold', 0.6)
        
        if not image_base64:
            return jsonify({'error': 'Thiếu ảnh'}), 400
        
        # Chuyển đổi ảnh
        image = base64_to_image(image_base64)
        
        # Phát hiện và mã hóa khuôn mặt
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            return jsonify({'name': None, 'message': 'Không phát hiện khuôn mặt'})
        
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({'name': None, 'message': 'Không thể mã hóa khuôn mặt'})
        
        # So sánh với database
        face_encoding = face_encodings[0]
        best_match = None
        min_distance = threshold
        
        for name, known_encoding in registered_faces.items():
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        if best_match:
            confidence = 1 - min_distance
            return jsonify({
                'name': best_match,
                'confidence': round(confidence, 3),
                'distance': round(min_distance, 3),
                'location': face_locations[0]
            })
        else:
            return jsonify({
                'name': None,
                'message': 'Không nhận diện được',
                'min_distance': round(min_distance, 3)
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_face():
    """
    Đăng ký khuôn mặt mới
    Input: { "name": "...", "image": "base64_string" }
    Output: { "success": true, "message": "..." }
    """
    try:
        data = request.json
        name = data.get('name')
        image_base64 = data.get('image')
        
        if not name or not image_base64:
            return jsonify({'error': 'Thiếu tên hoặc ảnh'}), 400
        
        # Chuyển đổi ảnh
        image = base64_to_image(image_base64)
        
        # Phát hiện và mã hóa
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            return jsonify({'error': 'Không phát hiện khuôn mặt trong ảnh'}), 400
        
        if len(face_locations) > 1:
            return jsonify({'error': 'Phát hiện nhiều hơn 1 khuôn mặt. Vui lòng chỉ có 1 người trong ảnh'}), 400
        
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({'error': 'Không thể mã hóa khuôn mặt'}), 400
        
        face_encoding = face_encodings[0]
        
        # Lưu vào Firebase
        db.collection('employees').document(name).set({
            'name': name,
            'descriptor': face_encoding.tolist(),
            'createdAt': firestore.SERVER_TIMESTAMP
        })
        
        # Cập nhật cache
        registered_faces[name] = face_encoding
        
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
        
        # Xóa từ Firebase
        db.collection('employees').document(name).delete()
        
        # Xóa từ cache
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
    """
    Chấm công
    Input: { "name": "...", "is_auto": true/false }
    Output: { "success": true, "message": "..." }
    """
    try:
        data = request.json
        name = data.get('name')
        is_auto = data.get('is_auto', False)
        
        if not name:
            return jsonify({'error': 'Thiếu tên nhân viên'}), 400
        
        # Kiểm tra thời gian chờ
        settings = db.collection('settings').document('attendance').get()
        if settings.exists:
            config = settings.to_dict()
            cooldown = config.get('cooldownMinutes', 30)
        else:
            cooldown = 30
        
        # Kiểm tra lần chấm công gần nhất
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
    """Tải lại danh sách khuôn mặt từ Firebase"""
    try:
        load_registered_faces()
        return jsonify({
            'success': True,
            'total': len(registered_faces)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Đang khởi động Face Recognition API...")
    print("📦 Đang tải dữ liệu từ Firebase...")
    load_registered_faces()
    print("✅ Sẵn sàng!")
    
    # Lấy PORT từ biến môi trường (Render yêu cầu)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
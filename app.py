from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import requests

app = Flask(__name__)

# Load images and student IDs
path = 'C:\\Users\sondos\\AttendanceImages'
images = []
studentIds = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    studentIds.append(os.path.splitext(cl)[0])

# Function to find encodings of images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Load known encodings
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Function to check if the image is real
def isRealImage(img):
    # Calculate image quality score (for simplicity, you can use grayscale intensity)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_img)
    
    # Threshold to classify as real or fake
    threshold = 100  # Adjust as needed
    
    # Return True if image quality score is above threshold (considered real)
    return mean_intensity > threshold

# API endpoint for face recognition and attendance marking
@app.route('/recognize-face', methods=['POST'])
def recognize_face():
    # Get image data from the request
    image_data = request.files['image'].read()
    
    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check if the image is real using anti-spoofing mechanism
    if isRealImage(img):
        # Convert image to RGB (face_recognition uses RGB images)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        facesCurFrame = face_recognition.face_locations(img_rgb)
        encodesCurFrame = face_recognition.face_encodings(img_rgb, facesCurFrame)

        # Initialize empty list to store recognized IDs
        recognized_ids = []

        # Compare each face encoding with known encodings
        for encodeFace in encodesCurFrame:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            if True in matches:
                matchIndex = matches.index(True)
                recognized_ids.append(studentIds[matchIndex].upper())

        # Post result (ID and time) to backend .NET server
        backend_url = 'http://www.FinalAttendence.somee.com/api/Attendences'
        current_time = datetime.now().strftime('%H:%M:%S')
        for ids in recognized_ids:
            data = {'id': ids, 'time': current_time}
            response = requests.put(backend_url, json=data)

        # Return recognized IDs as JSON response
        return jsonify({'recognized_ids': recognized_ids})
    else:
        # Image is not real, do not mark attendance
        return jsonify({'recognized_ids': [], 'message': 'Image is not real, attendance not marked'})

if __name__ == '__main__':
    app.run(debug=True)

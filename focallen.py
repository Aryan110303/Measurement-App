import cv2
import numpy as np
import urllib.request as request

KNOWN_DISTANCE = 25.0  
KNOWN_WIDTH = 12.0     

url = 'http://192.168.0.105:8080/shot.jpg'
img = request.urlopen(url)
img_bytes = bytearray(img.read())
img_np = np.array(img_bytes, dtype=np.uint8)
frame = cv2.imdecode(img_np, -1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    perceived_width = w
    focal_length = (perceived_width * KNOWN_DISTANCE) / KNOWN_WIDTH
    print(f"Calculated Focal Length: {focal_length:.2f} pixels")
else:
    print("Object not detected. Try again.")

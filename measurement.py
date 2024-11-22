import cv2
import numpy as np
import urllib.request as request

KNOWN_WIDTH = 2.0
FOCAL_LENGTH = 100  

url = 'http://192.168.0.105:8080/shot.jpg'

while True:
    img = request.urlopen(url)
    img_bytes = bytearray(img.read())
    img_np = np.array(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_np, -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        perceived_width = w

        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Virtual Measuring Tape", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv2.destroyAllWindows()

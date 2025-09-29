## [v1.0.1] - 2025-09-27
### Added
# - Frame movement tracking (Left, Right, No Motion, No Detection)
# - Direction display on top left of the screen

import cv2

# Load Haar cascade
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Start camera
cap = cv2.VideoCapture(0)

prev_x = None
direction = ""
error_threshold = 10  # pixels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    if len(bodies) > 0:
        (x, y, w, h) = bodies[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x = x + w // 2

        if prev_x is not None:
            diff = center_x - prev_x
            if abs(diff) > error_threshold:
                direction = "Right" if diff > 0 else "Left"
            else:
                direction = "No Motion"
        prev_x = center_x
    else:
        direction = "No Detection"
        prev_x = None

    # Display direction
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, direction, (10, 30), font, 0.7, (0, 0, 255), 2)

    cv2.imshow("NDsight - Movement Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
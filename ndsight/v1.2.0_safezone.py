## [v1.2.0] - 2025-09-27
### Added
# - Safe zone overlay (green triangle) to complement red warning zone
# - Visual feedback for "No Detection" when no person is present
# - Improved zone-based logic for clearer monitoring

### Notes
# - Safe zone helps distinguish between alert and normal areas
# - Direction tracking and red zone warning retained from v1.1.0

import cv2
import numpy as np

# Load the full body Haar cascade
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Start camera
cap = cv2.VideoCapture(0)

prev_x = None
direction = ""
error_threshold = 10  # pixels, to ignore small haar errors

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the green detection zone (triangle from the left corner, semi-transparent)
    overlay = frame.copy()
    zone_width = frame.shape[1] // 2
    points = np.array([[0, 0], [zone_width, 0], [0, frame.shape[0]]], dtype=np.int32)
    cv2.fillPoly(overlay, [points], (0, 255, 0))
    alpha = 0.2
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # draw another triangle for the green zone but on the right side
    points = np.array([[frame.shape[1], 0], [frame.shape[1] - zone_width, 0], [frame.shape[1], frame.shape[0]]], dtype=np.int32)
    cv2.fillPoly(overlay, [points], (0, 255, 0))
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Detect full bodies
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    in_green_zone = False

    # If at least one body is detected, use the first one
    if len(bodies) > 0:
        (x, y, w, h) = bodies[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x = x + w // 2
        if center_x < zone_width:
            in_green_zone = True
        if prev_x is not None:
            diff = center_x - prev_x
            if abs(diff) > error_threshold:
                if diff > 0:
                    direction = "Moving Right"
                else:
                    direction = "Moving Left"
            else:
                direction = "Stationary"
        prev_x = center_x
    else:
        direction = "No Detection"
        prev_x = None

    # Put direction text on the top right of the screen just shorten name like left, right, no motion 
    if direction == "Moving Right":
        text = "Right"
    elif direction == "Moving Left":
        text = "Left"
    elif direction == "Stationary":
        text = "No Motion"
    else:
        text = "No Detection"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_x = 10       # 10 pixels from the left
    text_y = 30       # 30 pixels from the top
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    # If in green zone, print warning in the center
    if in_green_zone:
        warning_text = "Safe"
        warn_size, _ = cv2.getTextSize(warning_text, font, 1.2, 3)
        warn_x = (frame.shape[1] - warn_size[0]) // 2
        warn_y = frame.shape[0] // 2
        cv2.putText(frame, warning_text, (warn_x, warn_y), font, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("NDsight - Full Body Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


## [v1.3.1] - 2025-09-27
### Changed
# - Removed green triangles, only orange warning triangle and red danger zone are drawn
# - Orange warning triangle: tip at top center, base at bottom corners
# - Red danger zone: trapezoid at bottom, top edge at 45 degrees from corners
# - Status text now shows 'Safe' if detection is outside both warning and danger zones

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

    # Draw only the orange warning triangle and red danger zone
    overlay = frame.copy()
    h, w = frame.shape[:2]
    # Orange warning triangle (tip at top center, base at bottom corners)
    warning_tip = (w // 2, 0)
    warning_left = (0, h)
    warning_right = (w, h)
    warning_points = np.array([warning_tip, warning_left, warning_right], dtype=np.int32)
    cv2.fillPoly(overlay, [warning_points], (0, 165, 255))

    # Red danger zone (trapezoid at the bottom, top edge matches triangle's sides)
    danger_top_y = h // 2
    # Find intersection of triangle sides with y = danger_top_y
    # Left side: from (w//2, 0) to (0, h)
    x1, y1 = w // 2, 0
    x2, y2 = 0, h
    if y2 != y1:
        left_x = int(x1 + (danger_top_y - y1) * (x2 - x1) / (y2 - y1))
    else:
        left_x = x1
    # Right side: from (w//2, 0) to (w, h)
    x3, y3 = w, h
    if y3 != y1:
        right_x = int(x1 + (danger_top_y - y1) * (x3 - x1) / (y3 - y1))
    else:
        right_x = x1
    danger_top_left = (left_x, danger_top_y)
    danger_top_right = (right_x, danger_top_y)
    danger_bottom_left = (0, h)
    danger_bottom_right = (w, h)
    danger_points = np.array([
        danger_top_left,
        danger_top_right,
        danger_bottom_right,
        danger_bottom_left
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [danger_points], (0, 0, 255))
    alpha = 0.2
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ...existing code...

    # Detect full bodies
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    in_green_zone = False
    in_warning_zone = False
    in_danger_zone = False

    # If at least one body is detected, use the first one
    if len(bodies) > 0:
        (x, y, bw, bh) = bodies[0]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        center_x = x + bw // 2
        center_y = y + bh // 2
        pt = (float(center_x), float(center_y))
        # Check danger zone (red trapezoid) first
        if cv2.pointPolygonTest(danger_points.astype(np.float32), pt, False) >= 0:
            in_danger_zone = True
        # Check warning zone (orange triangle)
        elif cv2.pointPolygonTest(warning_points.astype(np.float32), pt, False) >= 0:
            in_warning_zone = True
        else:
            in_green_zone = True
        # Direction logic
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

    # Show zone status in the center (Danger > Warning > Safe)
    zone_text = None
    zone_color = None
    if in_danger_zone:
        zone_text = "Danger"
        zone_color = (0, 0, 255)
    elif in_warning_zone:
        zone_text = "Warning"
        zone_color = (0, 165, 255)
    elif in_green_zone:
        zone_text = "Safe"
        zone_color = (0, 255, 0)
    if zone_text:
        warn_size, _ = cv2.getTextSize(zone_text, font, 1.2, 3)
        warn_x = (frame.shape[1] - warn_size[0]) // 2
        warn_y = frame.shape[0] // 2
        cv2.putText(frame, zone_text, (warn_x, warn_y), font, 1.2, zone_color, 3, cv2.LINE_AA)

    cv2.imshow("NDsight - Full Body Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

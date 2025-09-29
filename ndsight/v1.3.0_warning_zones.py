## [v1.3.0] - 2025-09-27
### Added
# - Orange warning zone (middle triangle) and red danger zone (bottom center trapezoid) overlays
# - Central warning text: "Safe", "Warning", or "Danger" based on detected body position

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

    # Draw the green detection zones (left and right triangles), orange warning zone (middle triangle), and red danger zone (bottom trapezoid)
    overlay = frame.copy()
    h, w = frame.shape[:2]
    zone_width = w // 2

    # Green left triangle
    points_left = np.array([[0, 0], [zone_width, 0], [0, h]], dtype=np.int32)
    cv2.fillPoly(overlay, [points_left], (0, 255, 0))

    # Green right triangle
    points_right = np.array([[w, 0], [w - zone_width, 0], [w, h]], dtype=np.int32)
    cv2.fillPoly(overlay, [points_right], (0, 255, 0))

    # Orange warning triangle (tip at top center, base at bottom between green triangles)
    warning_base_y = h - h // 3
    warning_left = (zone_width // 2, warning_base_y)
    warning_right = (w - zone_width // 2, warning_base_y)
    warning_tip = (w // 2, 0)
    warning_points = np.array([warning_tip, warning_left, warning_right], dtype=np.int32)
    cv2.fillPoly(overlay, [warning_points], (0, 165, 255))
    
    # Red danger zone (trapezoid at the bottom, top edge is the base of the orange triangle)
    danger_top_left = warning_left
    danger_top_right = warning_right
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
        # Check green zones (left or right)
        if center_x < zone_width or center_x > (frame.shape[1] - zone_width):
            in_green_zone = True
        # Check warning zone (middle triangle)
        elif cv2.pointPolygonTest(warning_points, (center_x, center_y), False) >= 0:
            in_warning_zone = True
        # Check danger zone (bottom center trapezoid)
        elif cv2.pointPolygonTest(danger_points, (center_x, center_y), False) >= 0:
            in_danger_zone = True
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

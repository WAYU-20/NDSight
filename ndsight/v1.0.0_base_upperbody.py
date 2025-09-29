## [v1.0.0] - Initial Release
### Added
# - Haar cascade detection using live camera feed
# - Basic bounding box display for detected humans

import cv2

# Load Haar cascade for full body or upper body (you can change this to face if needed)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for Haar detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects
    detections = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected objects
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("NDsight v1.0.0 - Base Eye", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("ArASL-Detection-master.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO prediction
    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Show window
    cv2.imshow("ArASL Real-Time Detection", annotated_frame)

    # Exit on Esc key (27)
    k = cv2.waitKey(0)
    if k == 27:  # Esc key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
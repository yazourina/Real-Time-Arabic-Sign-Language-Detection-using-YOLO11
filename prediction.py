import cv2
from ultralytics import YOLO


model = YOLO("ArASL-Detection-master.pt")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    
    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)

    
    annotated_frame = results[0].plot()  

    
    cv2.imshow("ArASL Real-Time Detection", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

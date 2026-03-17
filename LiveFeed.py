import cv2
from ultralytics import YOLO

model = YOLO(r"") # Load the trained YOLOv8 model (pretrained model included in runs/detect/train/weights/best.pt)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[class_id]} ({prob})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("Object Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

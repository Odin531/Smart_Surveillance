import cv2
from ultralytics import YOLO
import pygame
import numpy as np

# Initialize YOLO
yolo = YOLO("D:/IIT_ROORKEE/myenv/runs/detect/train9/weights/best.pt")

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((640, 320))  # adjust to your video size if needed

video_path = "D:\IIT_ROORKEE\myenv\sample.mp4"
videoCap = cv2.VideoCapture(video_path)
if not videoCap.isOpened():
    print("not getting the vedio")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = videoCap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 320))
    results = yolo.track(frame, verbose=False)

    screen.fill((0, 0, 0))  # clear screen

    # Loop through detections
    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.4:  # confidence threshold
                cls = int(box.cls[0])
                class_name = result.names[cls]

                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Draw rectangle using OpenCV
                color = (0, 0, 255) if class_name.lower() == "accident" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Put label text
                label = f"{class_name} {box.conf[0]:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # If the detected class is "accident"
                if class_name.lower() == "accident":
                    print("Accident")

    # Show video with bounding boxes
    cv2.imshow("YOLO Detection", frame)
    pygame.display.flip()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()
pygame.quit()
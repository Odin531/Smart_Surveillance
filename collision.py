import cv2
import random
from ultralytics import YOLO
import pygame
import math
# Initialize YOLO
yolo = YOLO("yolov8s.pt")

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((640, 320))  # adjust to your video size if needed

def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

video_path = "sample1.mp4"
videoCap = cv2.VideoCapture(video_path)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = videoCap.read()
    frame = cv2.resize(frame,(640,320))
    if not ret:
        break

    results = yolo.track(frame,verbose =False)

    rects = []
    screen.fill((0, 0, 0))  # clear screen

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])*100
                colour = getColours(cls)

                # Draw on OpenCV frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{class_name} {conf:0.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)

                # Create pygame rect
                rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
                rects.append(rect)
                pygame.draw.rect(screen, colour, rect, 2)

    # Collision detection
    threshold = 5  # pixels, adjust as needed
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            c1 = rects[i].center
            c2 = rects[j].center
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            if dist <= threshold:
                if class_name == "car":
                   print("collided")


    # Show OpenCV window (optional)
    cv2.imshow("YOLO Detection", frame)

    # Update Pygame window
    pygame.display.flip()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()
pygame.quit()
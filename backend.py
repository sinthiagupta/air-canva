import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pytesseract
import time

# For gesture tracking
y_positions = deque(maxlen=5)
scroll_cooldown = time.time()
page_number = 1

# Canvas settings
CANVAS_HEIGHT = 471
CANVAS_WIDTH = 636

# Pen thickness
pen_thickness = 2

# Lists to store drawings for multiple pages
pages = []
current_page_index = 0

# Button definitions
buttons = [
    (40, 1, 140, 65, "CLEAR", (0, 0, 0)),
    (160, 1, 255, 65, "BLUE", (255, 0, 0)),
    (275, 1, 370, 65, "GREEN", (0, 255, 0)),
    (390, 1, 485, 65, "RED", (0, 0, 255)),
    (505, 1, 600, 65, "YELLOW", (0, 255, 255))
]

# Helper to create a blank canvas
def create_blank_canvas():
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255
    for x1, y1, x2, y2, label, color in buttons:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1 + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas

# Initialize drawing buffers
def initialize_points():
    return [deque(maxlen=1024)], [deque(maxlen=1024)], [deque(maxlen=1024)], [deque(maxlen=1024)], 0, 0, 0, 0

bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index = initialize_points()

kernel = np.ones((5, 5), np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# First page
paintWindow = create_blank_canvas()
pages.append(paintWindow)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
ret = True

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw buttons
    for x1, y1, x2, y2, label, color in buttons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = [[int(lm.x * 640), int(lm.y * 480)] for lm in handslms.landmark]
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger = tuple(landmarks[8])
            thumb = tuple(landmarks[4])
            middle_tip = landmarks[12]
            wrist = landmarks[0]

            cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)

            # Gesture: Tap to start new stroke
            if abs(thumb[1] - fore_finger[1]) < 30:
                for plist in [bpoints, gpoints, rpoints, ypoints]:
                    plist.append(deque(maxlen=512))
                blue_index += 1
                green_index += 1
                red_index += 1
                yellow_index += 1

            # Gesture: Button click
            elif fore_finger[1] <= 65:
                for i, (x1, _, x2, _, label, _) in enumerate(buttons):
                    if x1 <= fore_finger[0] <= x2:
                        if label == "CLEAR":
                            bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index = initialize_points()
                            pages[current_page_index] = create_blank_canvas()
                        else:
                            colorIndex = i - 1
                        break

            # Gesture: Scroll down
            elif (wrist[1] - middle_tip[1]) > 100:
                current_page_index += 1
                if current_page_index >= len(pages):
                    pages.append(create_blank_canvas())
                bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index = initialize_points()
                paintWindow = pages[current_page_index]

            else:
                [bpoints, gpoints, rpoints, ypoints][colorIndex][[blue_index, green_index, red_index, yellow_index][colorIndex]].appendleft(fore_finger)
    else:
        for plist in [bpoints, gpoints, rpoints, ypoints]:
            plist.append(deque(maxlen=512))
        blue_index += 1
        green_index += 1
        red_index += 1
        yellow_index += 1

    # Draw lines
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], pen_thickness)
                cv2.line(pages[current_page_index], points[i][j][k - 1], points[i][j][k], colors[i], pen_thickness)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", pages[current_page_index])

    # Handle keyboard events
    key = cv2.waitKeyEx(1)

    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        pen_thickness = min(20, pen_thickness + 1)
    elif key == ord('-') or key == ord('_'):
        pen_thickness = max(1, pen_thickness - 1)
    elif key == 2621440:  # Down arrow
        current_page_index += 1
        if current_page_index >= len(pages):
            pages.append(create_blank_canvas())
        bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index = initialize_points()
    elif key == 2490368:  # Up arrow
        if current_page_index > 0:
            current_page_index -= 1
            bpoints, gpoints, rpoints, ypoints, blue_index, green_index, red_index, yellow_index = initialize_points()

cap.release()
cv2.destroyAllWindows()
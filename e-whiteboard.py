import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Giving different arrays to handle colour points of different colours
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Canvas setup
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
buttons = [(40, 1, 140, 65, "CLEAR", (0, 0, 0)),
           (160, 1, 255, 65, "BLUE", (255, 0, 0)),
           (275, 1, 370, 65, "GREEN", (0, 255, 0)),
           (390, 1, 485, 65, "RED", (0, 0, 255)),
           (505, 1, 600, 65, "YELLOW", (0, 255, 255))]

for x1, y1, x2, y2, label, color in buttons:
    cv2.rectangle(paintWindow, (x1, y1), (x2, y2), color, 2)
    cv2.putText(paintWindow, label, (x1 + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for x1, y1, x2, y2, label, color in buttons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = [[int(lm.x * 640), int(lm.y * 480)] for lm in handslms.landmark]
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger, thumb = tuple(landmarks[8]), tuple(landmarks[4])
            cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)
            
            if thumb[1] - fore_finger[1] < 30:
                for points_list in [bpoints, gpoints, rpoints, ypoints]:
                    points_list.append(deque(maxlen=512))
                blue_index += 1
                green_index += 1
                red_index += 1
                yellow_index += 1
            
            elif fore_finger[1] <= 65:
                for i, (x1, _, x2, _, label, _) in enumerate(buttons):
                    if x1 <= fore_finger[0] <= x2:
                        if label == "CLEAR":
                            bpoints, gpoints, rpoints, ypoints = [[deque(maxlen=512)] for _ in range(4)]
                            blue_index, green_index, red_index, yellow_index = 0, 0, 0, 0
                            paintWindow[67:, :, :] = 255
                        else:
                            colorIndex = i - 1  # Adjusting for CLEAR button
                        break
            else:
                [bpoints, gpoints, rpoints, ypoints][colorIndex][[blue_index, green_index, red_index, yellow_index][colorIndex]].appendleft(fore_finger)
    else:
        for points_list in [bpoints, gpoints, rpoints, ypoints]:
            points_list.append(deque(maxlen=512))
        blue_index += 1
        green_index += 1
        red_index += 1
        yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

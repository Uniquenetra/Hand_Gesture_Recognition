import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

gesture = []

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks)
            temp = []
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                # print(len(hand_landmarks.landmark))
                if hand_no == 0:
                    for data_point in hand_landmarks.landmark:
                        temp.extend([data_point.x, data_point.y, data_point.z])
                        if len(temp) == 3:
                            temp.extend([0, 0, 0])

            # 0, 5, 17
            if len(temp) == 66:
                a, b, c = 0, 5*3, 17*3
                palm = [(temp[a] + temp[a+1] + temp[a+2])/3, (temp[b] + temp[b+1] + temp[b+2])/3, (temp[c] + temp[c+1] + temp[c+2])/3]
                temp[3:6] = palm
                gesture.append(temp)
                print(len(gesture))
                if len(gesture) == 100:
                    break
            else:
                print("Not 21 landmarks ")

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
        cv2.waitKey(5) 
        if cv2.waitKey(1) & 0xFF == 27:
           break
cap.release()

import csv
with open("new_gesture.csv", "w+", newline='') as f:
    write = csv.writer(f) 
    write.writerows(gesture) 

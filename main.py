import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

while True:
    # Read the frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = mp_hands.Hands().process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if(results.multi_hand_landmarks):
        for hand_landmarks in results.multi_hand_landmarks:
          print(hand_landmarks)            
    # if(results.multi_hand_landmarks):
        # for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame,hand_landmarks,connections=mp_hands.HAND_CONNECTIONS)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
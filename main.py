import cv2
import mediapipe as mp
from mediapipe.tasks import python

cap = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# Create a hand landmarker instance with the image mode:
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    centerX = 0
    centerY = 0
    count = 0


    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        for hand in mp_hands.HandLandmark:
            centerX += hand_landmarks.landmark[hand].x
            centerY += hand_landmarks.landmark[hand].y
            count += 1

    if (count):
        centerX = centerX / count
        centerY = centerY / count
    

    print(centerX,centerY)

    if cv2.waitKey(25) & 0xFF == ord('r'):
      break

cap.release()
cv2.destroyAllWindows()
# WRIST = 0  hand body
# THUMB_CMC = 1 tump(1-4)
# THUMB_MCP = 2
# THUMB_IP = 3
# THUMB_TIP = 4
# INDEX_FINGER_MCP = 5 index(5-8)
# INDEX_FINGER_PIP = 6
# INDEX_FINGER_DIP = 7
# INDEX_FINGER_TIP = 8
# MIDDLE_FINGER_MCP = 9 middle(9-12)
# MIDDLE_FINGER_PIP = 10
# MIDDLE_FINGER_DIP = 11
# MIDDLE_FINGER_TIP = 12
# RING_FINGER_MCP = 13 ring(13-16)
# RING_FINGER_PIP = 14
# RING_FINGER_DIP = 15
# RING_FINGER_TIP = 16
# PINKY_MCP = 17 pinky(17-20)
# PINKY_PIP = 18
# PINKY_DIP = 19
# PINKY_TIP = 20

import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np

xml = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(3, 640) # webcam width
cap.set(4, 480) # webcam height

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

 
  while cap.isOpened():
    success, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.05, 5)
    
    if len(faces):
      for (x,y,w,h) in faces:
        face_img = image[y:y+h, x:x+w] # crop the face image
        face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04)
        face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
        image[y:y+h, x:x+w] = face_img
        
    if not success:
      print("Ignoring empty camera frame.")

      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)


    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:


        # # When the bone of each joint is above the bone below it, the state is changed to 1 to change the state of the finger being extended.
        thumb_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height:
              thumb_finger_state = 1

        index_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
              index_finger_state = 1

        middle_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
              middle_finger_state = 1

        ring_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
              ring_finger_state = 1

        pinky_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height:
              pinky_finger_state = 1


 
        font = ImageFont.truetype("fonts/gulim.ttc", 80)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        # Set 5, 4, 3, 2, 1 using 1 and 0 based on the if statement above.
        text = ""
        if thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
          text = "5"
        elif index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
          text = "4"
        elif index_finger_state == 0 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
          text = "3"
        elif index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 1 and pinky_finger_state == 1:
          text = "2"
        elif index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 1:
          text = "1"
        elif index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
          text = "0"
          
        # Complete the frame of your finger
        w, h = font.getsize(text)

        x = 50
        y = 50

        draw.text((x, y),  text, font=font, fill=(255, 255, 255))
        image = np.array(image)


        # Flip the image horizontally for a selfie-view display.
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('MediaPipe Hands', image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()

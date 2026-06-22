import cv2
import numpy as np
import sys
from keras.models import load_model
from pynput.keyboard import Key, Controller
from time import sleep

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

PROGRAMM_NAME = "media_control"
FLIP_CAMERA = True
BOX_SCALE = 0.5 # Needs to be between 0 and 1
IMG_SIZE = 64
COOLDOWN = 8

model = load_model("gesture_recognition.keras")
labels = ['like', 'no_gesture', 'dislike', 'stop', 'fist']

controller = Controller()

# Get window props
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
middle_x = w // 2
middle_y = h // 2

# Define the detection area
area_size = int(min(w, h) * BOX_SCALE)
area_top_x = middle_x - (area_size // 2)
area_top_y = middle_y - (area_size // 2)

previous = ''
counter = 0

# Simply check if enough of the hand is in the box
def check_for_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))
    zell_count = cv2.countNonZero(mask)
    max_count = mask.size

    # At least 10% need to look like the hand
    return zell_count / max_count > 0.1

# The main detection methode
def detection():
    global previous, counter

    ret, frame = cap.read()
    if not ret:
        return ''

    if FLIP_CAMERA:
        frame = cv2.flip(frame, 1)

    detection_area = frame.copy()[area_top_y:area_top_y + area_size, area_top_x:area_top_x + area_size]

    # Check if the hand is even in the area at all
    hand_detected = check_for_hand(detection_area)

    if hand_detected:
        cv2.rectangle(frame, (area_top_x, area_top_y), (area_top_x + area_size, area_top_y + area_size), (255, 0, 0), 2)
    else:
        cv2.rectangle(frame, (area_top_x, area_top_y), (area_top_x + area_size, area_top_y + area_size), (0, 255, 0), 2)
        cv2.imshow(PROGRAMM_NAME, frame)
        return ''

    # Detect hand gesture
    detection_area = cv2.resize(detection_area, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    prediction = model.predict(detection_area, verbose=0)
    prediction_label = labels[np.argmax(prediction)]
    prediction_confidence = np.max(prediction)

    # Show text
    cv2.putText(frame, f"Pred: {prediction_label} ({counter})", (area_top_x, area_top_y - 10), 0, 1, (255, 0, 0))
    cv2.imshow(PROGRAMM_NAME, frame)

    # Cooldown to prevent unintentional presses
    if prediction_label == previous:
        if counter == COOLDOWN:
            counter = 0
            return prediction_label
        else:
            counter = counter + 1
    else:
        counter = 0
        previous = prediction_label

    return ''

while True:

    # Detect pose
    pose = detection()
    if pose == 'like':
        controller.press(Key.media_volume_up)
        controller.release(Key.media_volume_up)
        print("The Volume should be louder now")
    elif pose == 'dislike':
        controller.press(Key.media_volume_down)
        controller.release(Key.media_volume_down)
        print("The Volume should be quieter now")
    elif pose == 'stop':
        controller.press(Key.media_next)
        controller.release(Key.media_next)
        print("The track should skip now")
    elif pose == 'fist':
        controller.press(Key.media_play_pause)
        controller.release(Key.media_play_pause)
        print("The audio should play/pause now")

    # Allow to quit with q
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    sleep(0.1)


cv2.destroyAllWindows()
cap.release()

#current code
#current code


"""
#object indexes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
           "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]
"""


import cv2
import numpy as np
import pyttsx3
from queue import Queue
import time

# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32

# Load the pre-trained model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Initialize text-to-speech engine once
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate (words per minute)

# Queue for speech instructions
speech_queue = Queue()

# Cooldown time (in seconds) between instructions
INSTRUCTION_COOLDOWN = 2.0
last_instruction_time = 0
last_zone = None  # Track the last significant position zone

# Open a connection to the ESP32 stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Connected to video stream. Press 'q' to quit.")

# Set confidence, proximity, and size thresholds
CONFIDENCE_THRESHOLD = 0.2
MIN_BOX_AREA = 3000  # Minimum bounding box area for proximity filter
MAX_BOX_AREA = 50000  # Maximum bounding box area

# Track the last spoken instruction to avoid repeated announcements
last_instruction = ""
prev_boxes = []

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not retrieve frame.")
            break

        # Prepare the frame and detect objects
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        current_boxes = []
        obstacle_detected = False
        navigation_instruction = ""
        current_zone = None

        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])

            # Check for high confidence and specific object classes (e.g., chair or table)
            if confidence > CONFIDENCE_THRESHOLD and (idx == 9):  # Only detect "chair" (index 9)
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                box_area = (endX - startX) * (endY - startY)

                if MIN_BOX_AREA < box_area < MAX_BOX_AREA:
                    current_boxes.append((startX, startY, endX, endY))
                    obstacle_detected = True

                    # Determine obstacle position relative to frame
                    box_center = (startX + endX) // 2
                    if box_center < w // 5:
                        navigation_instruction = "Obstacle far left. Move sharply right."
                        current_zone = "far left"
                    elif box_center < 2 * w // 5:
                        navigation_instruction = "Obstacle on the left. Move slightly right."
                        current_zone = "left"
                    elif box_center < 3 * w // 5:
                        navigation_instruction = "Obstacle ahead. Choose left or right."
                        current_zone = "center"
                    elif box_center < 4 * w // 5:
                        navigation_instruction = "Obstacle on the right. Move slightly left."
                        current_zone = "right"
                    else:
                        navigation_instruction = "Obstacle far right. Move sharply left."
                        current_zone = "far right"

                    # Draw bounding box for visual confirmation
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    break

        # Only add the instruction to the queue if cooldown has passed and position changed
        current_time = time.time()
        if (
            obstacle_detected 
            and navigation_instruction != last_instruction 
            and (current_time - last_instruction_time > INSTRUCTION_COOLDOWN or current_zone != last_zone)
        ):
            print(f"New instruction: {navigation_instruction}")
            speech_queue.put(navigation_instruction)  # Add instruction to queue
            last_instruction = navigation_instruction
            last_instruction_time = current_time
            last_zone = current_zone  # Update the zone

        # Process speech queue in main loop
        if not speech_queue.empty():
            instruction_to_speak = speech_queue.get()
            engine.say(instruction_to_speak)
            engine.runAndWait()

        # Apply smoothing to bounding boxes
        smoothed_boxes = []
        for i, box in enumerate(current_boxes):
            if i < len(prev_boxes):
                smooth_box = (
                    int(0.5 * box[0] + 0.5 * prev_boxes[i][0]),
                    int(0.5 * box[1] + 0.5 * prev_boxes[i][1]),
                    int(0.5 * box[2] + 0.5 * prev_boxes[i][2]),
                    int(0.5 * box[3] + 0.5 * prev_boxes[i][3]),
                )
                smoothed_boxes.append(smooth_box)
            else:
                smoothed_boxes.append(box)

        for (startX, startY, endX, endY) in smoothed_boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        prev_boxes = current_boxes
        cv2.imshow("Detecting Things", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()


"""
import cv2
import numpy as np
import pyttsx3
import threading
from queue import Queue

# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32

# Load the pre-trained model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')



engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate (words per minute)

# Queue for speech instructions
speech_queue = Queue()

# Function to continuously check the queue for instructions and speak them
def speak_from_queue():
    while True:
        instruction = speech_queue.get()  # Get the next instruction
        if instruction == "STOP":  # Check for termination signal
            break
        engine.say(instruction)
        engine.runAndWait()
        speech_queue.task_done()  # Mark the task as done

# Start the speech thread
speech_thread = threading.Thread(target=speak_from_queue, daemon=True)
speech_thread.start()



# Open a connection to the ESP32 stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Connected to video stream. Press 'q' to quit.")



CONFIDENCE_THRESHOLD = 0.2
MIN_BOX_AREA = 3000  #for prox. filter
MAX_BOX_AREA = 50000

prev_boxes = []

last_instruction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame.")
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    detections = net.forward()

    current_boxes = []

    obstacle_detected = False
    navigation_instruction = ""

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if confidence > CONFIDENCE_THRESHOLD and (idx==9): #9,11 chair, table
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            box_area = (endX - startX) * (endY - startY)

            # proximity filter
            if box_area > MIN_BOX_AREA:
                current_boxes.append((startX, startY, endX, endY))

            # Filter by proximity (box area size)
            if MIN_BOX_AREA < box_area < MAX_BOX_AREA:
                obstacle_detected = True

                # Determine obstacle position relative to frame
                box_center = (startX + endX) // 2

                if box_center < w // 5:
                    navigation_instruction = "Obstacle far left. Move sharply right."
                elif box_center < 2 * w // 5:
                    navigation_instruction = "Obstacle on the left. Move slightly right."
                elif box_center < 3 * w // 5:
                    navigation_instruction = "Obstacle ahead. Choose left or right."
                elif box_center < 4 * w // 5:
                    navigation_instruction = "Obstacle on the right. Move slightly left."
                else:
                    navigation_instruction = "Obstacle far right. Move sharply left."


                # Draw bounding box for visual confirmation
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                break  # Stop after first detected obstacle to avoid multiple instructions
    

    # Announce the navigation instruction only if it changes
    if obstacle_detected and navigation_instruction != last_instruction:
        speech_queue.put(navigation_instruction)
        last_instruction = navigation_instruction

    if not obstacle_detected:
        last_instruction = ""


    # making it super smooth
    smoothed_boxes = []
    for i, box in enumerate(current_boxes):
        if i < len(prev_boxes):

            smooth_box = (
                int(0.5 * box[0] + 0.5 * prev_boxes[i][0]),
                int(0.5 * box[1] + 0.5 * prev_boxes[i][1]),
                int(0.5 * box[2] + 0.5 * prev_boxes[i][2]),
                int(0.5 * box[3] + 0.5 * prev_boxes[i][3]),
            )
            smoothed_boxes.append(smooth_box)
        else:
            smoothed_boxes.append(box)

    for (startX, startY, endX, endY) in smoothed_boxes:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    prev_boxes = current_boxes
    cv2.imshow("Detecting Things", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    


cap.release()
cv2.destroyAllWindows()
"""

"""
-----------
#object indexes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
           "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]
"""

"""
import cv2
import numpy as np

# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32

# Load the pre-trained model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Open a connection to the ESP32 stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Connected to video stream. Press 'q' to quit.")


CONFIDENCE_THRESHOLD = 0.2
MIN_BOX_AREA = 3000  #for prox. filter

prev_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame.")
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    detections = net.forward()

    current_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        if confidence > CONFIDENCE_THRESHOLD and (idx==9 or index==11): #chair, table
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            box_area = (endX - startX) * (endY - startY)

            # proximity filter
            if box_area > MIN_BOX_AREA:
                current_boxes.append((startX, startY, endX, endY))
                

    # making it super smooth
    smoothed_boxes = []
    for i, box in enumerate(current_boxes):
        if i < len(prev_boxes):

            smooth_box = (
                int(0.5 * box[0] + 0.5 * prev_boxes[i][0]),
                int(0.5 * box[1] + 0.5 * prev_boxes[i][1]),
                int(0.5 * box[2] + 0.5 * prev_boxes[i][2]),
                int(0.5 * box[3] + 0.5 * prev_boxes[i][3]),
            )
            smoothed_boxes.append(smooth_box)
        else:
            smoothed_boxes.append(box)

    for (startX, startY, endX, endY) in smoothed_boxes:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    prev_boxes = current_boxes
    cv2.imshow("Detecting Things", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

"""
#final iter 2
import cv2
import numpy as np

# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32

# Load the pre-trained model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Open a connection to the ESP32 stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Connected to video stream. Press 'q' to quit.")

# Set confidence and proximity thresholds
CONFIDENCE_THRESHOLD = 0.6
MIN_BOX_AREA = 3000  # Minimum bounding box area to filter out distant objects

# Initialize variables for smoothing bounding boxes
prev_boxes = []

# Process the video stream frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame.")
        break

    # Resize frame and prepare the input blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Run forward pass to get detections
    detections = net.forward()

    # Initialize a list for current frame's bounding boxes
    current_boxes = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter by confidence
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            box_area = (endX - startX) * (endY - startY)

            # Filter by proximity (box area size)
            if box_area > MIN_BOX_AREA:
                current_boxes.append((startX, startY, endX, endY))

    # Smooth boxes by averaging with previous frame's boxes
    smoothed_boxes = []
    for i, box in enumerate(current_boxes):
        if i < len(prev_boxes):
            # Average current and previous coordinates for a smoother transition
            smooth_box = (
                int(0.5 * box[0] + 0.5 * prev_boxes[i][0]),
                int(0.5 * box[1] + 0.5 * prev_boxes[i][1]),
                int(0.5 * box[2] + 0.5 * prev_boxes[i][2]),
                int(0.5 * box[3] + 0.5 * prev_boxes[i][3]),
            )
            smoothed_boxes.append(smooth_box)
        else:
            # If there's no previous box, just use the current one
            smoothed_boxes.append(box)

    # Draw bounding boxes
    for (startX, startY, endX, endY) in smoothed_boxes:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Update previous boxes for the next frame
    prev_boxes = current_boxes

    # Display the frame with detections
    cv2.imshow("ESP32 Camera Stream with Object Detection", frame)

    # Press 'q' to quit the video stream display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
"""



"""
#final iter 1
import cv2
import numpy as np

# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32

# Load the pre-trained model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# List of class labels MobileNet SSD model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
           "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]

# Set colors for each detected object
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Open a connection to the ESP32 stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Connected to video stream. Press 'q' to quit.")

# Set confidence and proximity thresholds
CONFIDENCE_THRESHOLD = 0.6  # Increase to be more confident
MIN_BOX_AREA = 3000         # Minimum bounding box area to filter out distant objects

# Process the video stream frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame.")
        break

    # Resize frame and prepare the input blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Run forward pass to get detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter by confidence
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            box_area = (endX - startX) * (endY - startY)

            # Filter by proximity (box area size)
            if box_area > MIN_BOX_AREA:
                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

                # Draw label with black text
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow("ESP32 Camera Stream with Object Detection", frame)

    # Press 'q' to quit the video stream display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
"""



"""
#working detect with cv2 and feed
import cv2
import numpy as np

# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32

# Load the pre-trained model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# List of class labels MobileNet SSD model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
           "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]

# Set colors for each detected object
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Open a connection to the ESP32 stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Connected to video stream. Press 'q' to quit.")

# Process the video stream frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame.")
        break

    # Resize frame and prepare the input blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Run forward pass to get detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

            # Draw label with black text
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow("ESP32 Camera Stream with Object Detection", frame)

    # Press 'q' to quit the video stream display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
"""

"""
#objects with feed
import cv2
import numpy as np

# Load the pre-trained model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# List of class labels MobileNet SSD model can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
           "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]


# Start video capture
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame and prepare the input blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    
    # Run forward pass to get detections
    detections = net.forward()
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0,0,0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
    
    # Display the output frame
    cv2.imshow("Live Object Detection", frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
"""


"""
#face with feed

import cv2

# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32

# Load a pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the ESP32 camera stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Connected to video stream. Press 'q' to quit.")

# Display the video stream frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame.")
        break
    
    # Convert frame to grayscale (necessary for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the frame with rectangles
    cv2.imshow("ESP32 Camera Stream with Face Detection", frame)

    # Press 'q' to quit the video stream display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
"""

"""
#Feed on cv2

import cv2


# Replace with your ESP32-CAM's IP address
url = "http://10.230.150.104:81/stream"  # Stream URL from the ESP32


# Open a connection to the stream
cap = cv2.VideoCapture(url)


if not cap.isOpened():
   print("Error: Could not open video stream.")
else:
   print("Connected to video stream. Press 'q' to quit.")


# Display the video stream frame by frame
while True:
   ret, frame = cap.read()
   if not ret:
       print("Error: Could not retrieve frame.")
       break
  
   # Show the frame in a window
   cv2.imshow("ESP32 Camera Stream", frame)


   # Press 'q' to quit the video stream display
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
"""


"""
camera facial wihtout stream


import cv2

# Load a pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Convert frame to grayscale (necessary for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the frame with rectangles
    cv2.imshow("Webcam Feed", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
"""
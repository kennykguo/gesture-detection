"""
This script contains bounding box reference code
"""

import cv2  # Import the OpenCV library for computer vision tasks
import mediapipe as mp  # Import the MediaPipe library for hand tracking

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# MediaPipe drawing utilities for visualization
mp_drawing = mp.solutions.drawing_utils  

# Capture video from the default camera (usually the webcam)
video_capture = cv2.VideoCapture(0)

# Read an initial frame from the camera to get dimensions
_, initial_frame = video_capture.read()
frame_height, frame_width, _ = initial_frame.shape

# Dictionary to store landmark coordinates
landmarks_dict = {}

while True:

    # Continuously read frames from the camera
    _, current_frame = video_capture.read()

    # Wait for a key press for 1 ms
    key = cv2.waitKey(1)

    # ESC key pressed
    if key % 256 == 27:
        print("Exiting")
        break

    # SPACE key pressed
    elif key % 256 == 32:

        # Capture the current frame for analysis
        analysis_frame = current_frame  
        # Display the captured frame to the user
        # cv2.imshow("Captured Frame", analysis_frame)  
        
        # Convert the frame to RGB
        analysis_frame_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        
        # Get hand landmarks from the, if there are any
        # Note that the user can only do sign languages with one hand!!!
        analysis_result = hands.process(analysis_frame_rgb)
        hand_landmarks_analysis = analysis_result.multi_hand_landmarks
        
        # Configure hand landmark coordinates
        if hand_landmarks_analysis:
            # Hand landmarks detected
            for hand_landmarks in hand_landmarks_analysis:
                x_max, y_max = 0, 0
                x_min, y_min = frame_width, frame_height
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20

                # Draw a rectangle around the hand
                cv2.rectangle(analysis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Convert the frame to grayscale for further processing if needed
                analysis_frame_gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)

                # Crop the frame to the hand region
                cropped_hand_frame = analysis_frame_gray[y_min:y_max, x_min:x_max]

                # Resize the cropped frame to 28x28 pixels
                resized_hand_frame = cv2.resize(cropped_hand_frame, (28, 28))

            # Print message indicating hand landmarks detected
            print("Hand landmarks detected!")

        else:
            # No hand landmarks detected
            print("No hand landmarks detected.")

        # Display the captured frame with the bounding box
        cv2.imshow("Captured Frame", cropped_hand_frame)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------

    """
    This part of the code does live processing on the camera
    """
    # Convert the frame to RGB
    current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    # Configure hand landmark coordinates
    frame_processing_result = hands.process(current_frame_rgb)
    hand_landmarks = frame_processing_result.multi_hand_landmarks

    # Get hand landmarks, if there are any
    if hand_landmarks:
        for hand_landmarks in hand_landmarks:
            x_max, y_max = 0, 0
            x_min, y_min = frame_width, frame_height
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            # Draw a rectangle around the hand
            cv2.rectangle(current_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the LIVE frame
    cv2.imshow("Live Frame", current_frame)


# Release the video capture object
video_capture.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
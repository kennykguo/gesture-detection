import cv2  # Import the OpenCV library for computer vision tasks
import mediapipe as mp  # Import the MediaPipe library for hand tracking

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# MediaPipe drawing utilities for visualization
mp_drawing = mp.solutions.drawing_utils  

# Capture video from the default camera (usually the webcam)
video_capture = cv2.VideoCapture(0)

# Read a frame from the camera to get dimensions
_, initial_frame = video_capture.read()
frame_height, frame_width, _ = initial_frame.shape

while True:
    # Continuously read frames from the camera
    _, current_frame = video_capture.read()

    # Wait for a key press for 1 ms
    key = cv2.waitKey(1)

    if key % 256 == 27:
        # ESC key pressed
        print("Escape hit, closing...")
        break

    elif key % 256 == 32:
        # SPACE key pressed
        analysis_frame = current_frame  # Capture the current frame for analysis
        cv2.imshow("Captured Frame", analysis_frame)  # Display the captured frame
        
        # Convert the frame to RGB
        analysis_frame_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        # Process the frame for hand landmarks
        analysis_result = hands.process(analysis_frame_rgb)
        hand_landmarks_analysis = analysis_result.multi_hand_landmarks
        if hand_landmarks_analysis:
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

                # Convert the frame to grayscale
                analysis_frame_gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                # Crop the frame to the hand region
                cropped_hand_frame = analysis_frame_gray[y_min:y_max, x_min:x_max]
                # Resize the cropped frame to 28x28 pixels
                resized_hand_frame = cv2.resize(cropped_hand_frame, (28, 28))

    # Convert the frame to RGB
    current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    frame_processing_result = hands.process(current_frame_rgb)
    hand_landmarks = frame_processing_result.multi_hand_landmarks

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

    # Display the frame
    cv2.imshow("Live Frame", current_frame)

# Release the video capture object
video_capture.release()
# Close all OpenCV windows
cv2.destroyAllWindows()

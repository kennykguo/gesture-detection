"""
This script contains finger positions reference code
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

# Read a frame from the camera to get dimensions
_, initial_frame = video_capture.read()
frame_height, frame_width, _ = initial_frame.shape

# Dictionary to store landmark coordinates
landmarks_dict = {}

while True:
    # Continuously read frames from the camera
    _, current_frame = video_capture.read()

    # Wait for a key press for 1 ms
    key = cv2.waitKey(1)

    if key % 256 == 27:
        # ESC key pressed
        print("Escape hit, closing...")
        break

    # SPACE key pressed
    elif key % 256 == 32:

        # Capture the current frame for analysis
        analysis_frame = current_frame  
        # Display the captured frame to the user
        cv2.imshow("Captured Frame", analysis_frame)  
        
        # Convert the frame to RGB
        analysis_frame_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand landmarks
        analysis_result = hands.process(analysis_frame_rgb)
        hand_landmarks_analysis = analysis_result.multi_hand_landmarks
        
        # Hand landmarks detected
        if hand_landmarks_analysis:

            for hand_id, hand_landmarks in enumerate(hand_landmarks_analysis):

                landmark_positions = {}
                
                for idx, landmark in enumerate(hand_landmarks.landmark):

                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)

                    landmark_positions[f"Landmark_{idx}"] = (x, y)

                    # Draw landmark as a dot on the hand
                    cv2.circle(analysis_frame, (x, y), 5, (255, 0, 0), -1)
                
                # Save landmark positions to dictionary
                landmarks_dict[f"Hand_{hand_id}"] = landmark_positions

                # Print landmark positions

                print(f"Hand {hand_id} Landmark Positions:")

                for key, (x, y) in landmark_positions.items():
                    print(f"{key}: ({x}, {y})")

        else:
            # No hand landmarks detected
            print("No hand landmarks detected.")

        # Display the captured frame with landmarks
        cv2.imshow("Captured Frame", analysis_frame)

    # Process the frame for hand landmarks
    current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    frame_processing_result = hands.process(current_frame_rgb)
    hand_landmarks = frame_processing_result.multi_hand_landmarks

    if hand_landmarks:
        for hand_id, hand_landmarks in enumerate(hand_landmarks):
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                # Draw landmark as a dot on the hand in the live frame
                cv2.circle(current_frame, (x, y), 5, (255, 0, 0), -1)

    # Display the live frame with landmarks
    cv2.imshow("Live Frame", current_frame)

# Release the video capture object
video_capture.release()
# Close all OpenCV windows
cv2.destroyAllWindows()

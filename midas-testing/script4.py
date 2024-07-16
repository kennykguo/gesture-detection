import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from midas.midas_net import MidasNet

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load pretrained MiDaS model
midas_model = MidasNet(model_path='path_to_midas_model.pth')
midas_model.eval()

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
        
        # Convert the frame to RGB
        analysis_frame_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand landmarks
        analysis_result = hands.process(analysis_frame_rgb)
        hand_landmarks_analysis = analysis_result.multi_hand_landmarks
        
        # Check if hand landmarks are detected
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

                # Draw a rectangle around the hand
                cv2.rectangle(analysis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Convert the frame to grayscale for further processing if needed
                analysis_frame_gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)

                # Crop the frame to the hand region
                cropped_hand_frame = analysis_frame_gray[y_min:y_max, x_min:x_max]

                # Resize the cropped frame to 28x28 pixels
                resized_hand_frame = cv2.resize(cropped_hand_frame, (28, 28))

                # Store hand landmark positions in the dictionary
                landmark_positions = {}
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmark_positions[f"Landmark_{idx}"] = (int(landmark.x * frame_width), int(landmark.y * frame_height))
                landmarks_dict["Hand"] = landmark_positions

                # Print landmark positions
                print("Hand Landmark Positions:")
                for key, (x, y) in landmark_positions.items():
                    print(f"{key}: ({x}, {y})")

                # Example of depth estimation using MiDaS
                with torch.no_grad():
                    input_tensor = transforms.ToTensor()(analysis_frame_rgb).unsqueeze(0)
                    depth_map = midas_model(input_tensor)

                # Convert depth map to numpy array and normalize
                depth_map_np = depth_map.squeeze().cpu().numpy()
                depth_map_np = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min())

                # Visualize depth map (optional)
                cv2.imshow("Depth Map", depth_map_np)

            # Display the captured frame with bounding box
            cv2.imshow("Captured Frame", analysis_frame)

        else:
            # No hand landmarks detected
            print("No hand landmarks detected.")

    """
    This part of the code does live processing on the camera
    """

    # Convert the frame to RGB
    current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    frame_processing_result = hands.process(current_frame_rgb)
    hand_landmarks = frame_processing_result.multi_hand_landmarks

    # Check if hand landmarks are detected in live frame
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
            # Draw a rectangle around the hand in the live frame
            cv2.rectangle(current_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw landmarks as dots on the hand in the live frame
            for idx, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * frame_width), int(landmark.y * frame_height)
                cv2.circle(current_frame, (cx, cy), 5, (255, 0, 0), -1)

            # Example: Use the depth map for further processing
            with torch.no_grad():
                input_tensor = transforms.ToTensor()(current_frame_rgb).unsqueeze(0)
                depth_map_live = midas_model(input_tensor)

            # Process depth_map_live as needed

    # Display the live frame with bounding box and landmarks
    cv2.imshow("Live Frame", current_frame)

# Release the video capture object
video_capture.release()
# Close all OpenCV windows
cv2.destroyAllWindows()

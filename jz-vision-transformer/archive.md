# ### Debugging functions
# def extract_landmarks_from_video(video_file, target_size=(224, 224), max_frames=50):
#     cap = cv2.VideoCapture(video_file)
#     frames = []
#     landmarks = []
#     frame_count = 0
#     while cap.isOpened() and frame_count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, target_size)  # Resize frame
#         frames.append(frame)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 frame_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
#                 landmarks.append(frame_landmarks)
#         else:
#             landmarks.append([[0, 0, 0]] * 21)  # If no hand detected, append zero landmarks
#         frame_count += 1
#     cap.release()
#     return frames, landmarks

# # Function to normalize landmarks based on image dimensions
# def normalize_landmarks(landmarks, image_width, image_height):
#     normalized_landmarks = []
#     for frame_landmarks in landmarks:
#         normalized_frame_landmarks = [[lm[0] * image_width, lm[1] * image_height, lm[2]] for lm in frame_landmarks]
#         normalized_landmarks.append(normalized_frame_landmarks)
#     return normalized_landmarks

# # Function to display frames
# def visualize_frames(frames):
#     fig, axes = plt.subplots(5, 10, figsize=(20, 10))
#     fig.suptitle('Frames', fontsize=16)
    
#     for i, ax in enumerate(axes.flat):
#         if i >= len(frames):
#             break
#         ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(f'Frame {i+1}')
    
#     plt.tight_layout()
#     plt.show()

# # Function to display landmarks
# def visualize_landmarks(frames, landmarks):
#     image_height, image_width, _ = frames[0].shape
#     normalized_landmarks = normalize_landmarks(landmarks, image_width, image_height)
    
#     fig, axes = plt.subplots(5, 10, figsize=(20, 10))
#     fig.suptitle('Hand Landmarks for First 50 Frames', fontsize=16)
    
#     for i, ax in enumerate(axes.flat):
#         if i >= len(normalized_landmarks):
#             break
#         ax.scatter([lm[0] for lm in normalized_landmarks[i]], [lm[1] for lm in normalized_landmarks[i]], c='b', marker='o')
#         for j in range(len(normalized_landmarks[i])):
#             ax.text(normalized_landmarks[i][j][0], normalized_landmarks[i][j][1], str(j), fontsize=9)
#         ax.set_xlim(0, image_width)
#         ax.set_ylim(0, image_height)
#         ax.invert_yaxis()
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(f'Frame {i+1}')
    
#     plt.tight_layout()
#     plt.show()

# # Path to the video file
# video_file_path = '../data/ZJ-videos/z/100.avi'  # Update this path

# # Extract and visualize the first 50 frames and landmarks
# frames, landmarks = extract_landmarks_from_video(video_file_path)

# # Visualize frames
# visualize_frames(frames)

# # Visualize landmarks
# visualize_landmarks(frames, landmarks)

# # MediaPipe coordinates are normalized between 0 and 1, based on the grid size
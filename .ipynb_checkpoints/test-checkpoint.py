import cv2

# Try different indices to find the correct one
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"Camera not found at index {i}")

cv2.destroyAllWindows()

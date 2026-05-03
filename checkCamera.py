import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        ret, frame = cap.read()
        if ret:
            print(f"  -> frame read OK from index {i}")
        else:
            print(f"  -> opened but could not read frame")
        cap.release()
    else:
        print(f"No camera at index {i}")
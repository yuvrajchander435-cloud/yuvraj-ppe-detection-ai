import cv2

video_path = "C:/PPE Detection/runs/detect/predict23/0.avi"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("PPE Detection Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
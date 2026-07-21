import cv2

cap = cv2.VideoCapture(0)
print("Appuyez sur ESPACE pour capturer la photo, ou 'q' pour quitter sans capturer")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cv2.imshow("Capture - Appuyez sur ESPACE", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.imwrite("data/employes_autorises/mama.jpg", frame)
        print("Photo enregistrée : data/employes_autorises/aya.jpg")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
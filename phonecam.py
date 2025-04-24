import cv2

# Use your actual phone's IP address
video = cv2.VideoCapture('http://172.16.112.247:8080/video')

while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        print("Failed to retrieve frame.")
        break

    cv2.imshow("Phone Camera Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

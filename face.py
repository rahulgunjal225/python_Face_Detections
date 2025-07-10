import cv2

# Load the Haar cascade classifier
a = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
b = cv2.VideoCapture(0)

if not b.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    c_rec, d_image = b.read()
    if not c_rec:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to grayscale
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    f = a.detectMultiScale(e, scaleFactor=1.3, minNeighbors=6)

    # Draw rectangles around detected faces
    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)

    # Show the video feed with detected faces
    cv2.imshow('Face Detection', d_image)

    # Exit if the "Esc" key (27) is pressed
    h = cv2.waitKey(40) & 0xff
    if h == 27:
        break

# Release the webcam and close the window
b.release()
cv2.destroyAllWindows()

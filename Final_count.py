import numpy as np
import cv2

# Create a structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Start video capture (0 is for default camera, try other indices if needed)
cap = cv2.VideoCapture(0)

# Set resolution (optional, you can adjust these values)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

# Initialize counters
cin = 0
cout = 0
pre = 0
prei = 800

# Set video output codec and writer (use 'mp4v' for MP4 format)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
out = cv2.VideoWriter('Video_output.mp4', fourcc, 2, (640, 480), True)

# Define helper functions for vehicle crossing detection
def iscrossin(prei, cur, width):
    if prei < width / 2 and cur > width / 2:
        return 1
    else:
        return 0

def iscrossout(pre, cur, width):
    if pre > width / 2 and cur < width / 2:
        return 1
    else:
        return 0

# Video loop to process frames
while True:
    # Read the image from the camera
    ret, img = cap.read()

    if ret:
        # Crop and preprocess image
        img = img[80:, 100:]
        height, width, channels = img.shape
        img = cv2.medianBlur(img, 5)

        # Apply morphological operations (dilation and erosion)
        dilation = cv2.dilate(img, kernel, iterations=4)
        img = cv2.erode(dilation, kernel, iterations=6)

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for the mask
        lower = np.array([0, 0, 27])
        upper = np.array([200, 255, 255])

        # Create the mask
        mask = cv2.inRange(hsv, lower, upper)

        # Threshold the mask and find contours
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a red line in the center of the image
        img = cv2.line(img, (width // 2, 0), (width // 2, 600), (0, 0, 255), 4)

        # Iterate over contours and process objects
        for i in range(1, len(contours)):
            area = cv2.contourArea(contours[i])
            if 10000 < area < 25000:  # Filter contours based on area
                cnt = i
                M = cv2.moments(contours[cnt])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contours[cnt])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Check if the object crosses the center line
                cur = cx
                if iscrossin(prei, cur, width):
                    if abs(prei - cur) < 60:
                        cout += 1
                elif iscrossout(pre, cur, width):
                    if abs(pre - cur) < 60:
                        cin += 1

                # Update previous positions
                pre = cur
                prei = cur

        # Display the count of vehicles
        IN = "IN: " + str(cin)
        OUT = "OUT: " + str(cout)
        cv2.putText(img, IN, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, OUT, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the image in a window
        cv2.imshow('Image', img)

        # Write the processed frame to the output video file
        out.write(img)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()

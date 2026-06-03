import cv2 as cv
import numpy as np
# from matplotlib import plyplot as plt

camera= cv.VideoCapture(0)
frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv.CAP_PROP_FPS) 
# out = cv.VideoWriter('/Users/avni/Downloads/flip16t2.mp4', cv.VideoWriter_fourcc(*'mp4v'), 16.0, (frame_width, frame_height))
while True:
        ret, frame = camera.read() # Read a frame

        if not ret:
            print("Error: Failed to read frame.")
            break
        fliped= cv.flip(frame, 1)
        img= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img= cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        gaus= cv.GaussianBlur(img, (15, 15), 0)  
        t_lower = 40  
        t_upper = 90 
        canny = cv.Canny(gaus, t_lower, t_upper)
        top, bottom, left, right = 20, 20, 20, 20
        frame_color = [0, 255, 0]
        # cv.imshow('Camera Feed', canny)
        cv.imshow('og', frame) # Display the frame
        Mask = cv.bitwise_not(canny)
        # cv.imshow('invert', Mask)
        # out.write(fliped)
        #fourcc = cv2.VideoWriter_fourcc(*'XVID') # Example for .avi
        # Or for .mp4:
        #fourcc = cv.VideoWriter_fourcc(*'MJPEG')
        frame1= cv.copyMakeBorder(Mask, top, bottom, left, right, borderType=cv.BORDER_CONSTANT, value=frame_color )
        frame2= cv.copyMakeBorder(canny, top, bottom, left, right, borderType=cv.BORDER_CONSTANT, value=frame_color)
        vertical= np.concatenate(( frame1, frame2), axis=0)
        cv.imshow('together', vertical)
        # Press 'q' to exit the loop
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
camera.release()
# out.release()
cv.destroyAllWindows()
import cv2

cam = cv2.VideoCapture('double_pendulum.mp4')
last_frame = cam.get(7)
fps = cam.get(5)
mydelay = int( (1 / fps) * 1000)
print fps
while True:
    ret, img = cam.read()
    if ret == True:
        cv2.imshow('detection', img)
        if cam.get(1) == last_frame:
            cam.set(1,0)

    if cv2.waitKey(mydelay) & 0xFF == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break
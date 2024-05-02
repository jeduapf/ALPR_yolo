# Image imports
import cv2

# Basic imports
import time
import numpy as np


class VideoShower():
    def __init__(self, frame=None, win_name="Video"):
        """
        Class to show frames in a dedicated thread.
        Args:
            frame (np.ndarray): (Initial) frame to display.
            win_name (str): Name of `cv2.imshow()` window.
        """
        self.frame = frame
        self.win_name = win_name
        self.stopped = False

    def start(self):
        threading.Thread(target=self.show, args=()).start()
        return self

    def show(self):
        """
        Method called within thread to show new frames.
        """
        while not self.stopped:
            # We can actually see an ~8% increase in FPS by only calling
            # cv2.imshow when a new frame is set with an if statement. Thus,
            # set `self.frame` to None after each call to `cv2.imshow()`.
            if self.frame is not None:
                cv2.imshow(self.win_name, self.frame)
                self.frame = None

            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        cv2.destroyWindow(self.win_name)
        self.stopped = True

class VideoBuffer():
    def __init__(self, cam, buffer_seconds = 2, buffer_fps = 10):
        """
        Class to read frames from a VideoCapture in a dedicated thread.
        Args:
            src: Video source.
                type: str
            buffer: Buffer length, total amout of frames to get before processing (delayed frames)
                type: int
        """
        # assert isinstance(src, str) and src.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')), f"Source ({source}) must be a IP camera with RTSP/RTMP/HTTP/HTTPS protocol !"
        assert isinstance(buffer_seconds, int) and buffer_seconds != 0, "Buffer seconds must be an integer different from zero !"
        assert isinstance(buffer_fps, int) and buffer_fps != 0, "Buffer FPS must be an integer different from zero !"

        self.cam_fps = cam.info()['FPS']
        assert self.cam_fps > buffer_fps and self.cam_fps%buffer_fps==0, "Buffer FPS can't be higher than camera FPS and must be divisible!"


        # Start capturing and get first frame to know it's shape
        self.cap = cv2.VideoCapture(cam.url())
        self.grabbed, self.frame = self.cap.read()
        
        # Set buffer size 
        self.img_shape = np.shape(self.frame)

        # ---------------------[                    frames,    height, width, channels]
        # self.buffer = np.empty([int(buffer_fps*buffer_seconds), shp[0], shp[1], shp[2]], dtype=np.uint8)
        # Too complicated using numpy? better use list
        self.buffer = []

        self.buffer_seconds = buffer_seconds
        self.buffer_fps = buffer_fps

    def start(self):
        threading.Thread(target=self.get, args=(), daemon = True).start()
        return self

    def get(self):
        """
        """
        # count = 0 # Could just be int(i/div) instead of count, but seems so costfull
        div = int(self.cam_fps/self.buffer_fps)
        
        for i in range(int(self.buffer_seconds*self.cam_fps)):
            self.grabbed, self.frame = self.cap.read()
            if (i % div == 0): # Get buffer_fps frames per second
                if self.grabbed:
                    # self.buffer[count,:,:,:] = self.frame
                    self.buffer.append([time.time(),self.frame])

                else: # Blank frame 
                    # self.buffer[count,:,:,:] = np.zeros(np.shape(self.buffer)[1:])
                    self.buffer.append([time.time(),np.zeros(self.img_shape)])
                # count +=1

        return self.buffer


    def info(self):
        print(f"Buffer length: {len(self.buffer)}")
        return len(self.buffer)

class IP_cam:
    # Para o Stream principal: rtsp://USUÁRIO:SENHA@IP:PORTA/cam/realmonitor?channel=1&subtype=0
    # Para o Stream extra: rtsp://USUÁRIO:SENHA@IP:PORTA/cam/realmonitor?channel=1&subtype=1
    # cam_https = 'https://admin:Pai39mae39@192.168.0.74:443/cam/realmonitor?channel=1&subtype=0' #[tcp @ 000001fef27a3b80] Connection to tcp://192.168.0.74:443 failed: Error number -138 occurred
    # cam_http = 'https://admin:Pai39mae39@192.168.0.74:80/cam/realmonitor?channel=1&subtype=0' #[tls @ 000001e6913f3480] Failed to read handshake response

    def __init__(self, user, password, IP, port):
        self.user = user
        self.password = password
        self.IP = IP
        self.port = port
        self.URL = f"rtsp://{self.user}:{self.password}@{self.IP}:{self.port}/cam/realmonitor?channel=1&subtype=0"
        self.cap = cv2.VideoCapture(self.URL)

        if self.cap is None or not self.cap.isOpened():
            hidden_url = self.URL.split('//')[0]+ '//*****:*****@' +self.URL.split('@')[-1]
            raise Exception(f'Unable to open video source: {hidden_url}')

    def __str__(self):
        return f"{self.IP} camera in port {self.port}"

    def one_frame(self):
        # Read the frame
        ret, frame = self.cap.read()

        if ret:
            return frame

    def stream(self):
        print('\n\t\tStarting video streaming...\n\t\tTo end press (ESC)\n')
        while True:
            # Read the frame
            ret, img = self.cap.read()

            if ret:
                # Display
                cv2.imshow('img', img)
                
                # Stop if escape key is pressed
                k = cv2.waitKey(30) & 0xff
                if k==27:
                    break

        # Release the VideoCapture object
        self.cap.release()

    def info(self):
        cap = self.cap
        video_info_dict = {
            "FPS" : cap.get(cv2.CAP_PROP_FPS), 
            "HEIGTH" : cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "WIDTH" : cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "MODE" : cap.get(cv2.CAP_PROP_MODE),
            "FORMAT" : cap.get(cv2.CAP_PROP_FORMAT), 
            "BUFFERSIZE" : cap.get(cv2.CAP_PROP_BUFFERSIZE), 
            "CHANNEL" : cap.get(cv2.CAP_PROP_CHANNEL), 
            "BITRATE" : cap.get(cv2.CAP_PROP_BITRATE),       
        }

        return video_info_dict

    def url(self):
        return self.URL

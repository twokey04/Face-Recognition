import customtkinter
import pathlib

# import DetectedFace class
from DetectFace import DetectedFace

import cv2

from PIL import Image, ImageTk

class GUI():
    # set theme 
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")

    # set main window
    root = customtkinter.CTk()
    # set size of the main window
    root.geometry("1200x800")
    # set name of main window
    root.title("Face Detection")

    # create a frame on the main window
    frame = customtkinter.CTkFrame(master=root)
    # set margins for the frame
    frame.pack(pady=20, padx=60, fill="both", expand=True)

    # create a label to display the camera
    cameraLabel = customtkinter.CTkLabel(master=frame, text="")
    cameraLabel.pack()
    
    indexLabel = customtkinter.CTkLabel(master=frame, text="Index: ")
    indexLabel.pack()
    
    # open default camera
    camera = cv2.VideoCapture(0)
    
    def ShowFrames(self):
        # get the path for Haar cascade
        cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
        # load OpenCV detector ( Haar classifier is slow, but accurate )
        classifier = cv2.CascadeClassifier(str(cascade_path))
        # check if there is a camera
        if self.camera.isOpened():
            if self.frame.winfo_exists():
                # create object to detect faces
                detectFace = DetectedFace(cascade_path, classifier, self.camera)
                # get last frame from camera
                cv2LastFrame, numberOfFaces = detectFace.DetectFaceInLastFrame()
                # convert frame to RGB
                cv2LastFrame = cv2.cvtColor(cv2LastFrame, cv2.COLOR_BGR2RGB)
                # convert frame to PIL.Image
                img = Image.fromarray(cv2LastFrame)
                # convert frame to Tkinter image format
                imageTk = ImageTk.PhotoImage(image=img)            
                # display last frame
                self.cameraLabel.configure(image=imageTk)
                # update index label to show ID of face
                self.indexLabel.configure(text=f"Index: {numberOfFaces}")
                # repeat after an interval of ms to capture continuously
                self.cameraLabel.after(10, self.ShowFrames)               
            
        else:
            # close capturing device
            self.camera.release()
            # close all windows
            cv2.destroyAllWindows()
            # if there is no default camera an exception is raised
            raise Exception("[Camera must be open]")
        
    def Run(self):
        while True:
            self.ShowFrames()
            self.root.mainloop()

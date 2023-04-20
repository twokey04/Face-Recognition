import cv2
import numpy as np
import os
import glob
import time
import face_recognition

# declare model for face detection
prototxt_path = r'./Models/deploy_prototxt.txt'
model_path = r'./Models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

camera = cv2.VideoCapture(0)

# create an empty dictionary to store the IDs and coordinates of tracked objects
object_dict = {}

# load the face images and their encodings
face_images = {}
face_encodings = {}
face_dir = "./Faces/"
for img_file in glob.glob(os.path.join(face_dir, "*.jpg")):
    img = face_recognition.load_image_file(img_file)
    face_encodings_list = face_recognition.face_encodings(img, model='small')
    if len(face_encodings_list) > 0:
        encoding = face_encodings_list[0]
        name = os.path.splitext(os.path.basename(img_file))[0]
        face_images[name] = img
        face_encodings[name] = encoding
    else:
        print(f"No faces found in {img_file}")

name = 'Unknown'

prev_frame_time = 0
new_frame_time = 0

while True:
    check, frame = camera.read()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)

    output = np.squeeze(model.forward())

    font_scale = 0.7
    
    # create a list of the IDs of the objects that are currently being tracked
    object_ids = list(object_dict.keys())
    
    for i in range(0, output.shape[0]):
        # get the confidence
        confidence = output[i, 2]
        # if confidence is above 60% draw surrounding box
        if confidence > 0.6:
            # get the surrounding box cordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w, h, w, h])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int64)
            # draw the rectangle surrounding the face
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(0, 255, 0), thickness=2)
            
            # find the center of the bounding box
            center_x = int((start_x + end_x) / 2)
            center_y = int((start_y + end_y) / 2)
            
            # check if any of the tracked objects are close to the current object
            object_id = None
            for id, coord in object_dict.items():
                dist = np.sqrt((center_x - coord[0]) ** 2 + (center_y - coord[1]) ** 2)
                # if the current object is close to a tracked object use the same ID
                if dist < 50:
                    object_id = id
                    break
            
            # otherwise assign a new ID to the current object
            if object_id is None:
                object_id = len(object_dict) + 1
                object_dict[object_id] = (center_x, center_y)
            
            # recognize the face
            face_img = frame[start_y:end_y, start_x:end_x, :]
            unknown_face_encodings = face_recognition.face_encodings(face_img, model='small')
            if len(unknown_face_encodings) > 0:
                face_encoding = unknown_face_encodings[0]
                matches = face_recognition.compare_faces(list(face_encodings.values()), face_encoding, tolerance=0.6)
                if True in matches:
                    match_index = matches.index(True)
                    name = list(face_encodings.keys())[match_index].split('_')
                    name = name[0].capitalize() + ' ' + name[1].capitalize()
            
            # compute current FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            # draw the ID and Name above the bounding box
            cv2.putText(frame, f"ID: {object_id} Name: {name}", (start_x, start_y - 10), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)
            # draw the FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (0, 25), cv2.FONT_HERSHEY_COMPLEX, font_scale + 0.2, (0, 0, 255), 2)
            
            # update the coordinates of the tracked object
            object_dict[object_id] = (center_x, center_y)
            
    # remove the IDs of the objects that are no longer being tracked
    object_ids_to_delete = []
    for id in object_ids:
        if id not in object_dict:
            object_ids_to_delete.append(id)
    for id in object_ids_to_delete:
        del object_dict[id]
        
    # show the image
    cv2.imshow("image", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# after the loop release the cap object
camera.release()
# destroy all the windows
cv2.destroyAllWindows()
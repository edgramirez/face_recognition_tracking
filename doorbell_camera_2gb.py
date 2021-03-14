import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
import biblioteca as biblio

def main_loop():
    video_capture = cv2.VideoCapture('/home/edgar/Downloads/La_cronica_triunfo_AMLO.mp4')

    # Track how long since we last saved a copy of our known faces to disk as a backup.
    number_of_faces_since_save = 0

    # load data from binary db 
    default_data_file = '/tmp/known_faces.dat'
    total_visitors, known_face_encodings, known_face_metadata = biblio.read_pickle(default_data_file, False)

    frame_counter = 0
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_counter += 1

        # Process image every other frame to speed up
        if frame_counter % 3 == 0:
            continue

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                # See if this face is in our list of known faces.
                metadata = biblio.lookup_known_face(face_encoding, known_face_encodings, known_face_metadata)
    
                # If we found the face, label the face with some useful information.
                if metadata:
                    time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                    face_label = f"{metadata['name']} {int(time_at_door.total_seconds())}s"
                # If this is a brand new face, add it to our list of known faces
                else:
                    face_label = "New visitor" + str(total_visitors) + '!!'
                    total_visitors += 1
    
                    # Grab the image of the the face from the current frame of video
                    top, right, bottom, left = face_location
                    face_image = small_frame[top:bottom, left:right]
                    face_image = cv2.resize(face_image, (150, 150))
    
                    # Add the new face to our known faces metadata
                    known_face_metadata = biblio.register_new_face(known_face_metadata, face_image, 'visitor' + str(total_visitors))

                    # Add the face encoding to the list of known faces
                    known_face_encodings.append(face_encoding)

                face_labels.append(face_label)
    
            # Draw a box around each face and label each face
            biblio.draw_box_around_face(face_locations, face_labels, frame)
    
            # Display recent visitor images
            biblio.display_recent_visitors_face(known_face_metadata, frame)

        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            biblio.write_to_pickle(known_face_encodings, known_face_metadata, default_data_file)
            break

        # We need to save our known faces back to disk every so often in case something crashes.
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            biblio.write_to_pickle(known_face_encodings, known_face_metadata, default_data_file)
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()

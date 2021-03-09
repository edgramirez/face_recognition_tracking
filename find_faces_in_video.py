import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
import biblioteca as biblio

# Set this depending on your camera type:
# - True = Raspberry Pi 2.x camera module
# - False = USB webcam or other USB video input (like an HDMI capture device)
#USING_RPI_CAMERA_MODULE = True
USING_RPI_CAMERA_MODULE = False

# Our list of known face encodings and a matching list of metadata about each face.
encodings_of_known_faces = []
names_of_know_faces = []
known_face_metadata = []
pickle_file = 'train.pkl'


#video_file = '/home/edgar/Downloads/La_cronica_del_triunfo_de_AMLO_el_1_de_julio_2018.mp4'
video_file = '/home/edgar/Downloads/Halloween_obama.mp4'
video_file = '/home/edgar/Downloads/The_Final_Minutes_of_President_Obamas_Farewell_Address_Yes_we_can.mp4'
video_file = '/home/edgar/Downloads/Obama.mp4'
video_file = 'videos/MV12FaceRecognition8.mp4'
video_file = '/home/edgar/Downloads/Prince_Harry_and_Michelle_Obama_surprise_students_in_Chicago.mp4'
video_file = 'videos/edgar.mp4'
video_file = 'videos/deysi.mp4'
#video_file = 'Love_and_Happiness_An_Obama_Celebration.mp4'
#video_file = 'Prince_Harry_and_Michelle_Obama_surprise_students_in_Chicago.mp4'


def lookup_known_face(face_encoding):
    """
    See if this is a face we already have in our face list
    """
    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(encodings_of_known_faces) == 0:
        return None

    # compare to get a list of matches only to see if it is interesing to check
    matches = face_recognition.compare_faces(encodings_of_known_faces, face_encoding)

    if True in matches:
        # Calculate the face distance between the unknown face and every face on in our known face list
        # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
        # the more similar that face was to the unknown face.
        face_distances = face_recognition.face_distance(encodings_of_known_faces, face_encoding)

        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)

        # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
        # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
        # of the same person always were less than 0.6 away from each other.
        # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
        # people will come up to the door at the same time.
        if face_distances[best_match_index] < 0.65:
            # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
            print(best_match_index,names_of_know_faces[best_match_index])
            print(face_distances)
            return names_of_know_faces[best_match_index]

    return None

def main_loop():
    # Get access to the webcam. The method is different depending on if you are using a Raspberry Pi camera or USB input.
    # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
    # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
    video_capture = cv2.VideoCapture(video_file)
    #video_capture = cv2.VideoCapture(0)

    # Track how long since we last saved a copy of our known faces to disk as a backup.
    number_of_faces_since_save = 0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if not ret:
            break

        # Resize frame of video to 1/4 size for faster0face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            print('entro .....')

            #number_of_recent_visitors = 0
            for face_location, face_encoding in zip(face_locations, face_encodings):
                # See if this face is in our list of known faces.
                top, right, bottom, left = face_location
                metadata = lookup_known_face(face_encoding)
    
                # If we found the face, label the face with some useful information.
                if metadata:
                    #time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                    face_label = metadata
    
                # If this is a brand new face, add it to our list of known faces
                else:
                    face_label = "Desconocido"
                    # Grab the image of the the face from the current frame of video
                    #face_image = small_frame[top:bottom, left:right]
                    #face_image = cv2.resize(face_image, (150, 150))
    
                # Draw a box around the face
                top_x_4 = top * 4
                right_x_4 = right * 4
                bottom_x_4 = bottom * 4
                left_x_4 = left * 4
                cv2.rectangle(frame, (left_x_4, top_x_4), (right_x_4, bottom_x_4), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left_x_4, bottom_x_4 - 35), (right_x_4, bottom_x_4), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, face_label, (left_x_4 + 6, bottom_x_4 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # We need to save our known faces back to disk every so often in case something crashes.
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    encodings_and_names = biblio.read_pickle('train.pkl')
    names_of_know_faces = encodings_and_names[0]
    encodings_of_known_faces = encodings_and_names[1]
    #mi_main()
    main_loop()
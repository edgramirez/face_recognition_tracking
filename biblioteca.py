import pickle
import os
import cv2
from os import walk
import face_recognition
import numpy as np
from datetime import datetime, timedelta


#Names = []
#Encodings = []

#known_face_metadata = []
font = cv2.FONT_HERSHEY_SIMPLEX
#print(cv2.__version__)


def read_images_in_dir(path_to_read):
    dir_name, subdir_name, file_names = next(walk(path_to_read))
    images = [item for item in file_names if '.jpeg' in item[-5:] or '.jpg' in item[-4:] or 'png' in item[-4:] ]
    return images, dir_name


def read_pickle(pickle_file, exception=True):
    try:
        with open(pickle_file, 'rb') as f:
            encodings, known_face_metadata = pickle.load(f)
            return encodings, known_face_metadata
    except OSError as e:
        if exception:
            log_error("Unable to open pickle_file: {}, original exception {}".format(pickle_file, str(e)))
        else:
            return False


def write_to_pickle(encodings_list, known_face_metadata, output_file):
    with open(output_file,'wb') as f:
        face_data = [encodings_list, known_face_metadata]
        pickle.dump(face_data, f)


def log_error(msg, _quit=True):
    print("-- PARAMETER ERROR --\n"*5)
    print(" %s \n" % msg)
    print("-- PARAMETER ERROR --\n"*5)
    if _quit:
        quit()
    else:
        return False


def compare_pickle_against_video(pickle_file, frame):
    known_face_encodings, known_face_metadata = read_pickle(pickle_file)

    test_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # try to get the location of the face if there is one
    face_locations = face_recognition.face_locations(test_image)

    # if got a face, loads the image, else ignores it
    if face_locations:
        encoding_of_faces = face_recognition.face_encodings(test_image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, encoding_of_faces):
            face_title = 'desconocido'

            # compare to get a list of matches only to see if it is interesing to check
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if True in matches:
                # Calculate the face distance between the unknown face and every face on in our known face list
                # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
                # the more similar that face was to the unknown face.
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
                best_match_index = np.argmin(face_distances)

                # THIS IS NOT CORRECT BECAUSE NOT NECESARY THAT THE FIRT MATCH IS THE BEST ONE
                #first_match_index = matches.index(True)
                #print("first_match_index", first_match_index, '..........................................')
                #face_title = Names[first_match_index]
                face_title = Names[best_match_index]

            cv2.rectangle(test_image, (left, top),(right, bottom),(0, 0, 255), 2)
            cv2.putText(test_image, face_title, (left, top-6), font, .75, (180, 51, 225), 2)

        cv2.imshow('Imagen', test_image)
        cv2.moveWindow('Imagen', 0 ,0)

        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame)


def compare_pickle_against_unknown_images(pickle_file, image_dir):
    known_face_encodings, known_face_metadata = read_pickle(pickle_file)

    files, root = read_images_in_dir(image_dir)
    for file_name in files:
        file_path = os.path.join(root, file_name)
        test_image = face_recognition.load_image_file(file_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

        # try to get the location of the face if there is one
        face_locations = face_recognition.face_locations(test_image)

        # if got a face, loads the image, else ignores it
        if face_locations:
            encoding_of_faces = face_recognition.face_encodings(test_image, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, encoding_of_faces):
                face_title = 'desconocido'

                # compare to get a list of matches only to see if it is interesing to check
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                if True in matches:
                    # Calculate the face distance between the unknown face and every face on in our known face list
                    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
                    # the more similar that face was to the unknown face.
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
                    best_match_index = np.argmin(face_distances)

                    # THIS IS NOT CORRECT BECAUSE NOT NECESARY THAT THE FIRT MATCH IS THE BEST ONE
                    #first_match_index = matches.index(True)
                    #print("first_match_index", first_match_index, '..........................................')
                    #face_title = Names[first_match_index]
                    #face_title = Names[best_match_index]
                    face_title = known_face_metadata[best_match_index]['name']

                cv2.rectangle(test_image, (left, top),(right, bottom),(0, 0, 255), 2)
                cv2.putText(test_image, face_title, (left, top-6), font, .75, (180, 51, 225), 2)

            cv2.imshow('Imagen', test_image)
            cv2.moveWindow('Imagen', 0 ,0)

            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
        else:
            print("Image to search does not contains faces")
            print(file_path)


def encode_known_faces(known_faces_path, output_file):
    files, root = read_images_in_dir(known_faces_path)

    names = []
    known_face_encodings = []
    known_face_metadata = []

    for file_name in files:
        path = root + '/' + file_name
        name = os.path.splitext(file_name)[0]

        # load the image into face_recognition library
        face_obj = face_recognition.load_image_file(path)

        # try to get the location of the face if there is one
        face_location = face_recognition.face_locations(face_obj)

        # if got a face, loads the image, else ignores it
        if face_location:
            names.append(name)
            encoding = face_recognition.face_encodings(face_obj)[0]

            known_face_encodings.append(encoding)

            # Grab the image of the the face from the current frame of video
            top, right, bottom, left = face_location[0]
            face_image = face_obj[top:bottom, left:right]
            face_image = cv2.resize(face_image, (150, 150))

            date = datetime.now(),
            known_face_metadata.append({
                "first_seen": date,
                "first_seen_this_interaction": date,
                "last_seen": date,
                "seen_count": 1,
                "seen_frames": 1,
                "name": name,
                "face_image": face_image,
            })

    if names:
        print(names)
        write_to_pickle(known_face_encodings, known_face_metadata, output_file)
    else:
        print('Ningun archivo de imagen contine rostros')


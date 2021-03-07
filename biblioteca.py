import pickle
import os
import cv2
from os import walk
import face_recognition
import numpy as np


Encodings = []
Names = []
font = cv2.FONT_HERSHEY_SIMPLEX
#print(cv2.__version__)


def read_images_in_dir(path_to_read):
    dir_name, subdir_name, file_names = next(walk(path_to_read))
    images = [item for item in file_names if '.jpeg' in item[-5:] or '.jpg' in item[-4:] or 'png' in item[-4:] ]
    return images, dir_name

def read_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        Names = pickle.load(f)
        Encodings = pickle.load(f)
        return (Names, Encodings)

def write_to_pickle(Names, Encodings):
    with open('train.pkl','wb') as f:
        pickle.dump(Names, f)
        pickle.dump(Encodings, f)

def compare_pickle_against_unknown(pickle_file, image_dir):
    names_encodings = read_pickle(pickle_file)
    Names = names_encodings[0]
    Encodings = names_encodings[1]

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
                matches = face_recognition.compare_faces(Encodings, face_encoding)

                if True in matches:
                    # Calculate the face distance between the unknown face and every face on in our known face list
                    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
                    # the more similar that face was to the unknown face.
                    face_distances = face_recognition.face_distance(Encodings, face_encoding)

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

            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
        else:
            print("Image to search does not contains faces")
            print(file_path)

def encode_known_faces(known_faces_path):
    files, root = read_images_in_dir(known_faces_path)

    for file_name in files:
        path = root + '/' + file_name
        name = os.path.splitext(file_name)[0]

        # load the image into face_recognition library
        person = face_recognition.load_image_file(path)

        # try to get the location of the face if there is one
        face_locations = face_recognition.face_locations(person)

        # if got a face, loads the image, else ignores it
        if face_locations:
            encoding = face_recognition.face_encodings(person)[0]
            Encodings.append(encoding)
            Names.append(name)

    if Names:
        print(Names)
        write_to_pickle(Names, Encodings)
    else:
        print('Ningun archivo de imagen contine rostros')


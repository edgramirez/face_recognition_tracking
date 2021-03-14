import pickle
import os
import cv2
from os import walk
import face_recognition
import numpy as np
from datetime import datetime, timedelta


font = cv2.FONT_HERSHEY_SIMPLEX

def file_exists(file_name):
    try:
        with open(file_name) as f:
            return file_name
    except OSError as e:
        return False


def read_images_in_dir(path_to_read):
    dir_name, subdir_name, file_names = next(walk(path_to_read))
    images = [item for item in file_names if '.jpeg' in item[-5:] or '.jpg' in item[-4:] or 'png' in item[-4:] ]
    return images, dir_name


def read_pickle(pickle_file, exception=True):
    try:
        with open(pickle_file, 'rb') as f:
            known_face_encodings, known_face_metadata = pickle.load(f)
            print("Total Known faces loaded from disk = {}".format(len(known_face_encodings)))
            return len(known_face_metadata), known_face_encodings, known_face_metadata
    except OSError as e:
        if exception:
            log_error("Unable to open pickle_file: {}, original exception {}".format(pickle_file, str(e)))
        else:
            return 0, [], []


def register_new_face(known_face_metadata, face_image, name):
    """
    Add a new person to our list of known faces
    """
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
    today_now = datetime.now()
    known_face_metadata.append({
        "first_seen": today_now,
        "first_seen_this_interaction": today_now,
        "last_seen": today_now,
        "seen_count": 1,
        "seen_frames": 1,
        "name": name,
        "face_image": face_image,
    })

    return known_face_metadata


def write_to_pickle(known_face_encodings, known_face_metadata, data_file, new_file = True):
    if new_file and file_exists(data_file):
        os.remove(data_file)
        if file_exists(data_file):
            raise Exception('unable to delete file: %s' % file_name)

        with open(data_file,'wb') as f:
            face_data = [known_face_encodings, known_face_metadata]
            pickle.dump(face_data, f)
            print("Known faces backed up to disk.")
    else:
        with open(data_file,'ab') as f:
            face_data = [known_face_encodings, known_face_metadata]
            pickle.dump(face_data, f)


def log_error(msg, _quit=True):
    print("-- PARAMETER ERROR --\n"*5)
    print(" %s \n" % msg)
    print("-- PARAMETER ERROR --\n"*5)
    if _quit:
        quit()
    else:
        return False


def draw_box_around_face(face_locations, face_labels, image):
    # Draw a box around each face and label each face
    for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


def display_recent_visitors_face(known_face_metadata, frame):
    number_of_recent_visitors = 0
    for metadata in known_face_metadata:
        # If we have seen this person in the last minute, draw their image
        if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 5:
            # Draw the known face image
            x_position = number_of_recent_visitors * 150
            frame[30:180, x_position:x_position + 150] = metadata["face_image"]
            number_of_recent_visitors += 1

            # Label the image with how many times they have visited
            visits = metadata['seen_count']
            visit_label = f"{visits} visits"
            if visits == 1:
                visit_label = "First visit"
            cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)


def lookup_known_face(face_encoding, known_face_encodings, known_face_metadata):
    """
    See if this is a face we already have in our face list
    """
    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0:
        return None

    # Only check if there is a match
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    if True in matches:
        # If there is a match, then get the best distances only on the index with "True" to ignore the process on those that are False
        indexes = [ index for index, item in enumerate(matches) if item]
        only_true_known_face_encodings = [ known_face_encodings[ind] for ind in indexes ]

        face_distances = face_recognition.face_distance(only_true_known_face_encodings, face_encoding)
        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.65:
            # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
            metadata = [ known_face_metadata[ind] for ind in indexes ]
            metadata = metadata[best_match_index]

            # Update the metadata for the face so we can keep track of how recently we have seen this face.
            metadata["last_seen"] = datetime.now()
            metadata["seen_frames"] += 1

            if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
                metadata["first_seen_this_interaction"] = datetime.now()
                metadata["seen_count"] += 1

            return metadata

    return None


def compare_pickle_against_unknown_images(pickle_file, image_dir):
    total_known_faces, known_face_encodings, known_face_metadata = read_pickle(pickle_file)

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
                metadata = lookup_known_face(face_encoding, known_face_encodings, known_face_metadata)
                if metadata:
                    face_title = metadata['name']

                cv2.rectangle(test_image, (left, top),(right, bottom),(0, 0, 255), 2)
                cv2.putText(test_image, face_title, (left, top-6), font, .75, (180, 51, 225), 2)

            cv2.imshow('Imagen', test_image)
            cv2.moveWindow('Imagen', 0 ,0)

            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
        else:
            print("Image to search does not contains faces")
            print(file_path)


def encode_known_faces(known_faces_path, output_file, new_file = True):
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

            new_known_face_metadata = register_new_face(known_face_metadata, face_image, name)

    if names:
        print(names)
        write_to_pickle(known_face_encodings, new_known_face_metadata, output_file, new_file)
    else:
        print('Ningun archivo de imagen contine rostros')


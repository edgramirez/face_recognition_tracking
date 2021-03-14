#!/usr/bin/python3
import sys
import biblioteca as biblio 

param_length = len(sys.argv)

msg = 'Usage: ' + sys.argv[0] + ' [\n  newdb /PATH/TO/IMAGE/FILES output /PATH/TO/OUTPUT_FILE \n  appenTo /PATH/TO/EXISTING_DB \n  find_images KNOWN_INPUT_DATA_FILE | read_video | find_video ]'

if param_length < 1:
    biblio.log_error(msg)

if sys.argv[1] == 'newDb':
    if param_length == 2:
        known_faces = 'images/load'
        pickle_file = '/tmp/train.pkl'
    elif param_length == 4 and sys.argv[3] == 'output':
        known_faces = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        biblio.log_error(msg)

    biblio.encode_known_faces(known_faces, pickle_file)
elif sys.argv[1] == 'appenTo':
    if param_length == 2:
        known_faces = 'images/load'
        pickle_file = '/tmp/train.pkl'
    elif param_length == 4 and sys.argv[3] == 'output':
        known_faces = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        biblio.log_error(msg)

    biblio.encode_known_faces(known_faces, pickle_file, False)
elif sys.argv[1] == 'findImg':
    if param_length == 2:
        image_dir = 'images/find'
        pickle_file = '/tmp/train.pkl'
    elif param_length == 5 and sys.argv[3] == 'input':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        biblio.log_error(msg)

    biblio.compare_pickle_against_unknown_images(pickle_file, image_dir)
elif sys.argv[1] == 'readVideo':
    if param_length == 2:
        video_input = '/tmp/video/test_video.mp4'
        data_file = '/tmp/known_faces.dat'
    elif param_length == 5 and sys.argv[3] == 'input':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        biblio.log_error(msg)

    biblio.read_video(video_input, data_file)
else:
    biblio.log_error(msg)

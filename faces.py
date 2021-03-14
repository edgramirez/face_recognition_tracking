#!/usr/bin/python3
import sys
import common as com

param_length = len(sys.argv)

msg = 'Usage: ' + sys.argv[0] + ' newDb | appendTo | findImg | readVideo '

if param_length < 2:
    com.log_error(msg)

if sys.argv[1] == 'newDb':
    if param_length == 2:
        known_faces = 'images/load'
        pickle_file = '/tmp/train.pkl'
    elif param_length == 4 and sys.argv[3] == 'output':
        known_faces = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import biblioteca as biblio 
    biblio.encode_known_face(known_faces, pickle_file)
elif sys.argv[1] == 'appendTo':
    if param_length == 2:
        known_faces = 'images/load'
        pickle_file = '/tmp/train.pkl'
    elif param_length == 4 and sys.argv[3] == 'output':
        known_faces = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import biblioteca as biblio 
    biblio.encode_known_face(known_faces, pickle_file, False)
elif sys.argv[1] == 'findImg':
    if param_length == 2:
        image_dir = 'images/find'
        pickle_file = '/tmp/train.pkl'
    elif param_length == 5 and sys.argv[3] == 'input':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import biblioteca as biblio 
    biblio.compare_pickle_against_unknown_images(pickle_file, image_dir)
elif sys.argv[1] == 'readVideo':
    if param_length == 2:
        video_input = '/tmp/video/test_video.mp4'
        data_file = '/tmp/known_faces.dat'
    elif param_length == 5 and sys.argv[3] == 'input':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import biblioteca as biblio 
    biblio.read_video(video_input, data_file)
elif sys.argv[1] == 'compareData':
    if param_length == 2:
        video_data_file = '/tmp/known_faces.dat'
        known_data_file = '/tmp/train.pkl'
    elif param_length == 5 and sys.argv[3] == 'known_data':
        image_dir = sys.argv[2]
        pickle_file = sys.argv[4]
    else:
        com.log_error(msg)

    import biblioteca as biblio 
    biblio.compare_data(video_data_file, known_data_file)
else:
    com.log_error(msg)

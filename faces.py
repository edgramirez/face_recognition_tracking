import sys
import biblioteca as biblio 

param_length = len(sys.argv)
#print(param_length)
#print(sys.argv[1])
#quit()

msg = 'Usage: ' + sys.argv[0] + ' [load /PATH/TO/IMAGE/FILES output /PATH/TO/OUTPUT_FILE | find_images KNOWN_INPUT_DATA_FILE | read_video | find_video '

if param_length < 2:
    biblio.log_error(msg)

if sys.argv[1] == 'load' and param_length > 4 and sys.argv[3] == 'output':
    known_faces = sys.argv[2]
    pickle_file = sys.argv[4]
    biblio.encode_known_faces(known_faces, pickle_file)
elif sys.argv[1] == 'find_images' and param_length > 2:
    image_dir = 'images/find'
    pickle_file = sys.argv[2]
    biblio.compare_pickle_against_unknown_images(pickle_file, image_dir)
else:
    biblio.log_error(msg)

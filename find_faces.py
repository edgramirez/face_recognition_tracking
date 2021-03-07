import biblioteca as biblio

image_dir = 'images/Unknown_faces/obama'
pickle_file = 'train.pkl'
video = False

biblio.compare_pickle_against_unknown(pickle_file, image_dir, video)

import biblioteca as biblio

def mi_main_loop():
    video_input = '/home/edgar/Downloads/La_cronica_triunfo_AMLO.mp4'
    data_file = '/tmp/known_faces.dat'

    biblio.read_video(video_input, data_file)

if __name__ == "__main__":
    mi_main_loop()

Usage: ./faces.py newDb | findImg | readVideo | findVideo | appendTo | compareData | findSilence

	newDb - crea un nuevo archivo con la codificacion de los rostros y meta information, por default en /tmp/train.pkl

	findImg - busca en imagenes basado en el archivo de rostros codificados, usando /tmp/train.pkl por default

	readVideo - lee el video ubicado por default en /tmp/video/test_video.mp4 y genera el archivo de rostros codificados /tmp/known_faces.dat

	appendTo - para agregar mas imagenes de al archivo de codificacion, por default /tmp/train.pkl

	compareData - TODO para comparar 2 bases de datos

	findSilence - TODO para buscar imagen en video sin agregar los nuevos desconocidos y solo presentar las coincidencias

import pytube

a = "https://www.youtube.com/watch?v=Im3KS2D-nFY"
a = "https://www.youtube.com/watch?v=uWRmBjFxttc"
a = "https://www.youtube.com/watch?v=CuzLdMPbKl8"
a = "https://www.youtube.com/watch?v=yeq56BE_Pjs"
a = "https://www.youtube.com/watch?v=BAjSPqxP6cw"
a = "https://www.youtube.com/watch?v=XeFzDQCUjzE"
print(type(pytube))
print(dir(pytube))

you = pytube.YouTube(a)
video = you.streams.get_highest_resolution()
video.download("/home/edgar/Downloads")

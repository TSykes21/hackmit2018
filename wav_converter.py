from pydub import AudioSegment
import os


directory = "./data"


rootdir = './data'

print(os.path)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if '.mp3' in os.path.join(subdir, file):
            sound = AudioSegment.from_mp3(os.path.join(subdir, file))
            #print(os.path.join(subdir, file))
            #print(os.path.join(subdir) + "/" + file[:-4] + ".wav")
            sound.export(os.path.join(subdir) + "/" + file[:-4] + ".wav", format="wav")
        if '.mp3.wav' in os.path.join(subdir, file):
            os.remove(os.path.join(subdir, file))




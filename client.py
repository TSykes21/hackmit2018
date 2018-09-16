import pyaudio
import wave
import socket
import sys

#Server setup
server_ip = "35.221.27.142"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_addr = ('localhost', 10000)

#.wav file setup
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

#.wav file recording
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("*recording")

frames = []

for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)):
	data = stream.read(CHUNK)
	frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

#Write .wav
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

with open("output.wav", 'rb') as fd:
	data = fd.read()

temp = []

while(len(data) > 4096):
	piece = data[:4096]
	temp.append(piece)

	data = data[4096:]

	#Send .wav to server
	sock.sendto(piece, server_addr)

sock.sendto(data, server_addr)

#Receive .wav from server
while(len(s_data) < 440000):
	ser_data, server = sock.recvfrom(4096)
	s_data += ser_data

f = open("output.wav", 'w')
f.write(s_data)
f.close()

#Play .wav
p = pyaudio.PyAudio()

wf = wave.open("output.wav", 'rb')

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

w_data = wf.readframes(CHUNK)

while w_data != '':
    stream.write(data)
    w_data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()

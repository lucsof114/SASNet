import sounddevice as sd
import sys 
import torch
from params import params
from scipy.io import wavfile
import os
from wave_u_net import WaveUNet
import numpy as np

def get_song(idx): 
	mxPath = "../data/DSD100/Mixtures/Dev/"
	srcPath = "../data/DSD100/Sources/Dev/"
	filename = os.listdir(mxPath)[idx]
	songs = {}
	songs["mix"] = np.mean(wavfile.read(mxPath + filename + "/mixture.wav")[1], axis = 1)
	for inst in ["bass", "drums", "vocals", "other"]: 
		songs[inst] = np.mean(wavfile.read(srcPath + filename + "/" + inst + ".wav")[1], axis = 1)
	return songs



def estimate_tracks(model, song): 

	song = torch.from_numpy(song).float()
	song = song.view(1, -1)
	N = song.shape[1]
	ypred = []
	for i in range(int(N/params["song_length"])):
		segment = song[0, i * params["song_length"]:(i + 1) * params["song_length"]].view(1, 1, -1)
		mx = torch.max(segment)
		segment *= 1 if mx == 0 else 1/mx
		segment = model(segment)
		ypred.append(segment[0])

	return torch.cat(ypred, axis = 1)

def test_song(useModel, model_name, song_ind):
	tracks = get_song(song_ind)

	if useModel: 
		model = WaveUNet()
		model.load_state_dict(torch.load("save/" + model_name + "/" + model_name +".pkl"))
		with torch.no_grad():
			ypred = estimate_tracks(model, tracks["mix"])
	else:
		ypred = []
		for inst in ["bass", "drums", "vocals", "other"]: 
			ypred.append(tracks[inst])
		ypred = np.vstack(ypred)
	while True:
		user_input = input ("choose track [mix/bass/drums/vocals/other/quit]")
		if user_input == "quit":
			return 
		if user_input == "bass":
			sd.play(ypred[0,:], blocking = True)
		elif user_input == "drums":
			sd.play(ypred[1, :], blocking = True)
		elif user_input == "vocals":
			sd.play(ypred[2, :], blocking = True)
		elif user_input == "other":
			sd.play(ypred[3, :], blocking = True)
		elif user_input == "mix":
			sd.play(tracks["mix"],  blocking = True)
		else:
			print("not valid input")



if __name__ == '__main__':
	if len(sys.argv) != 4: 
		print("Please add the following input arguments: ")
		print("1. For model generated output add -m flag else add -t")
		print("2. Model Name")
		print("3. Song Index (number between 0 and 50)")

	test_song(sys.argv[1] == "-m", sys.argv[2], int(sys.argv[3]))
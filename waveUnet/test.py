
import sys 
import torch
from params import params
from scipy.io import wavfile
import os
from wave_u_net import WaveUNet
import numpy as np
from utils import torch2wav, numpy2wav

def get_song(idx): 
	mxPath = "../data/DSD100/Mixtures/Dev/"
	srcPath = "../data/DSD100/Sources/Dev/"
	filename = os.listdir(mxPath)[idx]
	svpath = "test/" + filename
	if not os.path.isdir(svpath):
		os.mkdir("test/" + filename)
	songs = {}
	songs["mix"] = np.mean(wavfile.read(mxPath + filename + "/mixture.wav")[1][::2], axis = 1, keepdims = True)
	numpy2wav(songs["mix"].T, ["mix"], svpath)

	for inst in ["bass", "drums", "vocals", "other"]: 
		songs[inst] =  np.mean(wavfile.read(srcPath + filename + "/" + inst + ".wav")[1][::2], axis = 1, keepdims = True)
		numpy2wav(songs[inst].T, [inst], svpath)

	return songs, svpath



def estimate_tracks(model, song): 

	song = torch.from_numpy(song).float()
	song = song.view(1, -1)
	N = song.shape[1]
	ypred = []
	for i in range(int(N/params["song_length"])): #int(N/params["song_length"])
		segment = song[0, i * params["song_length"]:(i + 1) * params["song_length"]].view(1, 1, -1)
		segment = model(segment)
		ypred.append(segment[0])

	return torch.cat(ypred, axis = 1)

def test_song(model_name, song_ind):
	tracks, svpath = get_song(song_ind)
	instrument = ["bass", "drums", "vocals", "other"]
	model = WaveUNet()
	model.load_state_dict(torch.load("save/" + model_name + "/" + model_name +".pkl"))
	with torch.no_grad():
		ypred = estimate_tracks(model, tracks["mix"])
		torch2wav(ypred, ["predicted_" + x for x in instrument], svpath)


if __name__ == '__main__':
	if len(sys.argv) != 3: 
		print("Please add the following input arguments: ")
		print("1. Model Name")
		print("2. Song Index (number between 0 and 50)")

	test_song(sys.argv[1], int(sys.argv[2]))
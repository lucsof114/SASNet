import sys 
import torch
from params import params
from scipy.io import wavfile
import os
from transformerNet import TransformerNet
import numpy as np
from utils import *
import json

def get_song(model_name, idx): 
	mxPath = "../data/DSD100/Mixtures/Dev/"
	srcPath = "../data/DSD100/Sources/Dev/"
	filename = os.listdir(mxPath)[idx]
	svpath = "test/{}/{}".format(model_name, filename)
	songs = {}
	track = wavfile.read(mxPath + filename + "/mixture.wav")[1][::2]
	songs["mix"] = track / np.max(np.abs(track))
	print("mix", track.shape)
	numpy2wav(np.mean(songs["mix"], axis=1).reshape(1, -1), ["mix"], svpath)

	for inst in ["bass", "drums", "vocals", "other"]: 
		track = wavfile.read(srcPath + filename + "/" + inst + ".wav")[1][::2]
		songs[inst] = track / np.max(np.abs(track))
		print(inst, songs[inst].shape)
		numpy2wav(np.mean(songs[inst], axis=1).reshape(1, -1), [inst], svpath)

	return songs, svpath, filename

def estimate_tracks(model, song): 
	print(song.shape)
	N = song.shape[1]
	ypred = []
	for i in range(int(N/params["song_length"])): #int(N/params["song_length"])
		segment = song[0, i * params["song_length"]:(i + 1) * params["song_length"]].view(1, 1, -1)
		segment = model(segment)
		ypred.append(segment[0])
	return torch.cat(ypred, axis = 1)

def test_song(model_name, song_ind, metrics):
	tracks, svpath, song_name = get_song(model_name, song_ind)
	instrument = ["bass", "drums", "vocals", "other"]
	device, gpu_ids = get_available_devices()
	print(device)
	model = TransformerNet()
	model.load_state_dict(torch.load("save/" + model_name + "/" + model_name +".pkl", map_location=device))
	with torch.no_grad():
		Ydata = torch.stack([tracks[instr] for instr in instrument], axis = 0)
		Xdata = torch.sum(Ydata, axis = 0, keepdims = True)
		ypred = estimate_tracks(model, Xdata)
		loss = torch.mean((ypred - Ydata)**2).item()
		print(song_name, "has loss", loss)
		if metrics:
			metrics[song_name] = loss
		torch2wav(ypred, ["predicted_" + x for x in instrument], svpath)

if __name__ == '__main__':
	if len(sys.argv) != 3: 
		print("Please add the following input arguments: ")
		print("1. Model Name")
		print("2. Either: all, many, or Song Index (number between 0 and 50)")
		# All: 50 songs, Many: 10 songs, otherwise uses given song index

	if not os.path.isdir("test/" + str(sys.argv[1]) + "/"):
		os.mkdir("test/" + str(sys.argv[1]) + "/")

	metrics = {}
	if sys.argv[2] == "all":
		for i in range(50):
			test_song(sys.argv[1], i, metrics)
	elif sys.argv[2] == "many":
		for i in range(5):
			test_song(sys.argv[1], i, metrics)
	else:
		metrics = None
		test_song(sys.argv[1], int(sys.argv[2]), metrics)
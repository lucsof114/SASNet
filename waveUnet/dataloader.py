
import torch
from scipy.io import wavfile

from params import params
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


## TODO : DATA AUGMENTATION AS DESCRIBED IN PAPER
class DSD100(Dataset):

	def __init__(self):

		#Load Paths 
		path = "../data/DSD100"
		mixPath = path + "/Mixtures/"
		sourcePath = path + "/Sources/"
		self.instruments = ["bass", "drums", "vocals", "other"]
		self.pathDict = { "X": [],  
			 			  "Y": {} }


		self.size = 0
		for inst in self.instruments:
			self.pathDict["Y"][inst] = []

		for dtype in ["Dev", "Test"]:
			for file in os.listdir(mixPath + dtype):
				sgPath = mixPath + dtype + "/" + file + "/mixture.wav"
				_, song = wavfile.read(sgPath)
				for i in range(int(song.shape[0]/params["song_length"])):
					self.pathDict["X"].append((sgPath, [i * params["song_length"], (i + 1) * params["song_length"]]))
					self.size += 1

			for file in os.listdir(sourcePath + dtype):
				for inst in self.instruments:
					sgPath = sourcePath + dtype + "/" + file + "/" + inst + ".wav"
					_, song = wavfile.read(sgPath)
					for i in range(int(song.shape[0]/params["song_length"])):
						self.pathDict["Y"][inst].append((sgPath, [i * params["song_length"], (i + 1) * params["song_length"]]))
	




	def __len__(self): 
		return self.size

	def loadsong(self, loc): 
		data = wavfile.read(loc[0])[1][loc[1][0] : loc[1][1]]
		
		return torch.from_numpy(data).float()

	def __getitem__(self, idx): 

		Ydata = [] 
		for inst in self.instruments: 
			Ydata.append(torch.mean(self.loadsong(self.pathDict["Y"][inst][idx]), axis = 1))

		Ydata = torch.stack(Ydata, axis = 0)

		if params["enforce_sum"] != "True":
			Xdata = self.loadsong(self.pathDict["X"][idx])
			if params["stereo"] == "False":
				Xdata = torch.mean(Xdata, axis = 1, keepdims = True).transpose(0, 1)
		else : 
			Xdata = torch.sum(Ydata, axis = 0, keepdims = True)

		return Xdata, Ydata




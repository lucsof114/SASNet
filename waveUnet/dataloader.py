
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

		for inst in self.instruments:
			self.pathDict["Y"][inst] = []

		for dtype in ["Dev", "Test"]:
			for file in os.listdir(mixPath + dtype):
				self.pathDict["X"].append(mixPath + dtype + "/" + file + "/mixture.wav")

			for file in os.listdir(sourcePath + dtype):
				for inst in self.instruments: 
					self.pathDict["Y"][inst].append(sourcePath + dtype + "/" + file + "/" + inst + ".wav")




	def __len__(self): 
		return params["nData"]

	def loadsong(self, path): 
		data = wavfile.read(path)[1] # may be wrong [n: (n + params["song_length"]
		n = 700000

		data = data[n: (n + params["song_length"])]

		mx = np.max(np.abs(data), axis = 0)
		for i in [0, 1]:
			if mx[i] == 0: 	
				mx[i] = 1
		data = data / mx
		return torch.from_numpy(data).float()

	def __getitem__(self, idx): 
		Xdata = self.loadsong(self.pathDict["X"][idx])
		if params["stereo"] == "False":
			Xdata = torch.mean(Xdata, axis = 1, keepdims = True).transpose(0, 1)

		Ydata = [] 
		for inst in self.instruments: 
			Ydata.append(torch.mean(self.loadsong(self.pathDict["Y"][inst][idx]), axis = 1))

		return Xdata, torch.stack(Ydata, axis = 0)




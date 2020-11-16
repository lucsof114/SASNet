
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
        self.load_data(params["overfit_dataset_size"] if params["overfit"] else None)

    def load_data(self, n = None):
        path = "../data/DSD100"
        mixPath = path + "/Mixtures/"
        sourcePath = path + "/Sources/"
        maxSize = n if n is not None else -1
        self.instruments = ["bass", "drums", "vocals", "other"]
        self.pathDict = { "X": [],  "Y": {} }
        self.songs = {}

        self.size = 0
        for inst in self.instruments:
            self.pathDict["Y"][inst] = []
        print("Loading data")
        for dtype in ["Dev", "Test"]:
            for file in os.listdir(mixPath + dtype):
                mxPath = mixPath + dtype + "/" + file + "/mixture.wav"
                base_srcPath = sourcePath + dtype + "/" + file + "/"
                track = wavfile.read(mxPath)[1][::2]
                self.songs[mxPath] = track/np.max(np.abs(track))
                N = self.songs[mxPath].shape[0]
                for inst in self.instruments:
                	srcPath = base_srcPath + inst + ".wav"
                	track = wavfile.read(srcPath)[1][::2]
                	self.songs[srcPath] = track/np.max(np.abs(track))

                for i in range(int(N/params["song_length"])):
                    self.pathDict["X"].append((mxPath, [i * params["song_length"], (i + 1) * params["song_length"]]))
                    for inst in self.instruments:
                    	srcPath = base_srcPath + inst + ".wav"
                    	self.pathDict["Y"][inst].append((srcPath, [i * params["song_length"], (i + 1) * params["song_length"]]))
                    self.size += 1
                    if self.size == maxSize: 
                    	return


    def __len__(self): 
        return self.size

    def loadsong(self, loc): 
        data = self.songs[loc[0]][loc[1][0]:loc[1][1]]
        return torch.from_numpy(data).float()

     

    def __getitem__(self, idx): 

        Ydata = [] 
        for inst in self.instruments: 
            Ydata.append(torch.mean(self.loadsong(self.pathDict["Y"][inst][idx]), axis = 1))

        Ydata = torch.stack(Ydata, axis = 0)

        if not params["enforce_sum"]:
            Xdata = self.loadsong(self.pathDict["X"][idx])
            if not params["stereo"]:
                Xdata = torch.mean(Xdata, axis = 1, keepdims = True).transpose(0, 1)
        else: 
            Xdata = torch.sum(Ydata, axis = 0, keepdims = True)
        return Xdata, Ydata




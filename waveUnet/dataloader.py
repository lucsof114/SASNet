
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
                srcPath = sourcePath + dtype + "/" + file + "/" + inst + ".wav"

                self.songs[mxPath] = torch.from_numpy(wavfile.read(mxPath)[1]).float()
                N = self.songs[mxPath].shape[0]
                for inst in self.instruments:
                	self.songs[srcPath] = torch.from_numpy(wavfile.read(srcPath)[1]).float()

                for i in range(int(N/params["song_length"])):
                    self.pathDict["X"].append((mxPath, [i * params["song_length"], (i + 1) * params["song_length"]]))
                    for inst in self.instruments:
                    	self.pathDict["Y"][inst].append((srcPath, [i * params["song_length"], (i + 1) * params["song_length"]]))
                    self.size += 1
                    if self.size == maxSize: 
                    	return


    def __len__(self): 
        return self.size

    def loadsong(self, loc): 
        data = self.songs[loc[0]][loc[1][0]:loc[1][1]]
        mx = torch.max(data)
        return data if mx == 0 else data / mx

     

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




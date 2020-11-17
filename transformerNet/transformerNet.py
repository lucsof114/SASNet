import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from params import params
from utils import *
#STANT95919


class EncoderBlock(nn.Module):

	def __init__(self, targetLen, inputLen, embedLen):
		super(EncoderBlock, self).__init__()

		self.multiHead = nn.MultiheadAttention(embed_dim = embedLen, 
											   num_heads = params["num_heads"])
		self.drop1 = nn.Dropout(params["dropout"])
		self.norm1 = nn.LayerNorm([embedLen, targetLen])
		self.l1 = nn.Linear(targetLen, targetLen)
		self.l2 = nn.Linear(targetLen, targetLen)
		self.drop2 = nn.Dropout(params["dropout"])
		self.norm2 = nn.LayerNorm([embedLen, targetLen])

	def forward(self, kq, v): 
		#E = embedLen, N = batch, S = inputLen, L = targetLen
		#v = (N, E, S)
		#kq = (N, E, L)

		v = v.permute(2, 0, 1) #v = (S, N, E)
		kq = kq.permute(2, 0, 1) #kq = (L, N, E)
		v = v + self.drop1(self.multiHead(kq, kq, v)[0]) #v = (L, N, E)
		v = self.norm1(v.permute(1, 2, 0)) #v = (N, E, L)
		v = self.l2(F.relu(self.l1(v)))
		v = self.norm2(v + self.drop2(v)) #v = (N, E, L)
		return v

class DFTEmbedder(nn.Module):

	def __init__(self):
		super(DFTEmbedder, self).__init__()
		self.nfft = 2 *  params["hidden_size"] + 2
		self.spector = torchaudio.transforms.Spectrogram(n_fft = self.nfft)

	def forward(self, x):

		spect =  self.spector(x)
		return spect[:,0,1:-1]

class TransformerNet(nn.Module):
	def __init__(self):
		super(TransformerNet, self).__init__()

		h = params["hidden_size"]
		self.dftEnc = DFTEmbedder() 
		kern = int(params["kernel_size"])
		
		self.timeSigEncoder = nn.ModuleList()
		for sze in [[1, 2], [2, 4], [4,8], [8, 16], [16, 32], [32, 64], [64, 128]] :
			self.timeSigEncoder.append(nn.Conv1d(sze[0], sze[1], kernel_size = 15, padding = 15//2))

		self.dftEncBlks =  nn.ModuleList()
		self.timeEncBlks1 =  nn.ModuleList()
		self.timeEncBlks2 =  nn.ModuleList()
		for i in range(params["numBlks"]):
			self.dftEncBlks.append(EncoderBlock(h, h, h))
			self.timeEncBlks1.append(EncoderBlock(h, h, h))
			self.timeEncBlks2.append(EncoderBlock(h, h, h))

		self.timeSigDecoder =  nn.ModuleList()
		for sze in [[256, 128], [192, 64], [96, 32], [48, 16], [24, 8], [12, 4], [6, params["k"] - 1]]: 
			self.timeSigDecoder.append(nn.Conv1d(sze[0], sze[1], kernel_size = 5, padding = 5//2))


	def forward(self, x):

		currDFT = self.dftEnc(x)

		down_pipe = [x]
		currSig = x
		for i in range(7): 
			junction = F.relu(self.timeSigEncoder[i](currSig))
			down_pipe.append(junction)
			currSig = F.interpolate(junction, scale_factor=0.5, mode='nearest')

		for i in range(params["numBlks"]):
			currDFT = self.dftEncBlks[i](currDFT, currDFT)
			currSig = self.timeEncBlks1[i](currSig, currSig)
			currSig = self.timeEncBlks2[i](currDFT, currSig)

		for i in range(7): 
			currSig = torch.cat((down_pipe[7 - i], F.interpolate(currSig, scale_factor= 2.0, mode='linear')), dim=1) 
			if i == 6:
				currSig = torch.tanh(self.timeSigDecoder[i](currSig))
			else:
				currSig = F.relu(self.timeSigDecoder[i](currSig)) 

		out_diff = x - torch.sum(currSig, 1, True)
		out = torch.cat((currSig, out_diff), dim=1)
		return out



model = TransformerNet()
x = torch.randn(16, 1, 16384)
print(model(x).shape)

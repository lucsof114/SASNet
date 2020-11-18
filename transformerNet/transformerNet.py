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

	def forward(self, v, k ,q): 
		#E = embedLen, N = batch, S = inputLen, L = targetLen
		#v = (N, E, S)
		#kq = (N, E, L)
		# import pdb 
		# pdb.set_trace()

		v = v.permute(2, 0, 1) #v = (S, N, E)
		k = k.permute(2, 0, 1) #k = (S, N, E)
		q = q.permute(2, 0, 1) #q = (L, N, E)
		q = q + self.drop1(self.multiHead(q, k, v)[0]) #v = (L, N, E)
		q = self.norm1(q.permute(1, 2, 0)) #v = (N, E, L)
		q = self.l2(F.relu(self.l1(q)))
		q = self.norm2(q + self.drop2(q)) #v = (N, E, L)
		return q

class DFTEmbedder(nn.Module):

	def __init__(self):
		super(DFTEmbedder, self).__init__()
		self.nfft = 2 * 216 - 2
		self.spector = torchaudio.transforms.Spectrogram(n_fft = self.nfft)

	def forward(self, x):

		spect =  self.spector(x)
		return spect[:,0]

class TransformerNet(nn.Module):
	def __init__(self):
		super(TransformerNet, self).__init__()

		kern = int(params["kernel_size"])
		self.L = params["L"]
		self.Fc = params["Fc"]
		self.fu = params["fu"]



		self.dftEnc = DFTEmbedder() 

		self.timeSigEncoder = nn.ModuleList()
		for i in range(self.L + 1):
			inchannel = self.Fc * i if i > 0 else 1
			self.timeSigEncoder.append(nn.Conv1d(inchannel, self.Fc * (i+1), kernel_size = 15, padding = 15//2))

		self.dftEncBlks =  nn.ModuleList()
		self.timeEncBlks1 =  nn.ModuleList()
		self.timeEncBlks2 =  nn.ModuleList()

		for i in range(params["numBlks"]):
			self.dftEncBlks.append(EncoderBlock(77, 77, 216)) #53, 53, 312
			self.timeEncBlks1.append(EncoderBlock(64, 64, 216)) #4, 4, 312
			self.timeEncBlks2.append(EncoderBlock(64, 77, 216)) #4, 53, 312

		self.timeSigDecoder =  nn.ModuleList()

		for i in range(self.L):
			downsample_inchannel = self.Fc * (i+1)
			upsample_inchannel = self.Fc * (i+2)
			self.timeSigDecoder.append(nn.Conv1d(in_channels= downsample_inchannel + upsample_inchannel, out_channels= self.Fc * (i+1), kernel_size=self.fu, padding=self.fu//2))
		
		self.final_conv = nn.Conv1d(in_channels=self.Fc + 1, out_channels=params["K"]-1, kernel_size=1)

	def forward(self, x):

		currDFT = self.dftEnc(x)

		down_pipe = [x]
		currSig = x  
		for i in range(self.L + 1): 
			junction = F.relu(self.timeSigEncoder[i](currSig))
			down_pipe.append(junction)
			currSig = F.interpolate(junction, scale_factor=0.5, mode='nearest')



		for i in range(params["numBlks"]):
			currDFT = self.dftEncBlks[i](currDFT, currDFT, currDFT)
			down_pipe[-1] = self.timeEncBlks1[i](down_pipe[-1], down_pipe[-1], down_pipe[-1])
			down_pipe[-1] = self.timeEncBlks2[i](currDFT, currDFT, down_pipe[-1])


		up_pipe = [down_pipe[-1]]
		for i in range(self.L):
			tmp = torch.cat((down_pipe[self.L - i], F.interpolate(up_pipe[-1], scale_factor=2.0, mode='linear')), dim=1)
			up_pipe.append(F.leaky_relu(self.timeSigDecoder[self.L - i - 1](tmp)))

		tmp = torch.cat((x, up_pipe[-1]), dim=1)
		out_k_minus_1 = torch.tanh(self.final_conv(tmp))
		out_diff = x - torch.sum(out_k_minus_1, 1, True)
		out = torch.cat((out_k_minus_1, out_diff), dim=1)
		return out



# model = TransformerNet()
# x = torch.randn(16, 1, 16384)
# print(model(x).shape)

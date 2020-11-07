import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveUNet(nn.Module):
    def __init__(self, L, K, Fc, fd, fu):
        super(WaveUNet, self).__init__()
        self.L = L
        self.K = K
        self.Fc = Fc
        self.fd = fd
        self.fu = fu

        self.down_convs = nn.ModuleList()
        for i in range(L + 1):
        	inchannel = Fc * i if i > 0 else 1
        	self.down_convs.append(nn.Conv1d(in_channels=inchannel, out_channels=Fc * (i+1), kernel_size=fd, padding=fd//2))
        	#print(inchannel, Fc * (i+1))

        
        self.up_convs = nn.ModuleList()
        for i in range(L):
        	downsample_inchannel = Fc * (i+1)
        	upsample_inchannel = Fc * (i+2)
        	self.up_convs.append(nn.Conv1d(in_channels= downsample_inchannel + upsample_inchannel, out_channels=Fc * (i+1), kernel_size=fu, padding=fu//2))
        	#print(downsample_inchannel, upsample_inchannel, downsample_inchannel + upsample_inchannel, Fc * (i+1))

        self.final_conv = nn.Conv1d(in_channels=Fc + 1, out_channels=K-1, kernel_size=1)

    def forward(self, x):
    	L = self.L
    	#print('Down Pipe')
    	down_pipe = [x]
    	curr = x
    	for i in range(L + 1):
    		junction = F.leaky_relu(self.down_convs[i](curr))
    		down_pipe.append(junction)
    		curr = F.interpolate(junction, scale_factor=0.5, mode='nearest')

    	#print('Up Pipe')
    	up_pipe = [down_pipe[-1]]
    	for i in range(L):
    		tmp = torch.cat((down_pipe[L - i], F.interpolate(up_pipe[-1], scale_factor=2, mode='linear')), dim=1)
    		up_pipe.append(F.leaky_relu(self.up_convs[L - i - 1](tmp)))

    	tmp = torch.cat((x, up_pipe[-1]), dim=1)
    	out_k_minus_1 = torch.tanh(self.final_conv(tmp))
    	out_diff = x - torch.sum(out_k_minus_1, 1, True)
    	out = torch.cat((out_k_minus_1, out_diff), dim=1)
    	#print(out.shape)
    	return out


wavenet = WaveUNet(12, 5, 24, 15, 5)

# Simple test to check dims
input = torch.randn(1, 1, 16384)
wavenet(input)


'''

Initial testing. May remove:

input = torch.randn(1, 1, 16384)
m = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=15, padding=7)
print(input.shape)
output = F.leaky_relu(m(input))
print(output.shape)

decimated_output = F.interpolate(output, scale_factor=0.5, mode='nearest')
print(decimated_output.shape)


m2 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=15, padding=7)
output2 = F.leaky_relu(m2(decimated_output))
print(output2.shape)
'''
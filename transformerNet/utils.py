import torch
import os 
import sys
import json
import shutil
from scipy.io import wavfile
from params import params
import numpy as np

def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def torch2wav(tensor, names, svpath):
    assert len(tensor.shape) == 2
    assert len(names) == tensor.shape[0]
    for i in range(tensor.shape[0]):
        track = tensor[i]/torch.max(torch.abs(tensor[i]))
        wavfile.write(svpath + "/" + names[i] + ".wav", params["fs"], track.numpy())


def numpy2wav(arr, names, svpath):
    assert len(arr.shape) == 2
    assert len(names) == arr.shape[0]
    for i in range(arr.shape[0]):
        track = arr[i]/np.max(np.abs(arr[i]))
        wavfile.write(svpath + "/" + names[i] + ".wav", params["fs"], track)

def wav2numpy(filename, downsample, isMono):
    track = wavfile.read(filename)[1]
    if isMono: 
        track = np.mean(track, axis = 1)
    if downsample:
        track = track[::2]
    return track

def getSegmentSize(hiddenSize, desiredSize):
    return (desiredSize//2 * (2*hiddenSize + 2)) - hiddenSize + 1

def save_params(params): 
    assert len(sys.argv) > 1 , "Please provide a name to your model"
    answer = None
    if os.path.exists(("save/" + str(sys.argv[1]) + "/" + str(sys.argv[1]) + ".json")):
        answer = input(str(sys.argv[1]) + " already exists. Do you want to overwrite [y/n]: " )
    if answer != "y" and answer != None: 
        sys.exit()
    if answer == "y":
        shutil.rmtree("save/" + str(sys.argv[1]) + "/")
    os.mkdir("save/" + str(sys.argv[1]) + "/")
    with open(("save/" + str(sys.argv[1]) + "/" + str(sys.argv[1]) + ".json"), 'w') as fp:
        json.dump(params,sort_keys=True, indent=4,fp= fp)



class AverageMeter:
	"""Keep track of average values over time.
	Adapted from:
	> https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	def __init__(self):
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		"""Reset meter."""
		self.__init__()

	def update(self, val, num_samples=1):
		"""Update meter with new value `val`, the average of `num` samples.
		Args:
		    val (float): Average value to update the meter with.
		    num_samples (int): Number of samples that were averaged to
		        produce `val`.
		"""
		self.count += num_samples
		self.sum += val#* num_samples
		self.avg = self.sum / self.count
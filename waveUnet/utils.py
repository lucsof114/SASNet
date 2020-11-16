import torch
import os 
import sys
import json
import shutil

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



def save_params(params): 
    assert len(sys.argv) > 1 , "Please provide a name to your model"
    answer = None
    if os.path.exists(("save/" + str(sys.argv[1]) + "/" + str(sys.argv[1]) + ".json")):
        answer = input(str(sys.argv[1]) + " already exists. Do you want to overwrite [y/n]: " )
    if answer != "y" and answer != None: 
        sys.exit()
    if answer == "y":
        shutil.rmtree("save/" + str(sys.argv[1]) + "/")
    os.system("sudo mkdir save/" + str(sys.argv[1]) )
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

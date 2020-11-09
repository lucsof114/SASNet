import torch
from tqdm import tqdm
from params import params

from dataloader import DSD100
from utils import *
import sys
import torch.utils.data as td

from wave_u_net import WaveUNet
from tensorboardX import SummaryWriter

import torch.nn.functional as F

def getDataLoader(dataset): 
	N = len(dataset)
	split_frac = params["train_dev_overfit"] if params["overfit"] == "True" else params["train_dev_normal"]

	lengths = [int(N*split_frac), int(N * (1 - split_frac))]
	train_set, dev_set = td.random_split(dataset, lengths)
	train_loader = td.DataLoader(train_set, batch_size = params["batch_size"], shuffle = True, drop_last = True)

	print("Training set :{}".format(lengths[0]))
	print("Dev set:{}".format(lengths[1]))
	if params["overfit"] == "True": 
		print("Overfitting Mode")
		dev_loader = td.DataLoader(train_set, batch_size = params["batch_size"], drop_last = True)
	else: 
		print("Training Mode")
		dev_loader = td.DataLoader(dev_set, batch_size = params["batch_size"], drop_last = True)

	return train_loader, dev_loader


def train(train_loader, dev_loader, model, device): 
	N_epoch = params["num_epochs"]
	tb_writer = SummaryWriter("save" + params["name"] + "/")

	avg = AverageMeter()
	optimizer = torch.optim.Adam(model.parameters(), lr = params["learning_rate"])

	time2eval = params["evaluate_every"]

	num_steps = 0

	for epoch in range(N_epoch):
		print("Starting Epoch: ", epoch)
		avg.reset()
		with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as pbar:
			for (X, ys) in train_loader:
				batch_size = X.shape[0]
				num_steps += batch_size
				# import pdb
				# pdb.set_trace()

				X = X.to(device)
				ys = ys.to(device)
				optimizer.zero_grad()
				y_pred = model(X)

				loss = F.mse_loss(y_pred, ys)
				loss_value = loss.item()

				pbar.update(batch_size)
				avg.update(loss_value, 1)
				pbar.set_postfix(loss =avg.avg, epoch= epoch)

				loss.backward()
				optimizer.step()

				tb_writer.add_scalar("batch train loss", loss_value, num_steps )

				time2eval -= batch_size
				if time2eval <= 0: 
					print("Evaluating...")
					time2eval = params["evaluate_every"]
					loss_dev = eval(model, dev_loader, device)
					tb_writer.add_scalar('dev loss', loss_dev, num_steps)



def eval(model, loader, device): 
	model.eval()

	loss = 0 
	accuracy = 0 
	num = 0 
	avg = AverageMeter()

	with torch.no_grad(),tqdm(total=len(loader.dataset)) as pbar2:

		for batch_index,(X, ys) in enumerate(loader):
			X = X.to(device)
			ys = ys.to(device)

			num += X.shape[0]

			y_pred = model(X)
			loss += F.mse_loss(y_pred, ys).item()     

			avg.update(loss, 1)
			pbar2.update(X.shape[0])
			pbar2.set_postfix(loss =avg.avg)

	avg_loss = loss/num
	if params["best_val_loss"] is None or params["best_val_loss"] > avg_loss:
		print("Saving New Model!")
		params["best_val_loss"] = avg_loss
		torch.save(model.state_dict(), "save/" + params["name"] + "/" + params["name"] + ".pkl")

	model.train()

	return avg_loss




if __name__ == "__main__": 

	# init seeds 
	torch.manual_seed(params["seed"])
	torch.cuda.manual_seed_all(params["seed"])

	#save work
	save_params(params)
	params["name"] = sys.argv[1]

	#check devices
	device, gpu_ids = get_available_devices()
	print(device)

	#load model
	model = WaveUNet()
	model = model.to(device)
	model.train()

	#Load Data
	train_loader, dev_loader = getDataLoader(DSD100())
	params["best_val_loss"] = None

	train(train_loader, dev_loader, model, device)


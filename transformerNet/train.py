import torch
from tqdm import tqdm
from params import params
import torch.nn as nn
import sounddevice as sd

from dataloader import DSD100
from utils import *
import sys
import torch.utils.data as td

from transformerNet import TransformerNet
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn.functional as F

def getDataLoader(dataset): 
	print("start loader")
	N = len(dataset)
	split_frac = params["train_dev_split"]

	lengths = [int(N*split_frac), 0]
	lengths[1] = N - lengths[0]
	train_set, dev_set = td.random_split(dataset, lengths)
	train_loader = td.DataLoader(train_set, batch_size = params["batch_size"], shuffle = True, drop_last = True, num_workers=4, pin_memory=True)

	print("Training set :{}".format(lengths[0]))
	print("Dev set:{}".format(lengths[1]))
	if params["overfit"] == "True": 
		print("Overfitting Mode")
		dev_loader = td.DataLoader(train_set, batch_size = params["batch_size"], drop_last = True, num_workers=4, pin_memory=True)
	else: 
		print("Training Mode")
		dev_loader = td.DataLoader(dev_set, batch_size = params["batch_size"], drop_last = True, num_workers=4, pin_memory=True)

	return train_loader, dev_loader


def train(train_loader, dev_loader, model, device): 
	N_epoch = params["num_epochs"]
	tb_writer = SummaryWriter("save/" + params["name"] + "/")

	avg = AverageMeter()
	optimizer = torch.optim.Adam(model.parameters(), lr = params["learning_rate"], betas = (params["decayB1"] , params["decayB2"]), eps = params["eps"])
	
	lrUpdate = lambda x : 1/np.sqrt(216) * min(1/np.sqrt(x * params["batch_size"]), params["batch_size"]* x * 4e-6)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [lrUpdate])
	
	time2eval = params["evaluate_every"]

	num_steps = 0
	lossFn = nn.MSELoss(reduction = "sum")

	for epoch in range(N_epoch):
		print("Starting Epoch: ", epoch)
		avg.reset()
		with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as pbar:
			for (X, ys) in train_loader:
				batch_size = X.shape[0]
				num_steps += batch_size
				X = X.to(device)
				ys = ys.to(device)


				optimizer.zero_grad()
				y_pred = model(X)
				loss = lossFn(y_pred, ys)
				loss_value = loss.item()


				pbar.update(batch_size)
				avg.update(loss_value, 1)
				pbar.set_postfix(loss =avg.avg, epoch= epoch)

				loss.backward()

				optimizer.step()
				scheduler.step()

				tb_writer.add_scalar("Batched Training Loss", loss_value, num_steps )

				time2eval -= batch_size
				if time2eval <= 0: 
					print("Evaluating...")
					time2eval = params["evaluate_every"]
					loss_dev = eval(model, dev_loader, device, lossFn)
					tb_writer.add_scalar('Average Dev Loss', loss_dev, num_steps)



def eval(model, loader, device, lossFn): 
	model.eval()

	loss = 0 
	accuracy = 0 
	num = 0 
	avg = AverageMeter()

	with torch.no_grad(): #,tqdm(total=len(loader.dataset)) as pbar2:

		for batch_index,(X, ys) in enumerate(loader):
			X = X.to(device)
			ys = ys.to(device)

			num += X.shape[0]

			y_pred = model(X)
			loss += lossFn(y_pred, ys).item()     

			avg.update(loss, 1)
			# pbar2.update(X.shape[0])
			# pbar2.set_postfix(loss =avg.avg)

	avg_loss = loss/num
	print("Loss: ", loss)
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
	model = TransformerNet()

	model = model.to(device)
	model.train()
	print ("Model Generated")
	#Load Data
	train_loader, dev_loader = getDataLoader(DSD100())
	params["best_val_loss"] = None
	print("Data Loaded")

	train(train_loader, dev_loader, model, device)


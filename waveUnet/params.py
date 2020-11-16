
params = {
	"batch_size" : 32, 
	"L" : 12, 
	"K" : 4, 
	"Fc": 24, 
	"fd": 15, 
	"fu": 5, 
	"song_length": 16384, 
	"fs" : 22050, 
	"train_dev_split": 0.75,
	"overfit_dataset_size": 100,
	"overfit": False,
	"seed": 669,
	"learning_rate": 0.00001, 
	"decayB1" : 0.9,
	"decayB2" : 0.999,
	"num_epochs": 100,
	"stereo": False,
	"evaluate_every": 50000,
	"enforce_sum" : True
}

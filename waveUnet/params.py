
params = {
	"batch_size" : 64, 
	"L" : 12, 
	"K" : 4, 
	"Fc": 24, 
	"fd": 15, 
	"fu": 5, 
	"song_length": 16384, 
	"fs" : 44100, 
	"train_dev_split": 0.75,
	"overfit_dataset_size": 1000,
	"overfit": False,
	"seed": 669,
	"learning_rate": 0.000001, 
	"decayB1" : 0.9,
	"decayB2" : 0.999,
	"num_epochs": 100,
	"stereo": False,
	"evaluate_every": 2000,
	"enforce_sum" : True
}

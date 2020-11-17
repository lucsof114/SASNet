
params = {

	"Fc": 24,
	"L" : 8,
	"fu" : 5,
	"batch_size" : 32, 
	"song_length": 16384,
	"hidden_size": 128,
	"num_heads": 8,
	"dropout": 0.1,
	"numBlks": 4,
	"kernel_size": 9,
	"K": 4,
	"fs" : 22050, 
	"train_dev_split": 0.75,
	"overfit_dataset_size": 1000,
	"overfit": True,
	"seed": 669,
	"learning_rate": 0.0001, 
	"decayB1" : 0.9,
	"decayB2" : 0.999,
	"num_epochs": 5000,
	"stereo": False,
	"evaluate_every": 50000,
	"enforce_sum" : True
}

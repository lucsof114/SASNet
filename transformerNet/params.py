
params = {


	"batch_size" : 32, 
	"song_length": 16384,
	"hidden_size": 128,
	"num_heads": 8,
	"dropout": 0.0,
	"numBlks": 12,
	"kernel_size": 9,
	"k": 4,
	"fs" : 22050, 
	"train_dev_split": 0.75,
	"overfit_dataset_size": 100,
	"overfit": True,
	"seed": 669,
	"learning_rate": 0.001, 
	"decayB1" : 0.9,
	"decayB2" : 0.999,
	"num_epochs": 10000,
	"stereo": False,
	"evaluate_every": 50000,
	"enforce_sum" : True
}

from datas import prepare_dataloaders
from train import train_autoencoders
from evaluation import evaluate_models


train_dataloaders, test_dataloaders, full_test_dataset = prepare_dataloaders()

autoencoders = train_autoencoders(train_dataloaders)

evaluate_models(autoencoders, full_test_dataset)
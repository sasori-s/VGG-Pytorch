from model import VGG11
from train import Training
from preprocessing import Preprocess
from precompute_mean_std import PreCompute

def main(dataset_path):
    preprocessing = Preprocess(dataset_path, scale_size=256, purpose='training')
    dataloader = preprocessing()
    trainer = Training(dataloader)

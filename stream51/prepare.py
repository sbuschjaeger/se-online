#!/usr/bin/env python3

import sys
sys.path.append("Stream-51/")
import numpy as np
import os
import torch
import argparse
import torchvision.transforms as transforms
import torchvision
from torchvision import models

from StreamDataset import *
import tqdm

def get_model(model_name):
    if model_name == "resnet18":
        return models.resnet18(pretrained=True), 512
    else:
        return None

def main(args):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x = StreamDataset(root="data", train=True, transform=transform)
    data = torch.utils.data.DataLoader(x, batch_size=args.batch_size, num_workers=args.n_jobs) 

    # Load pretrained InceptionV3 Classifier
    model, lin_size = get_model(args.model)

    # Replace Classifier with identity transformation
    model.fc = torch.nn.Linear(lin_size, lin_size) 
    torch.nn.init.eye_(model.fc.weight)
    torch.nn.init.zeros_(model.fc.bias)

    dataset = []
    labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for x, y in tqdm.tqdm(data):
            for yy in y:
                labels.append(yy.cpu().numpy())
            z = model(x.to(device)).cpu()
            for yy in z:
                dataset.append(yy.numpy())
    dataset = np.array(dataset)
    labels = np.array(labels)
    print(dataset.shape)
    np.save(os.path.join(args.out_path, "stream51_data.npy"), dataset)
    np.save(os.path.join(args.out_path, "stream51_labels.npy"), labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="The batch size.",type=int, default=256)
    parser.add_argument("-j", "--n_jobs", help="Number of threads.",type=int, default=4)
    parser.add_argument("-m", "--model", help="The model used for computing the embeddings. Can be {resnet18}. ",type=str, default="resnet18")
    parser.add_argument("-o", "--out_path", help="Path where results should be written",type=str, default=".")

    args = parser.parse_args()

    main(args)



import json
import os
from argparse import ArgumentParser
from glob import glob

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image as Im
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from train.main import resnet50
from train.utils import load_checkpoint, optimizer_to


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.images = glob(f"{self.image_dir}/*/*.png")
        self.transform = transform

    def __getitem__(self, item):
        image = Im.open(self.images[item]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return self.images[item], image

    def __len__(self):
        return len(self.images)


def process(x):
    path = x.split('/')[-2:]
    sset = path[0][-8:][-5:] + path[0][-8:][:2] if 'v' in path[0].lower() else path[0][-5:]
    image = os.path.splitext(path[1])[0].split('_')[1] if '_' in path[1] else \
        os.path.splitext(path[1])[0].split('-')[1]
    return [sset, image]


if __name__ == '__main__':
    parser = ArgumentParser("Train models.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing positive and negative images")
    parser.add_argument("--model_dir", default=None, type=str, required=True, help="Directory to save the models")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")
    parser.add_argument("--output", type=str, required=True, help="Output file to write the predictions.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=False)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model, optimizer, _, _, _ = load_checkpoint(
        model,
        optimizer,
        os.path.join(args.model_dir, "checkpoint-best.pth.tar")
    )

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    optimizer_to(optimizer, device)

    loader = DataLoader(
        ImageDataset(
            args.image_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        ),
    )

    model.eval()
    files = []
    preds = []
    with torch.no_grad():
        for i, (image_files, images) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            pred_proba = torch.sigmoid(outputs)
            files += image_files
            preds += pred_proba.cpu().numpy().tolist()

    # Process the output file
    temp_file = os.path.splitext(args.output)[0] + ".json"
    output = dict(zip(files, preds))
    with open(temp_file, "w") as f:
        json.dump(output, f)
    df = pd.read_json(temp_file, orient='index').reset_index().rename(
        columns={"index": "image", 0: "prediction"})

    df[['set', 'image']] = pd.DataFrame.from_records(df.image.apply(process))
    df = df[['set', 'image', 'prediction']]
    df['prediction'] = df['prediction'].map('{:,.3f}'.format)
    df.to_csv(args.output, index=False)

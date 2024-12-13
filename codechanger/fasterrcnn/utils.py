import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

folder_loc = None

def process_data(folder_location):
    annotation_files = {
        'dataset': os.path.join(folder_location, 'annotations.json'),
    }

    global folder_loc
    folder_loc = folder_location

    dataframes = {}

    for key, file_path in annotation_files.items():
        if os.path.exists(file_path):
            with open(file_path) as f:
                data = json.load(f)

            images = data['images']
            annotations = data['annotations']

            rows = []
            for annotation in annotations:
                image_info = next((img for img in images if img['id'] == annotation['image_id']), None)
                if image_info is not None:
                    row = {
                        'file_name': image_info['file_name'],
                        'bbox_x': annotation['bbox'][0],
                        'bbox_y': annotation['bbox'][1],
                        'bbox_width': annotation['bbox'][2],
                        'bbox_height': annotation['bbox'][3],
                        'class_label': annotation['category_id']
                    }
                    rows.append(row)
                else:
                    print(f"Warning: No image found for annotation with image_id {annotation['image_id']}")

            df = pd.DataFrame(rows)

            df['x1'] = df['bbox_x']
            df['y1'] = df['bbox_y']
            df['x2'] = df['bbox_x'] + df['bbox_width']
            df['y2'] = df['bbox_y'] + df['bbox_height']

            df = df[['file_name', 'x1', 'y1', 'x2', 'y2', 'class_label']]

            dataframes[key] = df
        else:
            print(f"Warning: {file_path} does not exist.")

    df = dataframes.get('dataset')

    return df


def split_data(df):

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    val_df.to_csv('val.csv', index=False)

    return train_df, test_df, val_df


class Preprocess_data(Dataset):
    def __init__(self, df, file_name, indices):
        self.df = df
        self.file_name = file_name
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        file_name = self.file_name[idx]
        boxes = self.df[self.df['file_name'] == file_name][['x1', 'y1', 'x2', 'y2']].values.astype("float")
        img = Image.open(folder_loc + '/images/' + file_name).convert('RGB')
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {
            'boxes': torch.tensor(boxes),
            'labels': labels
        }
        img_tensor = self.transform(img)
        return img_tensor, target
    

def custom_collate(batch):
    imgs = []
    targets = []
    for item in batch:
        imgs.append(item[0])
        targets.append(item[1])
    return imgs, targets


def data_loader(df):
    dl = torch.utils.data.DataLoader(
        Preprocess_data(df, df['file_name'].unique(), range(len(df['file_name'].unique()))),
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate
    )

    return dl


def train_model(model, train_dl, num_epochs, optimizer, device):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        epoch_loss = 0
        for imgs, targets in train_dl:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            optimizer.zero_grad()
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()
        avg_loss = epoch_loss / len(train_dl)
        print(f'Average Loss: {avg_loss:.4f}')
    model.eval()
    return model



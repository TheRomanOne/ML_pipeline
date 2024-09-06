import os
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_folder, max_size, ratio, to_horizontal):
        self.image_folder = image_folder
        self.max_size = max_size
        self.ratio = ratio
        self.to_horizontal = to_horizontal
        self._load_images()

    def _load_images(self):
        # List all BMP files in the directory
        file_names = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.bmp', '.png'))]
        self.file_paths = [os.path.join(self.image_folder, f) for f in file_names]
        
        X_gt = []
        y_gt = []

        normalized_shape = torch.tensor([.5, 1]) * self.max_size# make square
        rat = normalized_shape[1] / normalized_shape[0]
        small_shape = (normalized_shape / self.ratio)
        normalized_shape = (int(normalized_shape[0]), int(normalized_shape[1]))
        small_shape = (int(small_shape[0]), int(small_shape[1]))



        tq = tqdm(self.file_paths)
        tq.set_description(f"Processing images")
        for file_path in tq:
            img = Image.open(file_path).convert('RGB')
            img = np.array(img)

            # when uncommenting multiply by 255 in render imaes
            # if np.mean(img) > 10:
            #     img = img / 255
            #     img = np.array(img, dtype=np.float32)

            img_shape = torch.tensor(img.shape)[:2]           
            
            if img.shape[0] < normalized_shape[0] or img.shape[1] < normalized_shape[1]:
                continue

            target_height, target_width = (img_shape[1] / rat, img_shape[1])
            if target_height > target_width:
                print('sdf')


            # Crop and return the image
            center = torch.tensor(img.shape[:2]) / 2
            center = center.int().numpy()
            x1 = center[0] - (target_height / 2).int()
            x2 = center[0] + (target_height / 2).int()
            y1 = center[1] - (target_width / 2).int()
            y2 = center[1] + (target_width / 2).int()

            _img = np.transpose(img[x1 : x2, y1 : y2, :], (2, 0, 1))
            _img = torch.tensor(_img) / 255.
            X_tr = transforms.Compose([
                transforms.Resize(small_shape)
            ])
            X_frame = X_tr(_img)
            X_gt.append(X_frame.numpy())

            y_tr = transforms.Compose([
                transforms.Resize(normalized_shape)
            ])
            y_frame = y_tr(_img)
            y_gt.append(y_frame.numpy())

        # Convert lists to tensors

        self.X_gt = torch.tensor(np.array(X_gt))
        self.y_gt = torch.tensor(np.array(y_gt))

    def __len__(self):
        return len(self.X_gt)

    def __getitem__(self, idx):
        return self.X_gt[idx], self.y_gt[idx]

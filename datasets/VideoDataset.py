import cv2, torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, video_path, max_size, ratio, to_horizontal):
        self.video_path = video_path
        self.to_horizontal = to_horizontal
        self.max_size = max_size
        self.ratio = ratio
        self._load_video_frames()

    def _load_video_frames(self):
        # Open the video file
        cap = cv2.VideoCapture(self.video_path)
        print('\n\nLoading video:', self.video_path, '---->', 'success' if cap.isOpened() else 'fail')
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        X_gt = []
        y_gt = []

        tq = tqdm(range(n_frames))
        tq.set_description(f"Processing frames")
        for _ in tq:
            ret, frame = cap.read()
            if not ret:
                break

            tr_size = (2, 1, 0) if self.to_horizontal else (2, 0, 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).float().permute(tr_size)
            # normalize to [-1 ~ 1] tensor
            frame = frame / 255.
            frame = (frame * 2) - 1

            img_shape = torch.tensor(frame.shape)[1:].numpy()

            # make square
            # img_shape[[0, 1]] = np.max(img_shape)

            normalized_shape = (self.max_size * (img_shape / np.max(img_shape)))
            small_shape = (normalized_shape/self.ratio)

            normalized_shape = (int(normalized_shape[0]), int(normalized_shape[1]))
            small_shape = (int(small_shape[0]), int(small_shape[1]))

            X_tr = transforms.Compose([
              # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
              transforms.Resize(small_shape)
            ])
            X_frame = X_tr(frame)
            X_gt.append(X_frame.numpy())

            y_tr = transforms.Compose([
              # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
              transforms.Resize(normalized_shape)
            ])
            y_frame = y_tr(frame)
            y_gt.append(y_frame.numpy())


        cap.release()
        # X_gt = np.concatenate(X_gt, axis=0)
        self.X_gt = torch.tensor(np.array(X_gt))

        # y_gt = np.concatenate(y_gt, axis=0)
        self.y_gt = torch.tensor(np.array(y_gt))

    def __len__(self):
        return len(self.X_gt)

    def __getitem__(self, idx):
        return self.X_gt[idx], self.y_gt[idx]

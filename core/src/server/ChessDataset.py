from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as T

CLASS_NAMES = [
    'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
    'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'
]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

def resize_keep_height(img, target_h=128, target_w=128):
    h, w = img.shape[:2]
    scale = target_h / h
    nh, nw = target_h, int(w * scale)

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    if nw > target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    pad_w = target_w - img.shape[1]
    left = pad_w // 2
    right = pad_w - left

    img = cv2.copyMakeBorder(
        img, 0, 0, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return img

class ChessDataset(Dataset):
    def __init__(
        self,
        data_roots=("dataset/dataset1", "dataset/dataset2"),
        split="train",
        transform=transform,
        return_meta=False,
        extensions=(".png", ".jpg", ".jpeg"),
    ):
        if isinstance(data_roots, (str, os.PathLike)):
            data_roots = (str(data_roots),)

        self.data_roots = list(data_roots)
        self.split = split
        self.transform = transform
        self.return_meta = return_meta
        self.class_names = CLASS_NAMES
        self.extensions = tuple(e.lower() for e in extensions)

        self.samples = ['']  # (full_path, label_idx, class_name, source_root)

        for root in self.data_roots:
            split_root = os.path.join(root, split)

            for idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(split_root, class_name)
                if not os.path.isdir(class_dir):
                    continue

                for fname in sorted(os.listdir(class_dir)):
                    if fname.lower().endswith(self.extensions):
                        full_path = os.path.join(class_dir, fname)
                        self.samples.append((full_path, idx, class_name, root))

        if len(self.samples) == 0:
            print(f'BE CAREFUL: 0 samples found on {split}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, idx, class_name, root = self.samples[index]

        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        img = resize_keep_height(img, target_h=128, target_w=128)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_meta:
            meta = {"path": path, "class_name": class_name, "root": root}
            return img, idx, meta

        return img, idx

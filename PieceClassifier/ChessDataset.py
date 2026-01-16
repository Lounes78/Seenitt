from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as T

CLASS_NAMES = ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']


transform = T.Compose([
    T.ToTensor(),  
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

class ChessDataset(Dataset):
    def __init__(self,
                 data_path:str = 'dataset',
                 split: str = 'train',
                 transform = transform):
        
        self.data_path = os.path.join(data_path,split)
        self.transform = transform

        self.class_names = CLASS_NAMES
        self.dict = {} #it has name of the img: [label,class_name]

        self.samples = [] #all imgs
        

        for idx, class_name in enumerate(CLASS_NAMES):
            path = os.path.join(self.data_path,class_name)
            imgs = sorted([f for f in os.listdir(path) if f.endswith(".jpg")])

            for img in imgs:
                self.samples.append([img,class_name,idx])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path,class_name,idx = self.samples[index]
        path = os.path.join(self.data_path,class_name,img_path)

        img = cv2.imread(path,cv2.IMREAD_UNCHANGED)

        img = self.resize_keep_height(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img,idx
    
    def resize_keep_height(self,img, target_h=128, target_w=128):
        h, w = img.shape[:2]
        scale = target_h / h
        nh, nw = target_h, int(w * scale)

        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        if nw > target_w:
            img = cv2.resize(img, (target_w, target_h))

        pad_w = target_w - img.shape[1]
        left = pad_w // 2
        right = pad_w - left

        img = cv2.copyMakeBorder(
            img, 0, 0, left, right,
            cv2.BORDER_CONSTANT, value=(0,0,0)
        )
        
        return img


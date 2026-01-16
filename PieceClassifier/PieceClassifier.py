from Resnet18 import ResNet18Classifier
from ChessDataset import ChessDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

class PieceClassifier:
    def __init__(self, 
                 model = ResNet18Classifier(),
                 data_path: str = 'dataset',
                 epochs: int = 50,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 optimizer = AdamW,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 patience: int = 5,
                 criterion = torch.nn.CrossEntropyLoss,
                 save_model_path: str = 'models/',
                 load_model_path: str = None,
                 device: str = 'cuda'  
                 ):
        
        self.epochs = epochs
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_model_path = save_model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        

        self.model_instance = model

        self.dataset_train = ChessDataset(self.data_path,split='train')
        self.dataset_val = ChessDataset(self.data_path,split='valid')
        self.dataset_test = ChessDataset(self.data_path,split='test')

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.train_dataloader = DataLoader(self.dataset_train,
                                           batch_size = self.batch_size,
                                           shuffle = True,
                                           num_workers = self.num_workers)
        
        self.val_dataloader = DataLoader(self.dataset_val,
                                         batch_size = self.batch_size,
                                         shuffle = False,
                                         num_workers = self.num_workers)
        
        self.test_dataloader = DataLoader(self.dataset_test,
                                          batch_size = self.batch_size,
                                          shuffle = False,
                                          num_workers = self.num_workers)
        
        self.criterion = criterion()

        self.optimizer = optimizer(
            self.model_instance.parameters(),
            lr=self.lr,
            weight_decay = self.weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=patience,
            min_lr=1e-7,
        )

        self.best_loss = np.inf

        if load_model_path is not None:
            self._load_model(load_model_path)

        self.model_instance = model.to(self.device)

                                        

    
    def train(self):
        model = self.model_instance

        train_losses = []
        val_losses = [] 

        for epoch in range(self.epochs):

            model.train()

            train_running_loss = 0

            for idx, batch in enumerate(tqdm(self.train_dataloader)):
                img, nb_class = batch

                img = img.float().to(self.device)
                nb_class = nb_class.long().to(self.device)

                output = model(img)

                loss = self.criterion(output,nb_class)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_running_loss += loss.item()

            train_loss = train_running_loss / max(1, len(self.train_dataloader))
            self.train_losses.append(train_loss)

            val_loss = self.validate()

            last_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            if last_lr != current_lr:
                print(f"LR now: {current_lr:.2e}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_model(epoch=epoch,val_loss=val_loss)


            print("-" * 30)
            print(f"Epoch {epoch+1}/{self.epochs}")

            print(f"Train loss: {train_loss:.6f}")
            print(f"Val loss: {val_loss:.6f}")
            print("-" * 30)

        print('TRAINING FINISHED')

    @torch.no_grad()
    def validate(self):
        model = self.model_instance
        model.eval()

        val_running_loss = 0

        for batch_idx, batch in enumerate(self.val_dataloader):
            img, nb_class = batch

            img = img.float().to(self.device)
            nb_class = nb_class.long().to(self.device)

            output = model(img)

            loss = self.criterion(output,nb_class)

            val_running_loss += loss.item()

        val_loss = val_running_loss / max(1,len(self.val_dataloader))
        self.val_losses.append(val_loss)

        return val_loss
    
    @torch.no_grad()
    def test(self):
        model = self.model_instance
        model.eval()

        test_running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.test_dataloader):
            img, nb_class = batch

            img = img.float().to(self.device)
            nb_class = nb_class.long().to(self.device)

            output = model(img)
            loss = self.criterion(output,nb_class)
            test_running_loss += loss.item()

            preds = output.argmax(dim=1)
            correct += (preds == nb_class).sum().item()
            total += nb_class.size(0)

        test_loss = test_running_loss / max(1,len(self.val_dataloader))
        acc = correct / max(1, total)

        print(f"Test loss: {test_loss:.6f}")
        print(f"Test accuracy: {acc*100:.2f}%")

        return test_loss, acc
    
    def _save_model(self,epoch,val_loss):
        os.makedirs(self.save_model_path, exist_ok=True)
        path = os.path.join(self.save_model_path, f"best_{val_loss:.4f}.pt")
        torch.save(
            {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "model_state": self.model_instance.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict()
            },
            path,
        )

        print(f"Best model saved (val loss = {val_loss:.4f})")


    def _load_model(self,path):
        checkpoint = torch.load(
            path,
            map_location = self.device 
        )

        self.model_instance.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.best_loss = checkpoint['val_loss']

        print(
            f"Loaded best model from epoch {checkpoint['epoch']} "
            f"(val loss = {checkpoint['val_loss']:.4f})"
        )

    def export_onnx(self,onnx_path='piece_resnet18_128.onnx'):
        model = self.model_instance
        model = model.to(self.device).eval()

        dummy = torch.randn(1,3,128,128,device = self.device)
        
        torch.onnx.export(model,
                          dummy,
                          onnx_path,
                          export_params=True,
                          input_names=["images"],
                          output_names=["logits"],
                          dynamic_axes={
                            "images": {0: "batch"},
                            "logits": {0: "batch"},
                          },
                        )
        
        print('ONNX saved: ',onnx_path)





        

            



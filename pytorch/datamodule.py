from typing import Type, Any

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from torchvision import transforms
from torchvision.datasets import Food101

from torch.utils.data import random_split, DataLoader

config = {'seed': 2021, 
          'trainer': {
              'max_epochs': 100,
              'gpus': 1,
              'accumulate_grad_batches': 1,
              'progress_bar_refresh_rate': 1,
              'fast_dev_run': False,
              'num_sanity_val_steps': 0,
              'resume_from_checkpoint': None,
          },
          'data': {
              'dataset_name': 'Food101',
              'batch_size': 32,
              'num_workers': 4,
              'size': (224, 224),
              'data_root': '/data/food-101',
              'valid_ratio': 0.1
          },
          'model':{
                'backbone_init': {
                    'model': 'efficientnet_v2_m_in21k',
                    'nclass': 0, #don't change
                    'pretrained': True,
                    },
                'optimizer_init':{
                    'class_path': 'torch.optim.SGD',
                    'init_args': {
                        'lr': 0.01,
                        'momentum': 0.95,
                        'weight_decay': 0.0005
                        }
                    },
                'lr_scheduler_init':{
                    'class_path': 'torch.optim.lr_scheduler.CosineAnnealingLR',
                    'init_args':{
                        'T_max': 0 # don't change
                        }
                    }
            }
}

class BaseDataModule(LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 dataset: Type[Any],
                 train_transform: Type[Any],
                 test_transform: Type[Any],
                 batch_size: int = 64,
                 num_workers: int = 4,
                 data_root: str = 'data',
                 valid_ratio: float = 0.1):
        """
        Base Data Module
        :arg
            Dataset: Enter Dataset
            batch_size: Enter batch size
            num_workers: Enter number of workers
            size: Enter resized image
            data_root: Enter root data folder name
            valid_ratio: Enter valid dataset ratio
        """
        super(BaseDataModule, self).__init__()
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.valid_ratio = valid_ratio
        self.num_classes = None
        self.num_step = None
        self.prepare_data()

    def prepare_data(self) -> None:
        train = self.dataset(root=self.data_root, train=True, download=True)
        test = self.dataset(root=self.data_root, train=False, download=True)
        self.num_classes = len(train.classes)
        self.num_step = len(train) // self.batch_size

        print('-' * 50)
        print('* {} dataset class num: {}'.format(self.dataset_name, len(train.classes)))
        print('* {} train dataset len: {}'.format(self.dataset_name, len(train)))
        print('* {} test dataset len: {}'.format(self.dataset_name, len(test)))
        print('-' * 50)

    def setup(self, stage: str = None):
        if stage in (None, 'fit'):
            ds = self.dataset(root=self.data_root, train=True, transform=self.train_transform)
            self.train_ds, self.valid_ds = self.split_train_valid(ds)

        elif stage in (None, 'test', 'predict'):
            self.test_ds = self.dataset(root=self.data_root, train=False, transform=self.test_transform)

    def split_train_valid(self, ds):
        ds_len = len(ds)
        valid_ds_len = int(ds_len * self.valid_ratio)
        train_ds_len = ds_len - valid_ds_len
        return random_split(ds, [train_ds_len, valid_ds_len])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class Food101(BaseDataModule):
    def __init__(self, dataset_name: str, size: tuple, **kwargs):
    
        dataset, mean, std = Food101, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        

        train_transform, test_transform = self.get_trasnforms(mean, std, size)
        super(Food101, self).__init__(dataset_name, dataset, train_transform, test_transform, **kwargs)

    def get_trasnforms(self, mean, std, size):
        train = transforms.Compose([
            transforms.Resize(size),
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train, test
from torchvision import datasets

from PIL import Image

class CIFAR10C(datasets.CIFAR10):
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
        
   def __getitem__(self, index: int):
      img, target = self.data[index], self.targets[index]
      img = Image.fromarray(img)
      xi, xj = None, None
      
      if self.transform is not None:
         xi = self.transform(img)
         xj = self.transform(img)
    
      return xi, xj, target
   
   def __len__(self) -> int:
      return super().__len__()
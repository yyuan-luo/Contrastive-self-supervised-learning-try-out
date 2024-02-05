from torchvision import transforms, datasets


def train_transformations():
   trans = transforms.Compose([
      transforms.RandomResizedCrop(32), # cifar10 original 32*32
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.GaussianBlur(4, (0.1, 1)), #TODO: tune parameter
      transforms.ToTensor(),
      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
   ])
   return trans

def test_transformations():
   trans = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
   ])
   return trans
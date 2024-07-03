import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
import cv2
import torchvision.transforms as transforms


class SegModel(nn.Module):  # todo: move to models
    def __init__(self) -> None:
        super().__init__()
        self.model = deeplabv3_resnet50(pretrained=False, num_classes=3, pretrained_backbone=True)
        ckpt = '/content/checkpoint/segmentator.pt'
        ckpt = torch.load(ckpt, map_location='cpu')['state']
        ckpt = {k: v for k, v in ckpt.items() if k != 'loss.weight'}
        self.load_state_dict(ckpt)
        self.eval().requires_grad_(False)

    def forward(self, x):
        x = self.model(x)['out']
        x = F.softmax(x, dim=1)
        
        background = x[:,0].unsqueeze(1)
        body = x[:,1].unsqueeze(1)
        head = x[:,2].unsqueeze(1)
        return background, body, head

def for_model(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the OpenCV image (numpy array) to a PIL image
        transforms.Resize((256, 256)),  # Resize the image to 256x256
        transforms.ToTensor()  # Convert the PIL image to a PyTorch tensor
    ])
    tensor_image = transform(image)
    return tensor_image.view(-1,3,256,256)
def semantic_loss(Ig,real_image_path,device):
    model = SegModel()
    model = model.to(device)
    I = for_model(real_image_path)
    I=I.to(device)
    Ig=Ig.to(device)
    with torch.no_grad():  
      Sbg, Sbody, Shead = model(Ig)
      SbgI, SbodyI, SheadI = model(I)
    M = 1 - SbodyI
    Limg = torch.norm(M*(Ig-I), p=2) ** 2
    Lhead = torch.norm((SheadI - Shead), p=2) ** 2
    return Limg,Lhead


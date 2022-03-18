#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.facenet import Facenet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Facenet(num_classes = 10575, backbone = "mobilenet").to(device)
    summary(model, (3, 160, 160))

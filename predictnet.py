from net import PAN, ResNet50
import torch
import torch.nn as nn
class PResnet(nn.Module):
    def __init__(self):
        super(PResnet, self).__init__()
        self.model = ResNet50(pretrained=True)
        self.pan = PAN(self.model.blocks[::-1])
        self.fc = nn.Linear(2048,59)
    def forward(self, inputs):
        # global stream
        gf,ga = self.model(inputs)
        ga = self.fc(ga)
        gf1 = self.pan(gf[::-1])
        gaa = torch.cat((ga, gf1), 1)
        return gaa

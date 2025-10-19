import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models  # importa os modelos do torchvision

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # usa o modelo MobileNetV2 pré-treinado
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # remove a última camada (classificador)

        # congela os pesos do modelo base
        for param in self.model.parameters():
            param.requires_grad = False

        # camadas adicionais da sua CNN
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 11)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc4(x)
        return x

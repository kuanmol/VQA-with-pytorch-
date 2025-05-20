import torch
import torch.nn as nn
from torchvision import models

class SimpleVQAModel(nn.Module):
    """
    Concatenates frozen ResNet-18 features with averaged question embeddings.
    """
    def __init__(self, vocab_size, num_answers, embed_dim=128):
        super().__init__()
        # Image encoder: ResNet-18 without final FC
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()
        for p in cnn.parameters():
            p.requires_grad = False
        self.cnn = cnn

        # Question encoder: embedding + mean pooling
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(512 + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_answers)
        )

    def forward(self, images, questions):
        # Image features: [B, 512]
        feat = self.cnn(images)

        # Question features: [B, embed_dim]
        q_emb = self.embed(questions).float()  # [B, L, D]
        q_feat = q_emb.mean(dim=1)

        # Concatenate and classify
        x = torch.cat([feat, q_feat], dim=1)
        return self.classifier(x)
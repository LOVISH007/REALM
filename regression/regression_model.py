import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

class MOSPredictor(nn.Module):
    def __init__(self,x=529,freeze_cnn=False, freeze_bert=False):
        super(MOSPredictor, self).__init__()
        # ResNet-50
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # remove classification head


        # BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
   

        # Fusion + Regression
        self.fc = nn.Sequential(
            nn.Linear(2048 + 768, x),
            nn.ReLU(),
            nn.Linear(x, 1)
        )

        # Optional Freezing
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.cnn(image)  # (B, 2048)
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # (B, 768)
        combined = torch.cat((img_feat, text_feat), dim=1)  # (B, 2816)
        output = self.fc(combined)  # (B, 1)
        return output.squeeze(1)  # (B,)

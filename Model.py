import torch
from torch import nn
from transformers import BertModel

class CategoryModel(nn.Module):
    def __init__(self, model_name="tbs17/MathBERT", hidden_size=768, ffn_size=1024, num_labels=6):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_size, num_labels) 

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits  

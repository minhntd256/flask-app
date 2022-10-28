import torch
import torch.nn as nn
from PIL import Image
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import os

from transformers import BertModel, BertTokenizer, ViTModel, ViTFeatureExtractor
import os

if not os.path.exists('rvt'):
    os.system('git lfs install')
    os.system("git clone https://huggingface.co/minhntd/rvt")

model_vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model_bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

save_path = f'./rvt/saved_model.pth'

class QuestionEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.QEmb = model_bert

    def forward(self, x):
        outputs = self.QEmb(**x)
        last_hidden_states = outputs.last_hidden_state
        features = last_hidden_states
        return features

class ImageEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.IEmb = model_vit

    def forward(self, x):
        outputs = self.IEmb(**x)
        last_hidden_states = outputs.last_hidden_state
        features = last_hidden_states
        return features


class EncoderLayer(nn.Module):
    def __init__(self, nlayers):
        super().__init__()
        encoder_layers = TransformerEncoderLayer(d_model = 768, nhead = 4, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    def forward(self, src):
        output = self.transformer_encoder(src)
        return output

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.IEmb = ImageEmb()
        self.QEmb = QuestionEmb()
        self.SelfAtt = EncoderLayer(nlayers = 1)
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x, y):
        visn_att_output = self.IEmb(x)
        lang_att_output = self.QEmb(y)

        x = visn_att_output[:,0:1,:]
        y = lang_att_output[:,0:1,:]
        
        a = x*y
        
        k = torch.cat((torch.cat((x,a),dim = 1),y), dim = 1)
        k = self.SelfAtt(k)
        k = k[:,0:1,:]
        k = self.fc1(k)
        k = self.fc2(k)
        k = torch.sigmoid(k)
        k = torch.flatten(k)
        return k


model = Network()
m = torch.load(save_path, map_location=torch.device('cpu'))
model.load_state_dict(m)


def model_fit(question, image):
    test_q = tokenizer(question,padding=True, truncation=True,return_tensors='pt')
    test_i = feature_extractor(Image.open(image),return_tensors='pt')
    model.eval()
    with torch.no_grad():
        r = model(test_i, test_q)
    return 1 if r > 0.5 else 0
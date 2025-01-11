import torchvision
import torch.nn as nn
import torch.nn.functional as F

# def load_EchoNet(model_name, pretrained):
#     model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
#     model.fc = torch.nn.Linear(model.fc.in_features, 1)
#     model.fc.bias.data[0] = 55.6
#     return model

class load_EchoNet(nn.Module):
    def __init__(self, model_name, pretrained, add_classfy = True):
        super(load_EchoNet, self).__init__()
        self.add_classfy = add_classfy
        self.encoder = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 256)
        self.regression = nn.Sequential(nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(256,1))
        if add_classfy:
            self.classify = nn.Sequential(nn.Linear(self.encoder.fc.in_features, 256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(256,3))
        # if args.phase == 'pretrain':
            
            
    def forward(self, video):
        fea, out = self.encoder(video)
        pred_gls = self.regression(out)
        if self.add_classfy:
            pred_chamber = self.classify(fea)
            return pred_gls, pred_chamber
        return pred_gls, None
        
        
        
    

if __name__ == '__main__':
    model = load_EchoNet('r2plus1d_18', True)
    print(model)
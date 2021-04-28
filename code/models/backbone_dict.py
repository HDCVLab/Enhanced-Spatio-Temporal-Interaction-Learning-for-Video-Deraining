from models import densenet121, densenet169
from models.resnet import resnet18, resnet34, resnet50


backbone_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'densenet121': densenet121,
    'densenet169': densenet169
}
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import json
import PIL
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument("--image_path", default="assets/hard-leaved pocket orchid.png")
    parser.add_argument("--checkpoint_path", default="mysave/mycheckpoint.pth")
    parser.add_argument("--top_k", default=3)
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--gpu", default=False)
    return parser.parse_args()

def read_img(image_path):
    image = PIL.Image.open(image_path)
    image = image.convert("RGB")
    image_resized = image.resize((244, 244))
    return np.array(image_resized).transpose(2,0,1)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()   
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0)) 
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

def predict(img, model, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img2 = torch.Tensor(img).to(device).unsqueeze(0)
    return model.forward(img2)

def main():
    #get args
    args = get_args()
    device = 'cpu'
    if args.gpu:
        device = 'cuda'
    #read pic
    img = read_img(args.image_path)
    #read model
    model = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.eval()
    #check image
    output = predict(img, model , device)
    #print charts
    ax = plt.subplot(2, 1, 1)
    plt.xticks([])
    plt.yticks([])
    imshow(img, ax)
    values, indexs = output.topk(args.top_k)
    ps = F.softmax(values, dim=1)
    names = []
    for index in indexs.data[0].cpu().numpy():
        names.append(cat_dic.get(str(index),'NaN'))
    ax2 = plt.subplot(2, 1, 2)
    plt.yticks(range(len(names),0,-1), names)
    plt.barh(range(len(names),0,-1), ps.data[0].cpu().numpy())
    plt.show()
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

if __name__ == '__main__':
    main()

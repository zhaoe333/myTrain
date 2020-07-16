import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict

vgg11 = models.vgg11(pretrained=True)
models_dict = {'vgg11': vgg11}

    
def get_args():
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument("dir", default="flowers/")
    parser.add_argument("--save_dir", default="mysave/")
    parser.add_argument("--arch", default="vgg11")
    parser.add_argument("--gpu", default=True)
    parser.add_argument("--learning_rate", default=0.02)
#     parser.add_argument("--hidden_units", default=4096)
    parser.add_argument("--epochs", default=8)
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

def get_dataLoader(dir):
    data_transforms = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.RandomCrop((224,224)),
                                      transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])
    image_datasets = datasets.ImageFolder(dir, transform=data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    return image_datasets, dataloaders


def build_model(model_name, device, learning_rate):
    # TODO: Build and train your network
    model = models_dict[model_name]
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear(4096, 1000)),
                              ('relu3', nn.ReLU()),
                              ('fc4', nn.Linear(1000, 102)),
                              ]))
    model.classifier = classifier
    criterion = nn.CrossEntropyLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    return model, optimizer, criterion

def run_train(model, optimizer, criterion, dataloaders, epochs, device):
    running_loss = 0
    print_every = 5
    steps=0
    for e in range(epochs):
        for ii, (inputs, labels) in enumerate(dataloaders):
            steps+=1
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                              "Loss: {:.4f}".format(running_loss / print_every))
                running_loss = 0

def test_data(model, test_dataloaders,criterion, device):
    model.eval()
    test_loss = 0
    accuracy = 0
    for inputs, labels in test_dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    print(test_loss/len(test_dataloaders))
    print(accuracy/len(test_dataloaders))
                
def main():
    args = get_args()
    device = 'cpu'
    if args.gpu:
        device = 'cuda'
    #read data
    data_dir = args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_datasets, train_dataloaders = get_dataLoader(train_dir)
    test_datasets, test_dataloaders = get_dataLoader(train_dir)
    #build model
    model, optimizer, criterion = build_model(args.arch, device, args.learning_rate)
    #train
    run_train(model, optimizer, criterion, train_dataloaders, args.epochs, device)
    #test
    test_data(model, test_dataloaders, criterion, device)
    #save
    torch.save(model, args.save_dir + 'mycheckpoint.pth')
    

if __name__ == '__main__':
    main()
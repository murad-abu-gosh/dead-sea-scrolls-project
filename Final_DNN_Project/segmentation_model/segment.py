import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Specify a path
PATH = "/home/mihmadkh/PycharmProjects/Final_DNN_Project/segmentation_model/trained_model.pt"
data_path = "/home/mihmadkh/PycharmProjects/Final_DNN_Project/segmentation_model/images/"
transformation = transforms.Compose([
    # transform to tensors
    transforms.ToTensor(),
    # Garyscale the pixels.
    transforms.Grayscale(num_output_channels=3)
    , transforms.Resize((736, 1280))
])
full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transformation)  # 107
train_validation_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1, shuffle=False)

# Load
model = torch.load(PATH)
model.to(device)

# Spread the data across the GPU:
model= torch.nn.DataParallel(model)

model.eval()


for data in (train_validation_loader):
    img, _ = data
    img = img.to(device)
    predict = model(img)
    #plt.subplot(1, 4, 1), ,
    plt.axis('off') 
    plt.imshow(predict.cpu().detach().numpy()[0, 0, :, :].T, cmap='gray')
    
    # im = Image.fromarray(predict.cpu().detach().numpy())
    # im.save('result.jpeg')
    # plt.subplot(1, 4, 2), plt.axis('off'), plt.imshow(test_output[i][1].detach().numpy()[0, 0, :, :].T, cmap='gray')
    # plt.subplot(1, 4, 3), plt.axis('off'), plt.imshow(test_output[i][2].detach().numpy()[0, 0, :, :].T, cmap='gray')
    plt.show()

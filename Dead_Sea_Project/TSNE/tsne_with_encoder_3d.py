import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torch import nn
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm
import plotly.express as px


########################################################
# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()
])

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    # Initialize the model


model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# new_model = new_model
new_model = new_model.to(device)


########################################################################

data = []
folder = 'E:\\PycharmProjects\\Year_4\\Dead_Sea_Project\\extracting pieces\\pieces\\'

# Will contain the feature
features = []
images = []
for filename in tqdm((os.listdir(folder))):
    image = cv2.imread(os.path.join(folder, filename))
    if image is not None:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (45, 45))
        image = transform(image)
        image = image.reshape(1, 3, 448, 448).to(device)
        with torch.no_grad():
            # Extract the feature from the image
            feature = new_model(image)
            # Convert to NumPy Array, Reshape it, and save it to features variable
        features.append(feature.cpu().detach().numpy().reshape(-1))
        # image = image.flatten()
        images.append(folder + filename)
        # data.append([image, folder + filename])
# Convert to NumPy Array
features = np.array(features)


# _, images = zip(*data)
# from sklearn.decomposition import PCA

# features = np.array(features)
# pca = PCA(n_components=2)
# pca.fit(features)
# pca_features = pca.transform(features)
num_images_to_plot = len(images)
#
if len(images) > num_images_to_plot:
    sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
    images = [images[i] for i in sort_order]
    features = [features[i] for i in sort_order]
X = np.array(features)
tsne = TSNE(n_components=3, learning_rate=350, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

tx, ty = tsne[:, 0], tsne[:, 1]
tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
projections = tsne
df = px.data.iris()
fig = px.scatter_3d(
    projections, x=0, y=1, z=2,
)
fig.update_traces(marker_size=8)
fig.show()

# # plot the clusters
# import matplotlib.pyplot
# from matplotlib.pyplot import imshow
#
# width = 4000
# height = 3000
# max_dim = 100
#
# full_image = Image.new('RGBA', (width, height))
# for img, x, y in zip(images, tx, ty):
#     tile = Image.open(img)
#     rs = max(1, tile.width / max_dim, tile.height / max_dim)
#     tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
#     full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))
#
# matplotlib.pyplot.figure(figsize=(16, 12))
# imshow(full_image)
#
# # save json and png
# import os
# import json
#
# full_image.save("data/test_tSNE.png")
#
# tsne_path = "data/test_tSNE.json"
#
# data = [{"path": os.path.abspath(img), "point": [float(x), float(y)]} for img, x, y in zip(images, tx, ty)]
# with open(tsne_path, 'w') as outfile:
#     json.dump(data, outfile)
#
# print("saved t-SNE result to %s" % tsne_path)
#
# # paste into grid
# import rasterfairy
#
# nx = 20
# ny = 20
#
# grid_assignment = rasterfairy.transformPointCloud2D(tsne)
# tile_width = 50
# tile_height = 50
#
# full_width = tile_width * nx
# full_height = tile_height * ny
# aspect_ratio = float(tile_width) / tile_height
#
# grid_image = Image.new('RGBA', (full_width, full_height))
#
# for img, grid_pos in zip(images, grid_assignment[0]):
#     idx_x, idx_y = grid_pos
#     x, y = tile_width * idx_x, tile_height * idx_y
#     tile = Image.open(img)
#     tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
#     if (tile_ar > aspect_ratio):
#         margin = 0.5 * (tile.width - aspect_ratio * tile.height)
#         tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
#     else:
#         margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
#         tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
#     tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
#     grid_image.paste(tile, (int(x), int(y)))
#
# matplotlib.pyplot.figure(figsize=(16, 12))
# grid_image.save('rasta.png')

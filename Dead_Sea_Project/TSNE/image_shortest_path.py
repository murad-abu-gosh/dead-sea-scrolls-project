import random
import numpy as np
import pickle
from PIL import Image
from scipy.spatial import distance
import igraph
import cv2
import os
from skimage import io
from sklearn.feature_extraction import image
from sklearn.manifold import TSNE
from tqdm import tqdm

data = []
folder = 'E:\PycharmProjects\Year_4\Dead_Sea_Project\extracting pieces\pieces\\'

for filename in tqdm((os.listdir(folder))):
    image = cv2.imread(os.path.join(folder, filename))
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (10, 10))
        image = image.flatten()
        data.append([image, folder + filename])

features, images = zip(*data)
from sklearn.decomposition import PCA

features = np.array(features)
pca = PCA(n_components=2)
pca.fit(features)
pca_features = pca.transform(features)
len(pca_features), pca_features

import os

ids = [os.path.basename(os.path.splitext(x)[0]) for x in images]
import os
import json
import pandas as pd

model_path = "data/model.json"

data = [{"id": name, "feature": [pd.Series(feature).to_json(orient='values')]} for name, feature in
        zip(ids, pca_features)]

with open(model_path, 'w') as outfile:
    json.dump(data, outfile)

print("saved model to %s" % model_path)

num_images_to_plot = len(images)

if len(images) > num_images_to_plot:
    sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
    images = [images[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]

kNN = 5

graph = igraph.Graph(directed=True)
graph.add_vertices(len(images))

for i in tqdm(range(len(images))):
    distances = [distance.cosine(pca_features[i], feat) for feat in pca_features]
    idx_kNN = sorted(range(len(distances)), key=lambda k: distances[k])[1:kNN + 1]

    for j in idx_kNN:
        graph.add_edge(i, j, weight=distances[j])


def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = Image.open(images[idx])
        img = img.convert('RGB')
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


idx1 = int(len(images) * random.random())
idx2 = int(len(images) * random.random())


path = graph.get_shortest_paths(idx1, idx2, output='vpath', weights='weight')[0]
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

results_image = get_concatenated_images(path, 200)
plt.figure(figsize=(16, 12))
img = Image.fromarray(results_image, 'RGB')
img.save('data/shortest_path.jpg', 'JPEG', quality=100)

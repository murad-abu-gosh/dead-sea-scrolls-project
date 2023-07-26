import os

from roboflow import Roboflow

rf = Roboflow(api_key="Sev4ggmCixD8gUiorbW5")
project = rf.workspace().project("dead-sea-scrolls-fragments-detection")
model = project.version(2).model

# infer on a local image
# print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())


images_path = r"data/test/images"

for img in os.listdir(images_path):
    # visualize your prediction
    model.predict(images_path + '/' + img, confidence=20, overlap=20).save("eval_results/"+img)

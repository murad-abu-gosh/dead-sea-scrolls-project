from roboflow import Roboflow
rf = Roboflow(api_key="92NkozimSPOQAXUbjVYz")
project = rf.workspace().project("dead-sea-scrolls-text-detection")
model = project.version(1).model

# infer on a local image
# print(model.predict("your_image.jpg", confidence=20, overlap=20).json())

# visualize your prediction
model.predict("img1.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
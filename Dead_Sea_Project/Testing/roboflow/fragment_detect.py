from roboflow import Roboflow
rf = Roboflow(api_key="N8fbv1Q0CyIoyFLrwF8G")
project = rf.workspace().project("dead-sea-scrolls-fragments-detection-nc0sg")
model = project.version(1).model

# infer on a local image
# print(model.predict("M43674.jpg", confidence=20, overlap=10).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

if __name__ == '__main__':
    print(model.predict("M43674.jpg", confidence=20, overlap=10).json())
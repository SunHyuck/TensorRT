import json
import requests
import sys

arg = int(sys.argv[1])

imagenet_classes = requests.get("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json").json()

index_to_label = {int(index): label[1] for index, label in imagenet_classes.items()}

predicted_index = arg

print(f"The predicted class label is: {index_to_label[predicted_index]}")
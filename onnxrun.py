import torch
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

test_transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 128

test = datasets.CIFAR100(root="./", train=False, download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

sess = onnxruntime.InferenceSession("simple_model.onnx", providers=["CUDAExecutionProvider"])

# input_data = preprocess("images/cat.jpeg")

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

correct, all_data = 0,0
for img, label in tqdm(test_loader):
    img = img.numpy()
    label = label.numpy()

    if img.shape[0] != 128:
        continue
    input_dict = {input_name: img}
    output = sess.run([output_name], input_dict)[0]
    
    pred_label = np.argmax(output, axis=1)
    correct += np.sum(pred_label == label)
    all_data += len(label)
print(f"Accuracy: {correct / all_data * 100:.2f}%")
# input_name = sess.get_inputs()[0].name
# input_dict = {input_name: input_data}

# output = sess.run(None, input_dict)

# print("Input data: ")
# print(input)
# print("Output data: ")
# print(output[0])
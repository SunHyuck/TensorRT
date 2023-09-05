import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession("simple_model.onnx")

input_data = np.random.randn(1, 3).astype(np.float32)

input_name = sess.get_inputs()[0].name
input_dict = {input_name: input_data}

output = sess.run(None, input_dict)

print("Input data: ")
print(input_data)
print("Output data: ")
print(output[0])
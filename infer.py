import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from PIL import Image

def preprocess(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = (image_array / 255.0 - 0.5) * 2.0
    return image_array

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
with open("resnet18.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

BATCH_SIZE = 1
files = {}
# input_data = np.random.randn(BATCH_SIZE, 3, 224, 224).astype(np.float32)
input_data = preprocess("images/cat.jpeg")
input_data = np.expand_dims(input_data, axis=0)

d_input = cuda.mem_alloc(BATCH_SIZE*input_data.nbytes)
d_output = cuda.mem_alloc(BATCH_SIZE*1000*4)

cuda.memcpy_htod(d_input, input_data.ravel())

bindings = [int(d_input), int(d_output)]
with engine.create_execution_context() as context:
    if not context.execute_v2(bindings = bindings):
        print("Error in execute")

output_data = np.empty((BATCH_SIZE*1000), dtype=np.float32)
cuda.memcpy_dtoh(output_data.ravel(), d_output)

print("Predicted class index: ", np.argmax(output_data))

d_input.free()
d_output.free()
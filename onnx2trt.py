import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger()

builder = trt.Builder(TRT_LOGGER)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)
config = builder.create_builder_config()
config.max_workspace_size = 1 << 28
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

parser = trt.OnnxParser(network, TRT_LOGGER)

with open("resnet18.onnx", "rb") as model:
    if not parser.parse(model.read()):
        print("Error")
        for error in range(parser.num_errors):
            print(parser.get_error(error))

engine = builder.build_engine(network, config)

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

input_gpu = cuda.mem_alloc(input_data.nbytes)
output_gpu = cuda.mem_alloc(1000*4)

cuda.memcpy_htod(input_gpu, input_data.ravel())

context = engine.create_execution_context()
bindings=[int(input_gpu), int(output_gpu)]
context.execute_v2(bindings)

output_data = np.empty([1, 1000], dtype=np.float32)
cuda.memcpy_dtoh(output_data, output_gpu)

print("Inferred class: ", np.argmax(output_data))
import torch
import torch.onnx
import torchvision.models as models
import tensorrt as trt

model = models.resnet18(pretrained=True)
model.eval()
BATCH_SIZE = 1
x = torch.randn(BATCH_SIZE, 3, 224, 224)
torch.onnx.export(model, x, "resnet18.onnx", verbose=True, opset_version=11)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

builder = trt.Builder(TRT_LOGGER)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)
config = builder.create_builder_config()
config.max_workspace_size = 1 << 28

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

parser = trt.OnnxParser(network, TRT_LOGGER)

with open("resnet18.onnx", "rb") as model_file:
    if not parser.parse(model_file.read()):
        print("Error")
        for error in range(parser.num_errors):
            print(parser.get_error(error))

engine = builder.build_engine(network, config)

if engine is None:
    printf("Error: Engine could not be build.")

else:
    with open("resnet18.trt", "wb") as f:
        f.write(engine.serialize())
    print("Successfully built and saved the engine.")
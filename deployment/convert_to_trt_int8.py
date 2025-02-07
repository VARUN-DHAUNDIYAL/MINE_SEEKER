import tensorrt as trt
import onnx

onnx_model_path = "models/rf_landmine_model.onnx"
trt_engine_path = "models/rf_landmine_model_int8.trt"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open(onnx_model_path, "rb") as model:
    parser.parse(model.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)

engine = builder.build_engine(network, config)

with open(trt_engine_path, "wb") as f:
    f.write(engine.serialize())

print(f"INT8 TensorRT model saved to {trt_engine_path}")

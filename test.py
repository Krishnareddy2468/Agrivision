import torch
from ultralytics import YOLO

# Load the YOLO model (best.pt)
model = YOLO("//Users//krishnareddy//Downloads//GOOGLE DEVS//farmer bot//PlantFruitapp//backend//models//best.pt")

# Set model to evaluation mode
model.model.eval()

# Create a dummy input with shape (1, 3, 640, 640)
dummy_input = torch.randn(1, 3, 640, 640)

# Export the underlying model (model.model) to ONNX
onnx_path = "best.onnx"
torch.onnx.export(
    model.model, 
    dummy_input, 
    onnx_path, 
    export_params=True,
    opset_version=11,  # Use a compatible opset version
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("ONNX export successful:", onnx_path)

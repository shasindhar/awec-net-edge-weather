import os
import torch
from src.config import config
from src.models.awec_net import AWECNet

def export_to_onnx(output_dir: str = "./checkpoints/onnx"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    
    print(f"[+] Initializing AWEC-Net for ONNX Export...")
    model = AWECNet(num_classes=config.NUM_CLASSES).to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # 1. Export Full Adaptive AWEC-Net
    full_path = os.path.join(output_dir, "awec_net_full.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        full_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['adaptive_logits', 'out1', 'out2', 'out3', 'complexity_score', 'routing_weights'],
        dynamic_axes={'input_image': {0: 'batch_size'}}
    )
    print(f"[+] Exported Full AWEC-Net ONNX model to: {full_path}")
    
    # 2. Export Standalone Estimator Sub-graph
    estimator_path = os.path.join(output_dir, "complexity_estimator.onnx")
    torch.onnx.export(
        model.estimator,
        dummy_input,
        estimator_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['complexity_score', 'routing_weights'],
        dynamic_axes={'input_image': {0: 'batch_size'}}
    )
    print(f"[+] Exported Visual Complexity Estimator ONNX to: {estimator_path}")
    
    print("\n[+] ONNX Model Export Completed Successfully!")

if __name__ == "__main__":
    export_to_onnx()

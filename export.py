import os
import torch
from src.config import config
from src.models.awec_net import AWECNet

def export_to_onnx(output_dir: str = "./checkpoints/onnx"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    
    print(f"[+] Initializing AWEC-Net for ONNX Export & INT8 Quantization...")
    model = AWECNet(num_classes=config.NUM_CLASSES).to(device)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "awec_net_best.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[+] Loaded trained checkpoint: {checkpoint_path}")
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # 1. Export Full FP32 ONNX Model
    full_path = os.path.join(output_dir, "awec_net_fp32.onnx")
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
    print(f"[+] Exported Full AWEC-Net FP32 ONNX model to: {full_path}")
    
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
    
    # 3. Apply INT8 Dynamic Quantization via ONNX Runtime
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = os.path.join(output_dir, "awec_net_int8.onnx")
        quantize_dynamic(
            model_input=full_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8
        )
        fp32_size = os.path.getsize(full_path) / 1e6
        int8_size = os.path.getsize(quantized_path) / 1e6
        print(f"[+] Applied Dynamic INT8 Quantization: {quantized_path}")
        print(f"    --> FP32 Model Size: {fp32_size:.2f} MB | INT8 Model Size: {int8_size:.2f} MB ({((fp32_size - int8_size)/fp32_size)*100:.1f}% reduction)")
    except Exception as e:
        print(f"[!] INT8 Quantization step skipped or failed: {e}")
        
    print("\n[+] ONNX Export & INT8 Quantization Workflow Completed Successfully!")

if __name__ == "__main__":
    export_to_onnx()

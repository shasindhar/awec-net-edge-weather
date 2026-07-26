import os
import sys
import subprocess
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from src.config import config
from src.models.awec_net import AWECNet

class AWECNetONNXExportWrapper(nn.Module):
    """
    Clean wrapper for ONNX export & INT8 quantization.
    Returns tensor tuple (logits, complexity_score, routing_weights)
    without internal dictionary flattening shape mismatches.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(x, hard_routing=True)
        return outputs['logits'], outputs['complexity_score'], outputs['routing_weights']

def safe_onnx_export(
    model: nn.Module,
    dummy_input: torch.Tensor,
    export_path: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]]
) -> None:
    """
    Robust ONNX exporter compatible with PyTorch 2.x, Python 3.12, and ONNX Runtime.
    Uses opset 13 to avoid version_converter assertion failures.
    """
    kwargs = {
        "export_params": True,
        "opset_version": 13,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }
    
    try:
        torch.onnx.export(model, dummy_input, export_path, **kwargs)
        return
    except Exception as e:
        print(f"  [!] Standard ONNX export note ({e}). Retrying with legacy TorchScript backend...")
        try:
            torch.onnx.export(model, dummy_input, export_path, dynamo=False, **kwargs)
        except Exception as e2:
            print(f"  [!] Fallback export failed: {e2}")
            torch.onnx.export(model, dummy_input, export_path, **kwargs)

def export_to_onnx(output_dir: str = "./checkpoints/onnx") -> None:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print("[+] Initializing AWEC-Net for ONNX Export & INT8 Quantization...")
    base_model = AWECNet(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        backbone_type="resnet34",
        dropout_rate=config.DROPOUT_RATE
    ).to(device).eval()

    ckpt = os.path.join(config.CHECKPOINT_DIR, "awec_net_best.pth")
    if os.path.exists(ckpt):
        base_model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  [+] Loaded trained checkpoint: {ckpt}")
    else:
        print(f"  [!] Checkpoint not found at {ckpt} — exporting default weights")

    export_model = AWECNetONNXExportWrapper(base_model).to(device).eval()
    dummy = torch.randn(1, 3, 224, 224)

    # ── 1. Full FP32 ONNX ─────────────────────────────────────────────────────
    fp32_path = os.path.join(output_dir, "awec_net_fp32.onnx")
    safe_onnx_export(
        model=export_model,
        dummy_input=dummy,
        export_path=fp32_path,
        input_names=["input_image"],
        output_names=["adaptive_logits", "complexity_score", "routing_weights"],
        dynamic_axes={"input_image": {0: "batch_size"}}
    )
    fp32_mb = os.path.getsize(fp32_path) / 1e6
    print(f"  [+] Exported Full AWEC-Net FP32 ONNX model to: {fp32_path} ({fp32_mb:.2f} MB)")

    # ── 2. Complexity Estimator sub-graph ─────────────────────────────────────
    est_path = os.path.join(output_dir, "complexity_estimator.onnx")
    safe_onnx_export(
        model=base_model.estimator,
        dummy_input=dummy,
        export_path=est_path,
        input_names=["input_image"],
        output_names=["complexity_score", "routing_weights", "gate_logits"],
        dynamic_axes={"input_image": {0: "batch_size"}}
    )
    print(f"  [+] Exported Visual Complexity Estimator ONNX to: {est_path}")

    # ── 3. INT8 Dynamic Quantization via ONNX Runtime ─────────────────────────
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(output_dir, "awec_net_int8.onnx")
        
        try:
            quantize_dynamic(
                model_input=fp32_path,
                model_output=int8_path,
                weight_type=QuantType.QUInt8
            )
        except Exception as q_err:
            print(f"  [!] Primary quantization note: {q_err}. Retrying without strict shape inference...")
            quantize_dynamic(
                model_input=fp32_path,
                model_output=int8_path,
                weight_type=QuantType.QUInt8,
                extra_options={'EnableShapeInference': False}
            )
            
        int8_mb = os.path.getsize(int8_path) / 1e6
        reduction = (fp32_mb - int8_mb) / fp32_mb * 100.0
        print(f"  [+] Applied Dynamic INT8 Quantization: {int8_path}")
        print(f"      FP32 Model Size: {fp32_mb:.2f} MB | INT8 Model Size: {int8_mb:.2f} MB ({reduction:.1f}% reduction)")
    except ImportError:
        print("  [!] onnxruntime not installed — skip INT8 quantization")
    except Exception as e:
        print(f"  [!] INT8 Quantization step skipped or failed: {e}")

    # ── 4. TorchScript Export (for mobile / C++ deployment) ──────────────────
    try:
        ts_path = os.path.join(output_dir, "awec_net_torchscript.pt")
        traced = torch.jit.trace(
            base_model,
            (dummy,),
            strict=False
        )
        traced.save(ts_path)
        ts_mb = os.path.getsize(ts_path) / 1e6
        print(f"  [+] Exported TorchScript model to: {ts_path} ({ts_mb:.2f} MB)")
    except Exception as e:
        print(f"  [!] TorchScript export failed (non-critical): {e}")

    print("\n[+] ONNX Export & INT8 Quantization Workflow Completed Successfully!")
    print(f"    Output directory: {os.path.abspath(output_dir)}")
    print(f"    Generated Artifacts:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        print(f"      {fname:45s}  {os.path.getsize(fpath)/1e6:.2f} MB")

if __name__ == "__main__":
    export_to_onnx()

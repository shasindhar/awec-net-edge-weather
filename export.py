import os
import torch
from src.config import config
from src.models.awec_net import AWECNet

def export_to_onnx(output_dir: str = "./checkpoints/onnx"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print("[+] Loading AWEC-Net for ONNX export…")
    model = AWECNet(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        backbone_type="resnet34",
        dropout_rate=config.DROPOUT_RATE
    ).to(device).eval()

    ckpt = os.path.join(config.CHECKPOINT_DIR, "awec_net_best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  [+] Loaded checkpoint: {ckpt}")
    else:
        print(f"  [!] Checkpoint not found at {ckpt} — exporting untrained weights")

    dummy = torch.randn(1, 3, 224, 224)

    # ── 1. Full FP32 ONNX ─────────────────────────────────────────────────────
    fp32_path = os.path.join(output_dir, "awec_net_fp32.onnx")
    torch.onnx.export(
        model, dummy, fp32_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["adaptive_logits", "out1", "out2", "out3",
                      "complexity_score", "routing_weights", "gate_logits"],
        dynamic_axes={"input_image": {0: "batch_size"}}
    )
    fp32_mb = os.path.getsize(fp32_path) / 1e6
    print(f"  [+] FP32 ONNX → {fp32_path}  ({fp32_mb:.2f} MB)")

    # ── 2. Complexity Estimator sub-graph ─────────────────────────────────────
    est_path = os.path.join(output_dir, "complexity_estimator.onnx")
    torch.onnx.export(
        model.estimator, dummy, est_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["complexity_score", "routing_weights", "gate_logits"],
        dynamic_axes={"input_image": {0: "batch_size"}}
    )
    print(f"  [+] Complexity Estimator ONNX → {est_path}")

    # ── 3. INT8 Dynamic Quantization via ONNX Runtime ─────────────────────────
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(output_dir, "awec_net_int8.onnx")
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QUInt8
        )
        int8_mb = os.path.getsize(int8_path) / 1e6
        reduction = (fp32_mb - int8_mb) / fp32_mb * 100.0
        print(f"  [+] INT8 Quantized ONNX → {int8_path}")
        print(f"      FP32: {fp32_mb:.2f} MB  →  INT8: {int8_mb:.2f} MB  ({reduction:.1f}% smaller)")
    except ImportError:
        print("  [!] onnxruntime-tools not installed — skip INT8 quantization")
        print("       Install with: pip install onnxruntime-tools")
    except Exception as e:
        print(f"  [!] INT8 quantization failed: {e}")

    # ── 4. TorchScript Export (for mobile / C++ deployment) ──────────────────
    try:
        ts_path = os.path.join(output_dir, "awec_net_torchscript.pt")
        traced = torch.jit.trace(
            model,
            (dummy,),
            strict=False
        )
        traced.save(ts_path)
        ts_mb = os.path.getsize(ts_path) / 1e6
        print(f"  [+] TorchScript → {ts_path}  ({ts_mb:.2f} MB)")
    except Exception as e:
        print(f"  [!] TorchScript export failed (non-critical): {e}")

    print("\n[+] ONNX Export Workflow Complete!")
    print(f"    Output directory: {os.path.abspath(output_dir)}")
    print(f"    Files:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        print(f"      {fname:45s}  {os.path.getsize(fpath)/1e6:.2f} MB")

if __name__ == "__main__":
    export_to_onnx()

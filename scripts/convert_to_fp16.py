import onnx
from onnx import TensorProto
from onnxconverter_common import convert_float_to_float16
from pathlib import Path

FP32_MODEL_PATH = Path("models/parking_detector.onnx")
FP16_MODEL_PATH = Path("models/parking_detector_fp16.onnx")

BAD_CAST_NODES = {
    "/model/model.10/Resize_output_cast0",
    "/model/model.13/Resize_output_cast0",
}


def fix_resize_output_casts(model):
    graph = model.graph
    fixed = 0
    for node in graph.node:
        if node.name not in BAD_CAST_NODES:
            continue
        for attr in node.attribute:
            if attr.name == "to":
                old = attr.i
                attr.i = TensorProto.FLOAT16
                print(f"  Fixed {node.name}: to={old} -> {attr.i}")
                fixed += 1
    print(f"  Patched {fixed} Resize Cast node(s)")
    return model


def deduplicate_value_info(model):
    graph = model.graph
    seen = {}
    for vi in graph.value_info:
        name = vi.name
        elem_type = vi.type.tensor_type.elem_type
        if name not in seen:
            seen[name] = vi
        else:
            if elem_type == TensorProto.FLOAT16:
                seen[name] = vi
    original_count = len(graph.value_info)
    del graph.value_info[:]
    graph.value_info.extend(seen.values())
    print(f"  value_info: {original_count} -> {len(graph.value_info)} entries")
    return model


def fix_all_cast_mismatches(model):
    """
    Find every Cast node whose `to` attribute disagrees with the
    registered value_info type of its output tensor, and patch
    the `to` attribute to match value_info (which the converter
    set correctly). This catches /Cast and /Cast_3 input nodes
    as well as any other converter artifacts.
    """
    graph = model.graph

    # Build type map from value_info + graph inputs/outputs
    type_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        type_map[vi.name] = vi.type.tensor_type.elem_type

    fixed = 0
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        out_name = node.output[0]
        registered_type = type_map.get(out_name)
        if registered_type is None:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i != registered_type:
                print(f"  Fixed {node.name or '(unnamed)'}: "
                      f"to={attr.i} -> {registered_type} "
                      f"(output: {out_name})")
                attr.i = registered_type
                fixed += 1

    print(f"  Patched {fixed} Cast node(s) total")
    return model


def convert_to_fp16():
    print(f"Loading FP32 model: {FP32_MODEL_PATH}")
    model_fp32 = onnx.load(str(FP32_MODEL_PATH))

    print("Converting to FP16...")
    model_fp16 = convert_float_to_float16(
        model_fp32,
        keep_io_types=False,
        min_positive_val=1e-7,
        max_finite_val=1e4,
        disable_shape_infer=False,
    )

    print("Patching Resize output Cast nodes...")
    model_fp16 = fix_resize_output_casts(model_fp16)

    print("Deduplicating value_info...")
    model_fp16 = deduplicate_value_info(model_fp16)

    print("Fixing all remaining Cast attribute mismatches...")
    model_fp16 = fix_all_cast_mismatches(model_fp16)

    print("Validating model...")
    onnx.checker.check_model(model_fp16)
    print("  Model is valid ✓")

    onnx.save(model_fp16, str(FP16_MODEL_PATH))
    print(f"Saved: {FP16_MODEL_PATH}")


if __name__ == "__main__":
    convert_to_fp16()
import onnx

model_files = [
    "vision-encoder.onnx",
    "text-encoder.onnx",
    "decoder.onnx"
]

for model_file in model_files:
    print(f"\n--- Inspecting: {model_file} ---")
    try:
        # Load the model (load_external_data=False speeds up loading for inspection)
        model = onnx.load(model_file, load_external_data=False)
        
        print("INPUTS:")
        for input in model.graph.input:
            name = input.name
            
            # Extract shape
            tensor_type = input.type.tensor_type
            shape = []
            if tensor_type.HasField("shape"):
                for d in tensor_type.shape.dim:
                    if d.HasField("dim_value"):
                        shape.append(str(d.dim_value))
                    elif d.HasField("dim_param"):
                        shape.append(f"? ({d.dim_param})") # Dynamic dimension
                    else:
                        shape.append("?")
            
            print(f"  Name: '{name}' | Shape: {shape}")

    except Exception as e:
        print(f"  Error loading {model_file}: {e}")

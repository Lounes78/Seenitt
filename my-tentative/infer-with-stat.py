import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time
import random
from tokenizers import Tokenizer

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    return runtime.deserialize_cuda_engine(engine_data)

def allocate_buffers(engine):
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = list(engine.get_tensor_shape(tensor_name))
        tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
        # Handle dynamic shapes
        if -1 in tensor_shape:
            if 'image' in tensor_name: tensor_shape = [1, 3, 1008, 1008]
            elif 'fpn_feat_0' in tensor_name: tensor_shape = [1, 256, 288, 288]
            elif 'fpn_feat_1' in tensor_name: tensor_shape = [1, 256, 144, 144]
            elif 'fpn_feat_2' in tensor_name or 'fpn_pos_2' in tensor_name: tensor_shape = [1, 256, 72, 72]
            elif 'text_features' in tensor_name or 'prompt_features' in tensor_name: tensor_shape = [1, 32, 256]
            elif 'text_mask' in tensor_name or 'prompt_mask' in tensor_name: tensor_shape = [1, 32]
            elif 'input_ids' in tensor_name or 'attention_mask' in tensor_name: tensor_shape = [1, 32]
            elif 'pred_masks' in tensor_name: tensor_shape = [1, 200, 288, 288]
            elif 'pred_logits' in tensor_name: tensor_shape = [1, 200]
            else: tensor_shape = [1 if x == -1 else x for x in tensor_shape]
        
        size = np.prod(tensor_shape)
        gpu_buffer = cuda.mem_alloc(int(size * np.dtype(tensor_dtype).itemsize))
        cpu_buffer = np.zeros(tensor_shape, dtype=tensor_dtype)
        buffer_dict = {'name': tensor_name, 'gpu': gpu_buffer, 'cpu': cpu_buffer, 'shape': tensor_shape, 'dtype': tensor_dtype}
        
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(buffer_dict)
        else:
            outputs.append(buffer_dict)
    return inputs, outputs

def run_inference(context, inputs, outputs, input_data, stream):
    for inp in inputs:
        if inp['name'] in input_data:
            data = np.ascontiguousarray(input_data[inp['name']].astype(inp['dtype']))
            context.set_input_shape(inp['name'], data.shape)
            cuda.memcpy_htod_async(inp['gpu'], data, stream)
            context.set_tensor_address(inp['name'], int(inp['gpu']))
    for out in outputs:
        context.set_tensor_address(out['name'], int(out['gpu']))
    context.execute_async_v3(stream_handle=stream.handle)
    results = {}
    for out in outputs:
        cuda.memcpy_dtoh_async(out['cpu'], out['gpu'], stream)
        results[out['name']] = out['cpu']
    stream.synchronize()
    return results

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: raise ValueError("Failed to decode image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1008, 1008))
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return img

def tokenize_text(tokenizer, text):
    encoded = tokenizer.encode(text)
    return np.array([encoded.ids], dtype=np.int32), np.array([encoded.attention_mask], dtype=np.int32)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

# 1. SETUP
print("Loading engines...")
vision_engine = load_engine('vision_encoder_fp16.engine')
text_engine = load_engine('text_encoder_fp16.engine')
decoder_engine = load_engine('decoder_fp16.engine')

vision_context = vision_engine.create_execution_context()
text_context = text_engine.create_execution_context()
decoder_context = decoder_engine.create_execution_context()

vision_inputs, vision_outputs = allocate_buffers(vision_engine)
text_inputs, text_outputs = allocate_buffers(text_engine)
decoder_inputs, decoder_outputs = allocate_buffers(decoder_engine)

stream = cuda.Stream()
tokenizer = Tokenizer.from_file('tokenizer.json')
tokenizer.enable_padding(length=32, pad_id=0)
tokenizer.enable_truncation(max_length=32)

# 2. INPUTS
image_path = '/workspace/sam3/seniT/vid1/frame_0333.jpg'
prompts = ['chessboard', 'chess pieces'] 
image_tensor = preprocess_image(image_path)
input_ids, attention_mask = tokenize_text(tokenizer, "warmup")

# =============================================================================
# WARMUP (CRITICAL FIX)
# =============================================================================
print("Warming up GPU...")
# We run everything a few times to initialize CUDA contexts and caches
for _ in range(5):
    # Warmup Vision
    v_res = run_inference(vision_context, vision_inputs, vision_outputs, {'images': image_tensor}, stream)
    # Warmup Text
    t_res = run_inference(text_context, text_inputs, text_outputs, {'input_ids': input_ids, 'attention_mask': attention_mask}, stream)
    # Warmup Decoder
    d_data = {**v_res, 'prompt_features': t_res['text_features'], 'prompt_mask': attention_mask}
    run_inference(decoder_context, decoder_inputs, decoder_outputs, d_data, stream)

stream.synchronize()
print("Warmup complete. Starting Measurement.\n")

# =============================================================================
# MEASURED RUN
# =============================================================================

print(f"Processing '{image_path}'")
print(f"Prompts: {prompts}")

# 3. RUN VISION ENCODER (SHARED)
print("-" * 30)
print("1. Running Vision Encoder (Shared)...")

t_start_vis = time.time()
vision_res = run_inference(vision_context, vision_inputs, vision_outputs, {'images': image_tensor}, stream)
t_end_vis = time.time()
vision_latency = (t_end_vis - t_start_vis) * 1000
print(f"--> Vision Time: {vision_latency:.2f} ms")

# 4. LOOP OVER PROMPTS
original_img = cv2.imread(image_path)
h, w = original_img.shape[:2]
output_img = original_img.copy()

total_text_time = 0
total_dec_time = 0

BOX_THRESHOLD = 0.35
MASK_THRESHOLD = 0.5

print("-" * 30)
print("2. Running Decoder for each prompt...")

for i, prompt in enumerate(prompts):
    # A. Text Encoder
    t_s = time.time()
    input_ids, attention_mask = tokenize_text(tokenizer, prompt)
    text_data = {'input_ids': input_ids, 'attention_mask': attention_mask}
    text_res = run_inference(text_context, text_inputs, text_outputs, text_data, stream)
    t_text = (time.time() - t_s) * 1000
    total_text_time += t_text
    
    # B. Decoder
    t_s = time.time()
    decoder_data = {}
    decoder_data.update(vision_res)
    decoder_data['prompt_features'] = text_res['text_features']
    decoder_data['prompt_mask'] = attention_mask
    
    decoder_results = run_inference(decoder_context, decoder_inputs, decoder_outputs, decoder_data, stream)
    t_dec = (time.time() - t_s) * 1000
    total_dec_time += t_dec
    
    print(f"   Prompt: '{prompt}' | Text: {t_text:.2f} ms | Dec: {t_dec:.2f} ms")
    
    # C. Extract & Draw
    mask_key = [k for k in decoder_results.keys() if 'pred_masks' in k][0]
    score_key = [k for k in decoder_results.keys() if 'pred_logits' in k][0]
    
    pred_masks = decoder_results[mask_key]
    pred_logits = decoder_results[score_key]
    
    # FIX: Clamping to prevent overflow
    flat_logits = np.clip(pred_logits[0], -80, 80)
    pred_scores = 1.0 / (1.0 + np.exp(-flat_logits))
    
    valid_indices = np.where(pred_scores > BOX_THRESHOLD)[0]
    
    # Color logic: Blue for board (0), Green for pieces (1)
    color = (255, 0, 0) if i == 0 else (0, 255, 0)
        
    for idx in valid_indices:
        raw_mask = pred_masks[0, idx]
        
        # FIX: Clamping to prevent overflow in mask sigmoid
        raw_mask = np.clip(raw_mask, -80, 80)
        prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
        
        mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (mask_resized > MASK_THRESHOLD).astype(np.uint8)
        
        if binary_mask.sum() == 0: continue
        
        # Overlay
        colored_overlay = np.zeros_like(original_img)
        colored_overlay[binary_mask == 1] = color
        alpha = 0.5
        mask_indices = binary_mask == 1
        output_img[mask_indices] = cv2.addWeighted(
            output_img[mask_indices], 1 - alpha, 
            colored_overlay[mask_indices], alpha, 0
        )
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_img, contours, -1, color, 2)

cv2.imwrite('output_combined_prompts.jpg', output_img)

# =============================================================================
# SUMMARY
# =============================================================================
print("="*30)
print("TIMING SUMMARY")
print("="*30)
print(f"Vision (Once)     : {vision_latency:.2f} ms")
print(f"Text (Total)      : {total_text_time:.2f} ms")
print(f"Decoder (Total)   : {total_dec_time:.2f} ms")
print("-" * 30)
print(f"Total Model Time  : {vision_latency + total_text_time + total_dec_time:.2f} ms")
print("="*30)
print("Saved result to 'output_combined_prompts.jpg'")














# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import cv2
# import os
# import time
# import random
# from tokenizers import Tokenizer

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def load_engine(engine_path):
#     if not os.path.exists(engine_path):
#         raise FileNotFoundError(f"Engine not found: {engine_path}")
#     logger = trt.Logger(trt.Logger.WARNING)
#     runtime = trt.Runtime(logger)
#     with open(engine_path, 'rb') as f:
#         engine_data = f.read()
#     return runtime.deserialize_cuda_engine(engine_data)

# def allocate_buffers(engine):
#     inputs = []
#     outputs = []
    
#     for i in range(engine.num_io_tensors):
#         tensor_name = engine.get_tensor_name(i)
#         tensor_shape = list(engine.get_tensor_shape(tensor_name))
#         tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
#         # Handle dynamic shapes (-1 dimensions) for Batch Size = 1
#         if -1 in tensor_shape:
#             if 'image' in tensor_name: tensor_shape = [1, 3, 1008, 1008]
#             elif 'fpn_feat_0' in tensor_name: tensor_shape = [1, 256, 288, 288]
#             elif 'fpn_feat_1' in tensor_name: tensor_shape = [1, 256, 144, 144]
#             elif 'fpn_feat_2' in tensor_name or 'fpn_pos_2' in tensor_name: tensor_shape = [1, 256, 72, 72]
#             elif 'text_features' in tensor_name or 'prompt_features' in tensor_name: tensor_shape = [1, 32, 256]
#             elif 'text_mask' in tensor_name or 'prompt_mask' in tensor_name: tensor_shape = [1, 32]
#             elif 'input_ids' in tensor_name or 'attention_mask' in tensor_name: tensor_shape = [1, 32]
#             elif 'pred_masks' in tensor_name: tensor_shape = [1, 200, 288, 288]
#             elif 'pred_boxes' in tensor_name: tensor_shape = [1, 200, 4]
#             elif 'pred_logits' in tensor_name: tensor_shape = [1, 200]
#             elif 'presence_logits' in tensor_name: tensor_shape = [1, 1]
#             else: tensor_shape = [1 if x == -1 else x for x in tensor_shape]
        
#         size = np.prod(tensor_shape)
#         gpu_buffer = cuda.mem_alloc(int(size * np.dtype(tensor_dtype).itemsize))
#         cpu_buffer = np.zeros(tensor_shape, dtype=tensor_dtype)
        
#         buffer_dict = {'name': tensor_name, 'gpu': gpu_buffer, 'cpu': cpu_buffer, 'shape': tensor_shape, 'dtype': tensor_dtype}
        
#         if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
#             inputs.append(buffer_dict)
#         else:
#             outputs.append(buffer_dict)
    
#     return inputs, outputs

# def run_inference(context, inputs, outputs, input_data, stream):
#     for inp in inputs:
#         if inp['name'] in input_data:
#             data = np.ascontiguousarray(input_data[inp['name']].astype(inp['dtype']))
#             context.set_input_shape(inp['name'], data.shape)
#             cuda.memcpy_htod_async(inp['gpu'], data, stream)
#             context.set_tensor_address(inp['name'], int(inp['gpu']))
    
#     for out in outputs:
#         context.set_tensor_address(out['name'], int(out['gpu']))
    
#     context.execute_async_v3(stream_handle=stream.handle)
    
#     results = {}
#     for out in outputs:
#         cuda.memcpy_dtoh_async(out['cpu'], out['gpu'], stream)
#         results[out['name']] = out['cpu']
    
#     stream.synchronize()
#     return results

# def preprocess_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")
#     img = cv2.imread(image_path)
#     if img is None: raise ValueError("Failed to decode image.")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (1008, 1008))
#     img = img.astype(np.float32) / 255.0
#     img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#     img = img.transpose(2, 0, 1)[np.newaxis, ...]
#     return img

# def tokenize_text(tokenizer, text):
#     encoded = tokenizer.encode(text)
#     input_ids = np.array([encoded.ids], dtype=np.int32)
#     attention_mask = np.array([encoded.attention_mask], dtype=np.int32)
#     return input_ids, attention_mask

# # =============================================================================
# # MAIN
# # =============================================================================

# # 1. Load Resources
# print("Loading engines...")
# t_start_load = time.time()

# vision_engine = load_engine('vision_encoder_fp16.engine')
# text_engine = load_engine('text_encoder_fp16.engine')
# decoder_engine = load_engine('decoder_fp16.engine')

# vision_context = vision_engine.create_execution_context()
# text_context = text_engine.create_execution_context()
# decoder_context = decoder_engine.create_execution_context()

# vision_inputs, vision_outputs = allocate_buffers(vision_engine)
# text_inputs, text_outputs = allocate_buffers(text_engine)
# decoder_inputs, decoder_outputs = allocate_buffers(decoder_engine)

# stream = cuda.Stream()
# tokenizer = Tokenizer.from_file('tokenizer.json')
# tokenizer.enable_padding(length=32, pad_id=0)
# tokenizer.enable_truncation(max_length=32)

# t_end_load = time.time()
# print(f"--> Engine Loading Time: {(t_end_load - t_start_load)*1000:.2f} ms")

# # 2. Prepare Data
# image_path = '/workspace/sam3/seniT/board_crops/crop_0550.jpg'
# text_prompt = 'chessboard and chess pieces'

# print(f"Processing '{image_path}' with prompt '{text_prompt}'")

# t_start_pre = time.time()
# image_tensor = preprocess_image(image_path)
# input_ids, attention_mask = tokenize_text(tokenizer, text_prompt)
# t_end_pre = time.time()
# print(f"--> Preprocessing Time: {(t_end_pre - t_start_pre)*1000:.2f} ms")

# # =============================================================================
# # WARMUP
# # =============================================================================
# print("="*30)
# print("WARMING UP GPU (10 Iterations)...")
# print("="*30)

# warmup_text_data = {'input_ids': input_ids, 'attention_mask': attention_mask}

# for i in range(10):
#     w_vision_res = run_inference(vision_context, vision_inputs, vision_outputs, {'images': image_tensor}, stream)
#     w_text_res = run_inference(text_context, text_inputs, text_outputs, warmup_text_data, stream)
    
#     w_decoder_data = {}
#     w_decoder_data.update(w_vision_res)
#     w_decoder_data['prompt_features'] = w_text_res['text_features']
#     w_decoder_data['prompt_mask'] = attention_mask
    
#     run_inference(decoder_context, decoder_inputs, decoder_outputs, w_decoder_data, stream)

# stream.synchronize()
# print("Warmup Done. Starting Timed Run...")
# print("-" * 30)

# # =============================================================================
# # RUN PIPELINE (TIMED)
# # =============================================================================

# print("Running Vision...")
# t_start_vision = time.time() # <--- START TIME
# vision_res = run_inference(vision_context, vision_inputs, vision_outputs, {'images': image_tensor}, stream)
# t_end_vision = time.time() # <--- END TIME
# print(f"--> Vision Encoder Time: {(t_end_vision - t_start_vision)*1000:.2f} ms")

# print("Running Text...")
# t_start_text = time.time() # <--- START TIME
# text_data = {'input_ids': input_ids, 'attention_mask': attention_mask}
# text_res = run_inference(text_context, text_inputs, text_outputs, text_data, stream)
# t_end_text = time.time() # <--- END TIME
# print(f"--> Text Encoder Time: {(t_end_text - t_start_text)*1000:.2f} ms")

# print("Running Decoder...")
# decoder_data = {}
# decoder_data.update(vision_res)
# decoder_data['prompt_features'] = text_res['text_features']
# decoder_data['prompt_mask'] = attention_mask

# t_start_decoder = time.time() # <--- START TIME
# decoder_results = run_inference(decoder_context, decoder_inputs, decoder_outputs, decoder_data, stream)
# t_end_decoder = time.time() # <--- END TIME
# print(f"--> Decoder Time: {(t_end_decoder - t_start_decoder)*1000:.2f} ms")

# # =============================================================================
# # POST-PROCESS AND VISUALIZE (MULTI-MASK UPDATE)
# # =============================================================================
# print("Processing results...")
# t_start_post = time.time()

# # 1. Configuration
# BOX_THRESHOLD = 0.35  # Confidence threshold (0.0 to 1.0)
# MASK_THRESHOLD = 0.5  # Pixel threshold

# # 2. Extract Outputs
# mask_key = [k for k in decoder_results.keys() if 'pred_masks' in k][0]
# score_key = [k for k in decoder_results.keys() if 'pred_logits' in k][0]

# pred_masks = decoder_results[mask_key]   # Shape: [1, 200, 288, 288]
# pred_logits = decoder_results[score_key] # Shape: [1, 200]

# # 3. Convert Logits to Probabilities (Sigmoid)
# flat_logits = pred_logits[0]
# pred_scores = 1.0 / (1.0 + np.exp(-flat_logits))

# # 4. Filter by Threshold
# valid_indices = np.where(pred_scores > BOX_THRESHOLD)[0]
# print(f"Found {len(valid_indices)} masks with score > {BOX_THRESHOLD}")

# # 5. Visualize All Valid Masks
# original_img = cv2.imread(image_path)
# h, w = original_img.shape[:2]
# output_img = original_img.copy()

# for idx in valid_indices:
#     score = pred_scores[idx]
#     raw_mask = pred_masks[0, idx]
    
#     # Sigmoid on mask to get 0-1 range
#     prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
    
#     # Resize and threshold
#     mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
#     binary_mask = (mask_resized > MASK_THRESHOLD).astype(np.uint8)
    
#     if binary_mask.sum() == 0: continue

#     # Random Color
#     color = [random.randint(0, 255) for _ in range(3)]
    
#     # Draw Mask Overlay
#     colored_overlay = np.zeros_like(original_img)
#     colored_overlay[binary_mask == 1] = color
    
#     alpha = 0.5
#     mask_indices = binary_mask == 1
#     output_img[mask_indices] = cv2.addWeighted(
#         output_img[mask_indices], 1 - alpha, 
#         colored_overlay[mask_indices], alpha, 0
#     )
    
#     # Draw Contours
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(output_img, contours, -1, color, 2)
    
#     print(f" - Drawn mask {idx} (Score: {score:.4f})")

# cv2.imwrite('output_fixed_multi.jpg', output_img)

# t_end_post = time.time()
# print(f"--> Post-Processing Time: {(t_end_post - t_start_post)*1000:.2f} ms")

# # =============================================================================
# # TIMING SUMMARY
# # =============================================================================
# total_time = (t_end_vision - t_start_vision) + \
#              (t_end_text - t_start_text) + \
#              (t_end_decoder - t_start_decoder)

# print("="*30)
# print("TIMING SUMMARY")
# print("="*30)
# print(f"Preprocessing : {(t_end_pre - t_start_pre)*1000:.2f} ms")
# print(f"Vision Encoder: {(t_end_vision - t_start_vision)*1000:.2f} ms")
# print(f"Text Encoder  : {(t_end_text - t_start_text)*1000:.2f} ms")
# print(f"Decoder       : {(t_end_decoder - t_start_decoder)*1000:.2f} ms")
# print(f"Post-Process  : {(t_end_post - t_start_post)*1000:.2f} ms")
# print("-" * 30)
# print(f"Total Inference (Model Only): {total_time*1000:.2f} ms")
# print("="*30)


# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import cv2
# import os
# import time
# from tokenizers import Tokenizer

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def load_engine(engine_path):
#     """Load TensorRT engine from file"""
#     if not os.path.exists(engine_path):
#         raise FileNotFoundError(f"Engine not found: {engine_path}")
        
#     logger = trt.Logger(trt.Logger.WARNING)
#     runtime = trt.Runtime(logger)
#     with open(engine_path, 'rb') as f:
#         engine_data = f.read()
#     engine = runtime.deserialize_cuda_engine(engine_data)
#     return engine

# def allocate_buffers(engine):
#     """Pre-allocate GPU memory for exactly Batch Size = 1"""
#     inputs = []
#     outputs = []
    
#     for i in range(engine.num_io_tensors):
#         tensor_name = engine.get_tensor_name(i)
#         tensor_shape = list(engine.get_tensor_shape(tensor_name))
#         tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
#         # Handle dynamic shapes (-1 dimensions) for Batch Size = 1
#         if -1 in tensor_shape:
#             if 'image' in tensor_name:
#                 tensor_shape = [1, 3, 1008, 1008]
#             elif 'fpn_feat_0' in tensor_name:
#                 tensor_shape = [1, 256, 288, 288]
#             elif 'fpn_feat_1' in tensor_name:
#                 tensor_shape = [1, 256, 144, 144]
#             elif 'fpn_feat_2' in tensor_name:
#                 tensor_shape = [1, 256, 72, 72]
#             elif 'fpn_pos_2' in tensor_name:
#                 tensor_shape = [1, 256, 72, 72]
#             elif 'text_features' in tensor_name or 'prompt_features' in tensor_name:
#                 tensor_shape = [1, 32, 256]
#             elif 'text_mask' in tensor_name or 'prompt_mask' in tensor_name:
#                  tensor_shape = [1, 32]
#             elif 'input_ids' in tensor_name or 'attention_mask' in tensor_name:
#                 tensor_shape = [1, 32]
#             elif 'pred_masks' in tensor_name:
#                 tensor_shape = [1, 200, 288, 288]
#             elif 'pred_boxes' in tensor_name:
#                 tensor_shape = [1, 200, 4]
#             elif 'pred_logits' in tensor_name:
#                 tensor_shape = [1, 200]
#             elif 'presence_logits' in tensor_name:
#                 tensor_shape = [1, 1]
#             else:
#                 tensor_shape = [1 if x == -1 else x for x in tensor_shape]
        
#         size = np.prod(tensor_shape)
#         gpu_buffer = cuda.mem_alloc(int(size * np.dtype(tensor_dtype).itemsize))
#         cpu_buffer = np.zeros(tensor_shape, dtype=tensor_dtype)
        
#         buffer_dict = {
#             'name': tensor_name,
#             'gpu': gpu_buffer,
#             'cpu': cpu_buffer,
#             'shape': tensor_shape,
#             'dtype': tensor_dtype
#         }
        
#         if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
#             inputs.append(buffer_dict)
#         else:
#             outputs.append(buffer_dict)
    
#     return inputs, outputs

# def run_inference(engine, context, inputs, outputs, input_data, stream):
#     """Execute inference"""
#     # Step 1: Copy input data from CPU to GPU
#     for inp in inputs:
#         if inp['name'] in input_data:
#             data = input_data[inp['name']]
#             data = np.ascontiguousarray(data.astype(inp['dtype']))
#             context.set_input_shape(inp['name'], data.shape)
#             cuda.memcpy_htod_async(inp['gpu'], data, stream)
#             context.set_tensor_address(inp['name'], int(inp['gpu']))
    
#     # Step 2: Set output addresses
#     for out in outputs:
#         context.set_tensor_address(out['name'], int(out['gpu']))
    
#     # Step 3: Execute
#     context.execute_async_v3(stream_handle=stream.handle)
    
#     # Step 4: Copy results back
#     results = {}
#     for out in outputs:
#         cuda.memcpy_dtoh_async(out['cpu'], out['gpu'], stream)
#         results[out['name']] = out['cpu']
    
#     # Wait for GPU to finish (CRITICAL for timing)
#     stream.synchronize()
#     return results

# # =============================================================================
# # PREPROCESSING
# # =============================================================================

# def preprocess_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     img = cv2.imread(image_path)
#     if img is None:
#          raise ValueError("Failed to decode image.")

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (1008, 1008))
#     img = img.astype(np.float32) / 255.0
#     img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#     img = img.transpose(2, 0, 1)
#     img = img[np.newaxis, ...]
#     return img

# def tokenize_text(tokenizer, text):
#     encoded = tokenizer.encode(text)
#     input_ids = np.array([encoded.ids], dtype=np.int32)
#     attention_mask = np.array([encoded.attention_mask], dtype=np.int32)
#     return input_ids, attention_mask

# # =============================================================================
# # MAIN
# # =============================================================================

# # 1. Load Resources
# print("Loading engines...")
# t_start_load = time.time()

# vision_engine = load_engine('vision_encoder_fp16.engine')
# text_engine = load_engine('text_encoder_fp16.engine')
# decoder_engine = load_engine('decoder_fp16.engine')

# vision_context = vision_engine.create_execution_context()
# text_context = text_engine.create_execution_context()
# decoder_context = decoder_engine.create_execution_context()

# vision_inputs, vision_outputs = allocate_buffers(vision_engine)
# text_inputs, text_outputs = allocate_buffers(text_engine)
# decoder_inputs, decoder_outputs = allocate_buffers(decoder_engine)

# stream = cuda.Stream()
# tokenizer = Tokenizer.from_file('tokenizer.json')
# tokenizer.enable_padding(length=32, pad_id=0)
# tokenizer.enable_truncation(max_length=32)

# t_end_load = time.time()
# print(f"--> Engine Loading Time: {(t_end_load - t_start_load)*1000:.2f} ms")

# # 2. Prepare Data
# image_path = '/workspace/sam3/seniT/board_crops/crop_0550.jpg'
# text_prompt = 'chessboard and chess pieces'

# print(f"Processing '{image_path}' with prompt '{text_prompt}'")

# t_start_pre = time.time()
# image_tensor = preprocess_image(image_path)
# input_ids, attention_mask = tokenize_text(tokenizer, text_prompt)
# t_end_pre = time.time()
# print(f"--> Preprocessing Time: {(t_end_pre - t_start_pre)*1000:.2f} ms")

# # =============================================================================
# # WARMUP (NEW SECTION)
# # =============================================================================
# print("="*30)
# print("WARMING UP GPU (10 Iterations)...")
# print("="*30)

# # We use the actual inputs for warmup, it doesn't matter what the data is
# # as long as the shapes are correct.
# warmup_text_data = {'input_ids': input_ids, 'attention_mask': attention_mask}

# for i in range(10):
#     # Run Vision
#     w_vision_res = run_inference(vision_engine, vision_context, vision_inputs, vision_outputs, {'images': image_tensor}, stream)
    
#     # Run Text
#     w_text_res = run_inference(text_engine, text_context, text_inputs, text_outputs, warmup_text_data, stream)
    
#     # Run Decoder
#     w_decoder_data = {}
#     w_decoder_data.update(w_vision_res)
#     w_decoder_data['prompt_features'] = w_text_res['text_features']
#     w_decoder_data['prompt_mask'] = attention_mask
    
#     run_inference(decoder_engine, decoder_context, decoder_inputs, decoder_outputs, w_decoder_data, stream)

# # Ensure GPU is finished with warmup
# stream.synchronize()
# print("Warmup Done. Starting Timed Run...")
# print("-" * 30)


# # =============================================================================
# # RUN PIPELINE (TIMED)
# # =============================================================================

# print("Running Vision...")
# t_start_vision = time.time() # <--- START TIME
# vision_res = run_inference(vision_engine, vision_context, vision_inputs, vision_outputs, {'images': image_tensor}, stream)
# t_end_vision = time.time() # <--- END TIME
# print(f"--> Vision Encoder Time: {(t_end_vision - t_start_vision)*1000:.2f} ms")

# print("Running Text...")
# t_start_text = time.time() # <--- START TIME
# text_data = {'input_ids': input_ids, 'attention_mask': attention_mask}
# text_res = run_inference(text_engine, text_context, text_inputs, text_outputs, text_data, stream)
# t_end_text = time.time() # <--- END TIME
# print(f"--> Text Encoder Time: {(t_end_text - t_start_text)*1000:.2f} ms")

# print("Running Decoder...")
# # Prepare inputs
# decoder_data = {}
# decoder_data.update(vision_res)
# decoder_data['prompt_features'] = text_res['text_features']
# decoder_data['prompt_mask'] = attention_mask

# t_start_decoder = time.time() # <--- START TIME
# decoder_results = run_inference(decoder_engine, decoder_context, decoder_inputs, decoder_outputs, decoder_data, stream)
# t_end_decoder = time.time() # <--- END TIME
# print(f"--> Decoder Time: {(t_end_decoder - t_start_decoder)*1000:.2f} ms")


# # =============================================================================
# # POST-PROCESS AND VISUALIZE
# # =============================================================================

# print("Processing results...")
# t_start_post = time.time()

# # 1. Get the Raw Outputs
# mask_key = [k for k in decoder_results.keys() if 'pred_masks' in k][0]
# score_key = [k for k in decoder_results.keys() if 'pred_logits' in k][0]

# pred_masks = decoder_results[mask_key]   # Shape: [1, 200, 288, 288]
# pred_scores = decoder_results[score_key] # Shape: [1, 200]

# # 2. Find the Best Mask
# flat_scores = pred_scores[0]
# best_idx = np.argmax(flat_scores)
# best_score = flat_scores[best_idx]

# print(f"Best detection found at Index {best_idx} with Score {best_score:.4f}")

# # 3. Extract that specific mask
# raw_mask = pred_masks[0, best_idx]

# # 4. Apply Sigmoid
# if raw_mask.max() > 1.0 or raw_mask.min() < 0.0:
#     prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
# else:
#     prob_mask = raw_mask

# # 5. Resize to Original Image
# original_img = cv2.imread(image_path)
# h, w = original_img.shape[:2]
# mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
# binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

# # 6. Visualization
# colored_mask = np.zeros_like(original_img)
# colored_mask[:, :] = (0, 255, 0) # Green color

# alpha = 0.5
# mask_indices = binary_mask > 0
# output_img = original_img.copy()
# output_img[mask_indices] = cv2.addWeighted(original_img[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)

# contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)

# cv2.imwrite('output_fixed.jpg', output_img)

# t_end_post = time.time()
# print(f"--> Post-Processing Time: {(t_end_post - t_start_post)*1000:.2f} ms")

# # =============================================================================
# # SUMMARY
# # =============================================================================
# total_time = (t_end_vision - t_start_vision) + \
#              (t_end_text - t_start_text) + \
#              (t_end_decoder - t_start_decoder)

# print("="*30)
# print("TIMING SUMMARY")
# print("="*30)
# print(f"Preprocessing : {(t_end_pre - t_start_pre)*1000:.2f} ms")
# print(f"Vision Encoder: {(t_end_vision - t_start_vision)*1000:.2f} ms")
# print(f"Text Encoder  : {(t_end_text - t_start_text)*1000:.2f} ms")
# print(f"Decoder       : {(t_end_decoder - t_start_decoder)*1000:.2f} ms")
# print(f"Post-Process  : {(t_end_post - t_start_post)*1000:.2f} ms")
# print("-" * 30)
# print(f"Total Inference (Model Only): {total_time*1000:.2f} ms")
# print("="*30)
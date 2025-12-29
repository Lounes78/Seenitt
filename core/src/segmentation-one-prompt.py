import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
from tokenizers import Tokenizer

class ChessboardSegmenter:
    def __init__(self, assets_dir, prompt="chessboard"):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.stream = cuda.Stream()
        
        # Paths
        vision_path = os.path.join(assets_dir, 'vision_encoder_fp16.engine')
        text_path = os.path.join(assets_dir, 'text_encoder_fp16.engine')
        decoder_path = os.path.join(assets_dir, 'decoder_fp16.engine')
        tokenizer_path = os.path.join(assets_dir, 'tokenizer.json')

        # 1. Load Tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=32, pad_id=0)
        self.tokenizer.enable_truncation(max_length=32)

        # 2. Load Engines
        self.vision_engine = self._load_engine(vision_path)
        self.text_engine = self._load_engine(text_path)
        self.decoder_engine = self._load_engine(decoder_path)

        # 3. Create Contexts
        self.vision_ctx = self.vision_engine.create_execution_context()
        self.text_ctx = self.text_engine.create_execution_context()
        self.decoder_ctx = self.decoder_engine.create_execution_context()

        # 4. Allocate Pinned Memory
        self.vision_io = self._allocate_buffers(self.vision_engine)
        self.text_io = self._allocate_buffers(self.text_engine)
        self.decoder_io = self._allocate_buffers(self.decoder_engine)
        
        # 5. Pre-compute Text Embeddings
        print(f"Caching text embeddings for prompt: '{prompt}'...")
        encoded = self.tokenizer.encode(prompt)
        input_ids = np.array([encoded.ids], dtype=np.int32)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int32)
        
        # We manually run the inference setup for the caching step to ensure shapes are set
        # 'input_ids' and 'attention_mask' are usually the first two inputs
        text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # Run inference properly (this populates self.text_io outputs on GPU)
        self._run_inference(self.text_ctx, self.text_io, text_inputs)
        
        # Store pointers/data for the decoder step
        self.cached_text_features = self.text_io['outputs'][0]['cpu'] # Keep CPU copy if needed, or GPU ptr
        self.cached_attention_mask = attention_mask

    def _load_engine(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self, engine):
        inputs = []
        outputs = []
        
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            # Retrieve shape and dtype
            tensor_shape = list(engine.get_tensor_shape(tensor_name))
            tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
            # Helper to map names to explicit shapes for dynamic dims (-1)
            if -1 in tensor_shape:
                if 'image' in tensor_name: tensor_shape = [1, 3, 1008, 1008]
                elif 'fpn_feat_0' in tensor_name: tensor_shape = [1, 256, 288, 288]
                elif 'fpn_feat_1' in tensor_name: tensor_shape = [1, 256, 144, 144]
                elif 'fpn_feat_2' in tensor_name: tensor_shape = [1, 256, 72, 72]
                elif 'fpn_pos_2' in tensor_name: tensor_shape = [1, 256, 72, 72]
                elif 'text_features' in tensor_name or 'prompt_features' in tensor_name: tensor_shape = [1, 32, 256]
                elif 'text_mask' in tensor_name or 'prompt_mask' in tensor_name: tensor_shape = [1, 32]
                elif 'input_ids' in tensor_name or 'attention_mask' in tensor_name: tensor_shape = [1, 32]
                elif 'pred_masks' in tensor_name: tensor_shape = [1, 200, 288, 288]
                elif 'pred_boxes' in tensor_name: tensor_shape = [1, 200, 4]
                elif 'pred_logits' in tensor_name: tensor_shape = [1, 200]
                elif 'presence_logits' in tensor_name: tensor_shape = [1, 1]
                else: tensor_shape = [1 if x == -1 else x for x in tensor_shape]
            
            # Allocations
            cpu_buffer = cuda.pagelocked_empty(tensor_shape, dtype=tensor_dtype)
            gpu_buffer = cuda.mem_alloc(cpu_buffer.nbytes)
            
            buffer_dict = {
                'name': tensor_name,
                'gpu': gpu_buffer,
                'cpu': cpu_buffer,
                'shape': tensor_shape,
                'dtype': tensor_dtype
            }
            
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(buffer_dict)
            else:
                outputs.append(buffer_dict)
        
        return {'inputs': inputs, 'outputs': outputs}

    def _run_inference(self, context, io_dict, input_data):
        """
        Executes inference. 
        input_data: Dict of {tensor_name: numpy_array}
        """
        # 1. Process Inputs (Copy + Set Shape)
        for inp in io_dict['inputs']:
            name = inp['name']
            
            if name in input_data:
                # Retrieve data
                data = input_data[name]
                
                # CRITICAL FIX: Explicitly cast data to the engine's expected dtype
                # This fixes the TypeError (int32 -> bool)
                if data.dtype != inp['dtype']:
                    data = data.astype(inp['dtype'])
                
                # Check shape compatibility
                # Flatten/Reshape to match buffer
                np.copyto(inp['cpu'], data.ravel().reshape(inp['shape']))
                
                # Copy to GPU
                cuda.memcpy_htod_async(inp['gpu'], inp['cpu'], self.stream)
                
                # CRITICAL FIX: Always set input shape for TRT
                context.set_input_shape(name, inp['shape'])
            else:
                # If data isn't passed but buffer exists (e.g., cached or chained outputs),
                # we MUST still set the input shape based on the buffer's allocation shape
                context.set_input_shape(name, inp['shape'])
            
            # Bind address
            context.set_tensor_address(name, int(inp['gpu']))
        
        # 2. Set Output Addresses
        for out in io_dict['outputs']:
            context.set_tensor_address(out['name'], int(out['gpu']))
        
        # 3. Execute
        if not context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError(f"TensorRT execution failed for tensors: {[x['name'] for x in io_dict['inputs']]}")
        
        # 4. Copy Outputs Back
        results = {}
        for out in io_dict['outputs']:
            cuda.memcpy_dtoh_async(out['cpu'], out['gpu'], self.stream)
            results[out['name']] = out['cpu']
        
        self.stream.synchronize()
        return results

    def _preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # IMREAD_COLOR is faster than default because it skips alpha checks
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
             raise ValueError("Failed to decode image.")
        
        original_img = img # Reference for return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1008, 1008))
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
        return img, original_img

    def predict(self, image_path):
        # 1. Prepare Image
        image_tensor, original_img = self._preprocess_image(image_path)
        
        # 2. Vision Inference
        vision_res = self._run_inference(
            self.vision_ctx, self.vision_io, {'images': image_tensor}
        )

        # 3. Decoder Inference
        decoder_input_data = {}
        decoder_input_data.update(vision_res)
        
        # Add Cached Text Features
        # The text features are already in the CPU buffer of self.text_io['outputs']
        # We need to grab them to feed into the decoder
        text_features_buffer = self.text_io['outputs'][0]['cpu'] # Assuming 0 is features
        
        # Note: We must ensure we grab the right output tensor name for text features
        # Usually it is 'text_features' or 'prompt_features'
        # To be safe, we look it up by name in the io_dict outputs
        
        # We pass the cached CPU data. The _run_inference loop handles the copy to GPU/Casting.
        decoder_input_data['prompt_features'] = text_features_buffer
        decoder_input_data['prompt_mask'] = self.cached_attention_mask
        
        decoder_res = self._run_inference(
            self.decoder_ctx, self.decoder_io, decoder_input_data
        )

        # 4. Post-Processing
        # Dynamically find keys just in case
        mask_key = next(k for k in decoder_res.keys() if 'pred_masks' in k)
        score_key = next(k for k in decoder_res.keys() if 'pred_logits' in k)

        pred_masks = decoder_res[mask_key]
        pred_scores = decoder_res[score_key]

        best_idx = np.argmax(pred_scores[0])
        best_score = float(pred_scores[0][best_idx]) 
        raw_mask = pred_masks[0, best_idx]

        if raw_mask.max() > 1.0 or raw_mask.min() < 0.0:
            prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
        else:
            prob_mask = raw_mask

        h, w = original_img.shape[:2]
        # Nearest neighbor resize is faster for masks
        mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        return binary_mask, original_img, best_score

# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import cv2
# import os
# from tokenizers import Tokenizer

# class ChessboardSegmenter:
#     def __init__(self, assets_dir):
#         """
#         Initialize TensorRT engines, contexts, and memory buffers.
#         """
#         self.logger = trt.Logger(trt.Logger.WARNING)
#         self.runtime = trt.Runtime(self.logger)
#         self.stream = cuda.Stream()
        
#         # Paths
#         vision_path = os.path.join(assets_dir, 'vision_encoder_fp16.engine')
#         text_path = os.path.join(assets_dir, 'text_encoder_fp16.engine')
#         decoder_path = os.path.join(assets_dir, 'decoder_fp16.engine')
#         tokenizer_path = os.path.join(assets_dir, 'tokenizer.json')

#         # 1. Load Tokenizer
#         if not os.path.exists(tokenizer_path):
#             raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
#         self.tokenizer = Tokenizer.from_file(tokenizer_path)
#         self.tokenizer.enable_padding(length=32, pad_id=0)
#         self.tokenizer.enable_truncation(max_length=32)

#         # 2. Load Engines & Create Contexts
#         self.vision_engine = self._load_engine(vision_path)
#         self.text_engine = self._load_engine(text_path)
#         self.decoder_engine = self._load_engine(decoder_path)

#         self.vision_ctx = self.vision_engine.create_execution_context()
#         self.text_ctx = self.text_engine.create_execution_context()
#         self.decoder_ctx = self.decoder_engine.create_execution_context()

#         # 3. Allocate Buffers
#         self.vision_io = self._allocate_buffers(self.vision_engine)
#         self.text_io = self._allocate_buffers(self.text_engine)
#         self.decoder_io = self._allocate_buffers(self.decoder_engine)

#     def _load_engine(self, engine_path):
#         if not os.path.exists(engine_path):
#             raise FileNotFoundError(f"Engine not found: {engine_path}")
#         with open(engine_path, 'rb') as f:
#             engine_data = f.read()
#         return self.runtime.deserialize_cuda_engine(engine_data)

#     def _allocate_buffers(self, engine):
#         inputs = []
#         outputs = []
        
#         for i in range(engine.num_io_tensors):
#             tensor_name = engine.get_tensor_name(i)
#             tensor_shape = list(engine.get_tensor_shape(tensor_name))
#             tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
#             # Handle dynamic shapes (-1) for Batch Size = 1
#             if -1 in tensor_shape:
#                 if 'image' in tensor_name:
#                     tensor_shape = [1, 3, 1008, 1008]
#                 elif 'fpn_feat_0' in tensor_name:
#                     tensor_shape = [1, 256, 288, 288]
#                 elif 'fpn_feat_1' in tensor_name:
#                     tensor_shape = [1, 256, 144, 144]
#                 elif 'fpn_feat_2' in tensor_name:
#                     tensor_shape = [1, 256, 72, 72]
#                 elif 'fpn_pos_2' in tensor_name:
#                     tensor_shape = [1, 256, 72, 72]
#                 elif 'text_features' in tensor_name or 'prompt_features' in tensor_name:
#                     tensor_shape = [1, 32, 256]
#                 elif 'text_mask' in tensor_name or 'prompt_mask' in tensor_name:
#                      tensor_shape = [1, 32]
#                 elif 'input_ids' in tensor_name or 'attention_mask' in tensor_name:
#                     tensor_shape = [1, 32]
#                 elif 'pred_masks' in tensor_name:
#                     tensor_shape = [1, 200, 288, 288]
#                 elif 'pred_boxes' in tensor_name:
#                     tensor_shape = [1, 200, 4]
#                 elif 'pred_logits' in tensor_name:
#                     tensor_shape = [1, 200]
#                 elif 'presence_logits' in tensor_name:
#                     tensor_shape = [1, 1]
#                 else:
#                     tensor_shape = [1 if x == -1 else x for x in tensor_shape]
            
#             size = np.prod(tensor_shape)
#             gpu_buffer = cuda.mem_alloc(int(size * np.dtype(tensor_dtype).itemsize))
#             cpu_buffer = np.zeros(tensor_shape, dtype=tensor_dtype)
            
#             buffer_dict = {
#                 'name': tensor_name,
#                 'gpu': gpu_buffer,
#                 'cpu': cpu_buffer,
#                 'shape': tensor_shape,
#                 'dtype': tensor_dtype
#             }
            
#             if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
#                 inputs.append(buffer_dict)
#             else:
#                 outputs.append(buffer_dict)
        
#         return {'inputs': inputs, 'outputs': outputs}

#     def _run_inference(self, context, io_dict, input_data):
#         # 1. Copy input to GPU
#         for inp in io_dict['inputs']:
#             if inp['name'] in input_data:
#                 data = input_data[inp['name']]
#                 data = np.ascontiguousarray(data.astype(inp['dtype']))
#                 context.set_input_shape(inp['name'], data.shape)
#                 cuda.memcpy_htod_async(inp['gpu'], data, self.stream)
#                 context.set_tensor_address(inp['name'], int(inp['gpu']))
        
#         # 2. Set output addresses
#         for out in io_dict['outputs']:
#             context.set_tensor_address(out['name'], int(out['gpu']))
        
#         # 3. Execute
#         context.execute_async_v3(stream_handle=self.stream.handle)
        
#         # 4. Copy results back
#         results = {}
#         for out in io_dict['outputs']:
#             cuda.memcpy_dtoh_async(out['cpu'], out['gpu'], self.stream)
#             results[out['name']] = out['cpu']
        
#         self.stream.synchronize()
#         return results

#     def _preprocess_image(self, image_path):
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Image not found: {image_path}")
#         img = cv2.imread(image_path)
#         if img is None:
#              raise ValueError("Failed to decode image.")
        
#         original_img = img.copy() # Keep original for return
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (1008, 1008))
#         img = img.astype(np.float32) / 255.0
#         img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#         img = img.transpose(2, 0, 1)
#         img = img[np.newaxis, ...]
#         return img, original_img

#     def predict(self, image_path, prompt="chessboard"):
#         """
#         Run the full segmentation pipeline.
#         Returns: (binary_mask, original_image)
#         """
#         # 1. Prepare Inputs
#         image_tensor, original_img = self._preprocess_image(image_path)
        
#         encoded = self.tokenizer.encode(prompt)
#         input_ids = np.array([encoded.ids], dtype=np.int32)
#         attention_mask = np.array([encoded.attention_mask], dtype=np.int32)

#         # 2. Vision Inference
#         vision_res = self._run_inference(
#             self.vision_ctx, self.vision_io, {'images': image_tensor}
#         )

#         # 3. Text Inference
#         text_data = {'input_ids': input_ids, 'attention_mask': attention_mask}
#         text_res = self._run_inference(
#             self.text_ctx, self.text_io, text_data
#         )

#         # 4. Decoder Inference
#         decoder_data = {}
#         decoder_data.update(vision_res)
#         decoder_data['prompt_features'] = text_res['text_features']
#         decoder_data['prompt_mask'] = attention_mask
        
#         decoder_res = self._run_inference(
#             self.decoder_ctx, self.decoder_io, decoder_data
#         )

#         # 5. Post-Processing
#         # Get Mask and Score keys dynamically
#         mask_key = [k for k in decoder_res.keys() if 'pred_masks' in k][0]
#         score_key = [k for k in decoder_res.keys() if 'pred_logits' in k][0]

#         pred_masks = decoder_res[mask_key]
#         pred_scores = decoder_res[score_key]

#         # Select best mask
#         best_idx = np.argmax(pred_scores[0])
#         raw_mask = pred_masks[0, best_idx]

#         # Sigmoid
#         if raw_mask.max() > 1.0 or raw_mask.min() < 0.0:
#             prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
#         else:
#             prob_mask = raw_mask

#         # Resize to original dimensions
#         h, w = original_img.shape[:2]
#         mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
#         binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

#         return binary_mask, original_img
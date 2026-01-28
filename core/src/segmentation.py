import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
from tokenizers import Tokenizer

class ChessboardSegmenter:
    def __init__(self, assets_dir, prompts=["chessboard", "chess pieces"]):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # --- FIX: Use a Dedicated Stream (Prevents Deadlock) ---
        self.stream = cuda.Stream()
        
        # Paths
        vision_path = os.path.join(assets_dir, 'vision_encoder_fp16.engine')
        text_path = os.path.join(assets_dir, 'text_encoder_fp16.engine')
        decoder_path = os.path.join(assets_dir, 'decoder_fp16.engine')
        tokenizer_path = os.path.join(assets_dir, 'tokenizer.json')

        # Load Tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=32, pad_id=0)
        self.tokenizer.enable_truncation(max_length=32)

        # Load Engines
        self.vision_engine = self._load_engine(vision_path)
        self.text_engine = self._load_engine(text_path)
        self.decoder_engine = self._load_engine(decoder_path)

        # Create Contexts
        self.vision_ctx = self.vision_engine.create_execution_context()
        self.text_ctx = self.text_engine.create_execution_context()
        self.decoder_ctx = self.decoder_engine.create_execution_context()

        # Allocate Memory
        self.vision_io = self._allocate_buffers(self.vision_engine)
        self.text_io = self._allocate_buffers(self.text_engine)
        self.decoder_io = self._allocate_buffers(self.decoder_engine)
        
        # Pre-compute Embeddings
        self.prompt_cache = {}
        for p in prompts:
            encoded = self.tokenizer.encode(p)
            input_ids = np.array([encoded.ids], dtype=np.int32)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int32)
            
            text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            self._run_inference(self.text_ctx, self.text_io, text_inputs)
            
            features = np.array(self.text_io['outputs'][0]['cpu'], copy=True)
            self.prompt_cache[p] = {'features': features, 'mask': attention_mask}

    def _load_engine(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self, engine):
        inputs, outputs = [], []
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = list(engine.get_tensor_shape(tensor_name))
            tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
            # Dynamic shape handling
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
                elif 'pred_logits' in tensor_name: tensor_shape = [1, 200]
                else: tensor_shape = [1 if x == -1 else x for x in tensor_shape]
            
            cpu_buffer = cuda.pagelocked_empty(tensor_shape, dtype=tensor_dtype)
            gpu_buffer = cuda.mem_alloc(cpu_buffer.nbytes)
            
            buffer_dict = {'name': tensor_name, 'gpu': gpu_buffer, 'cpu': cpu_buffer, 'shape': tensor_shape, 'dtype': tensor_dtype}
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(buffer_dict)
            else:
                outputs.append(buffer_dict)
        return {'inputs': inputs, 'outputs': outputs}

    def _run_inference(self, context, io_dict, input_data):
        # 1. Copy Inputs (ASYNC)
        for inp in io_dict['inputs']:
            name = inp['name']
            if name in input_data:
                data = input_data[name]
                if data.dtype != inp['dtype']: data = data.astype(inp['dtype'])
                np.copyto(inp['cpu'], data.ravel().reshape(inp['shape']))
                
                # Use Async Copy
                cuda.memcpy_htod_async(inp['gpu'], inp['cpu'], self.stream)
                context.set_input_shape(name, inp['shape'])
            else:
                context.set_input_shape(name, inp['shape'])
            context.set_tensor_address(name, int(inp['gpu']))
        
        for out in io_dict['outputs']:
            context.set_tensor_address(out['name'], int(out['gpu']))
        
        # 2. Execute on Dedicated Stream (Non-Blocking)
        if not context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError(f"Execution failed for {io_dict['inputs'][0]['name']}")
        
        # 3. Copy Outputs (ASYNC)
        results = {}
        for out in io_dict['outputs']:
            # Use Async Copy
            cuda.memcpy_dtoh_async(out['cpu'], out['gpu'], self.stream)
            results[out['name']] = out['cpu']
        
        # 4. Synchronize just this stream (Safe)
        self.stream.synchronize()
        return results

    def _preprocess_image(self, image_source):
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image not found: {image_source}")
            img = cv2.imread(image_source, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("Failed to decode image.")
        elif isinstance(image_source, np.ndarray):
            img = image_source
        else:
            raise ValueError("Invalid input type. Expected path or numpy array.")

        original_img = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1008, 1008))
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return img, original_img

    def encode_image(self, image_source):
        image_tensor, original_img = self._preprocess_image(image_source)
        vision_res = self._run_inference(self.vision_ctx, self.vision_io, {'images': image_tensor})
        return vision_res, original_img

    def decode_from_features(self, vision_res, original_img, prompt="chessboard"):
        if prompt not in self.prompt_cache:
            raise ValueError(f"Prompt '{prompt}' not cached.")
        
        decoder_input = {}
        decoder_input.update(vision_res)
        decoder_input['prompt_features'] = self.prompt_cache[prompt]['features']
        decoder_input['prompt_mask'] = self.prompt_cache[prompt]['mask']
        
        decoder_res = self._run_inference(self.decoder_ctx, self.decoder_io, decoder_input)

        mask_key = next(k for k in decoder_res.keys() if 'pred_masks' in k)
        score_key = next(k for k in decoder_res.keys() if 'pred_logits' in k)
        pred_masks = decoder_res[mask_key] 
        pred_scores = decoder_res[score_key]

        if prompt == "chessboard":
            best_idx = np.argmax(pred_scores[0])
            best_score = float(pred_scores[0][best_idx])
            raw_mask = pred_masks[0, best_idx]
            raw_mask = np.clip(raw_mask, -80, 80)
            prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))

            h, w = original_img.shape[:2]
            mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            return binary_mask, original_img, best_score

        else:
            h, w = original_img.shape[:2]
            valid_masks_list = []
            flat_scores = 1.0 / (1.0 + np.exp(-np.clip(pred_scores[0], -80, 80)))
            valid_indices = np.where(flat_scores > 0.2)[0] 
            
            for idx in valid_indices:
                raw_mask = np.clip(pred_masks[0, idx], -80, 80)
                prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
                mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                instance_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                if cv2.countNonZero(instance_mask) > 0:
                    valid_masks_list.append(instance_mask)
            
            return valid_masks_list, original_img, 0.0

    def predict(self, image_source, prompt="chessboard"):
        return self.decode_from_features(*self.encode_image(image_source), prompt)

# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import cv2
# import os
# from tokenizers import Tokenizer

# class ChessboardSegmenter:
#     def __init__(self, assets_dir, prompts=["chessboard", "chess pieces"]):
#         self.logger = trt.Logger(trt.Logger.WARNING)
#         self.runtime = trt.Runtime(self.logger)
#         # self.stream = cuda.Stream()
#         self.stream = None
        
#         # Paths
#         vision_path = os.path.join(assets_dir, 'vision_encoder_fp16.engine')
#         text_path = os.path.join(assets_dir, 'text_encoder_fp16.engine')
#         decoder_path = os.path.join(assets_dir, 'decoder_fp16.engine')
#         tokenizer_path = os.path.join(assets_dir, 'tokenizer.json')

#         # Load Tokenizer
#         if not os.path.exists(tokenizer_path):
#             raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
#         self.tokenizer = Tokenizer.from_file(tokenizer_path)
#         self.tokenizer.enable_padding(length=32, pad_id=0)
#         self.tokenizer.enable_truncation(max_length=32)

#         # Load Engines
#         self.vision_engine = self._load_engine(vision_path)
#         self.text_engine = self._load_engine(text_path)
#         self.decoder_engine = self._load_engine(decoder_path)

#         # Create Contexts
#         self.vision_ctx = self.vision_engine.create_execution_context()
#         self.text_ctx = self.text_engine.create_execution_context()
#         self.decoder_ctx = self.decoder_engine.create_execution_context()

#         # Allocate Memory
#         self.vision_io = self._allocate_buffers(self.vision_engine)
#         self.text_io = self._allocate_buffers(self.text_engine)
#         self.decoder_io = self._allocate_buffers(self.decoder_engine)
        
#         # Pre-compute Embeddings
#         self.prompt_cache = {}
#         for p in prompts:
#             encoded = self.tokenizer.encode(p)
#             input_ids = np.array([encoded.ids], dtype=np.int32)
#             attention_mask = np.array([encoded.attention_mask], dtype=np.int32)
            
#             text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
#             self._run_inference(self.text_ctx, self.text_io, text_inputs)
            
#             features = np.array(self.text_io['outputs'][0]['cpu'], copy=True)
#             self.prompt_cache[p] = {'features': features, 'mask': attention_mask}

#     def _load_engine(self, engine_path):
#         if not os.path.exists(engine_path):
#             raise FileNotFoundError(f"Engine not found: {engine_path}")
#         with open(engine_path, 'rb') as f:
#             engine_data = f.read()
#         return self.runtime.deserialize_cuda_engine(engine_data)

#     def _allocate_buffers(self, engine):
#         inputs, outputs = [], []
#         for i in range(engine.num_io_tensors):
#             tensor_name = engine.get_tensor_name(i)
#             tensor_shape = list(engine.get_tensor_shape(tensor_name))
#             tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
#             if -1 in tensor_shape:
#                 if 'image' in tensor_name: tensor_shape = [1, 3, 1008, 1008]
#                 elif 'fpn_feat_0' in tensor_name: tensor_shape = [1, 256, 288, 288]
#                 elif 'fpn_feat_1' in tensor_name: tensor_shape = [1, 256, 144, 144]
#                 elif 'fpn_feat_2' in tensor_name: tensor_shape = [1, 256, 72, 72]
#                 elif 'fpn_pos_2' in tensor_name: tensor_shape = [1, 256, 72, 72]
#                 elif 'text_features' in tensor_name or 'prompt_features' in tensor_name: tensor_shape = [1, 32, 256]
#                 elif 'text_mask' in tensor_name or 'prompt_mask' in tensor_name: tensor_shape = [1, 32]
#                 elif 'input_ids' in tensor_name or 'attention_mask' in tensor_name: tensor_shape = [1, 32]
#                 elif 'pred_masks' in tensor_name: tensor_shape = [1, 200, 288, 288]
#                 elif 'pred_logits' in tensor_name: tensor_shape = [1, 200]
#                 else: tensor_shape = [1 if x == -1 else x for x in tensor_shape]
            
#             cpu_buffer = cuda.pagelocked_empty(tensor_shape, dtype=tensor_dtype)
#             gpu_buffer = cuda.mem_alloc(cpu_buffer.nbytes)
            
#             buffer_dict = {'name': tensor_name, 'gpu': gpu_buffer, 'cpu': cpu_buffer, 'shape': tensor_shape, 'dtype': tensor_dtype}
#             if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
#                 inputs.append(buffer_dict)
#             else:
#                 outputs.append(buffer_dict)
#         return {'inputs': inputs, 'outputs': outputs}

#     def _run_inference(self, context, io_dict, input_data):
#         # 1. Copy Inputs (Synchronous)
#         for inp in io_dict['inputs']:
#             name = inp['name']
#             if name in input_data:
#                 data = input_data[name]
#                 if data.dtype != inp['dtype']: data = data.astype(inp['dtype'])
#                 np.copyto(inp['cpu'], data.ravel().reshape(inp['shape']))
                
#                 # CHANGE: Use synchronous copy (htod, not htod_async)
#                 cuda.memcpy_htod(inp['gpu'], inp['cpu'])
#                 context.set_input_shape(name, inp['shape'])
#             else:
#                 context.set_input_shape(name, inp['shape'])
#             context.set_tensor_address(name, int(inp['gpu']))
        
#         # 2. Set Output Addresses
#         for out in io_dict['outputs']:
#             context.set_tensor_address(out['name'], int(out['gpu']))
        
#         # 3. Execute (CHANGE: Use stream_handle=0 for Default Stream)
#         if not context.execute_async_v3(stream_handle=0):
#             raise RuntimeError(f"Execution failed for {io_dict['inputs'][0]['name']}")
        
#         # 4. Copy Outputs (Synchronous)
#         results = {}
#         for out in io_dict['outputs']:
#             # CHANGE: Use synchronous copy (dtoh, not dtoh_async)
#             cuda.memcpy_dtoh(out['cpu'], out['gpu'])
#             results[out['name']] = out['cpu']
        
#         # No need to synchronize, these calls are blocking
#         return results

    
#     def _preprocess_image(self, image_source):
#         if isinstance(image_source, str):
#             if not os.path.exists(image_source):
#                 raise FileNotFoundError(f"Image not found: {image_source}")
#             img = cv2.imread(image_source, cv2.IMREAD_COLOR)
#             if img is None: raise ValueError("Failed to decode image.")
#         elif isinstance(image_source, np.ndarray):
#             img = image_source
#         else:
#             raise ValueError("Invalid input type. Expected path or numpy array.")

#         original_img = img
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (1008, 1008))
#         img = img.astype(np.float32) / 255.0
#         img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#         img = img.transpose(2, 0, 1)[np.newaxis, ...]
#         return img, original_img

#     def encode_image(self, image_source):
#         """Run ONLY the Vision Encoder and return features."""
#         image_tensor, original_img = self._preprocess_image(image_source)
#         vision_res = self._run_inference(self.vision_ctx, self.vision_io, {'images': image_tensor})
#         return vision_res, original_img

#     def decode_from_features(self, vision_res, original_img, prompt="chessboard"):
#         """Run ONLY the Decoder using cached vision features."""
#         if prompt not in self.prompt_cache:
#             raise ValueError(f"Prompt '{prompt}' not cached.")
        
#         decoder_input = {}
#         decoder_input.update(vision_res)
#         decoder_input['prompt_features'] = self.prompt_cache[prompt]['features']
#         decoder_input['prompt_mask'] = self.prompt_cache[prompt]['mask']
        
#         decoder_res = self._run_inference(self.decoder_ctx, self.decoder_io, decoder_input)

#         mask_key = next(k for k in decoder_res.keys() if 'pred_masks' in k)
#         score_key = next(k for k in decoder_res.keys() if 'pred_logits' in k)
#         pred_masks = decoder_res[mask_key] 
#         pred_scores = decoder_res[score_key]

#         if prompt == "chessboard":
#             best_idx = np.argmax(pred_scores[0])
#             best_score = float(pred_scores[0][best_idx])
#             raw_mask = pred_masks[0, best_idx]
            
#             # Clip and Sigmoid
#             raw_mask = np.clip(raw_mask, -80, 80)
#             prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))

#             h, w = original_img.shape[:2]
#             mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
#             binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
#             return binary_mask, original_img, best_score

#         else:
#             h, w = original_img.shape[:2]
#             valid_masks_list = []
            
#             # Apply Clip and Sigmoid to scores
#             flat_scores = 1.0 / (1.0 + np.exp(-np.clip(pred_scores[0], -80, 80)))
#             valid_indices = np.where(flat_scores > 0.35)[0] 
            
#             for idx in valid_indices:
#                 raw_mask = np.clip(pred_masks[0, idx], -80, 80)
#                 prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
                
#                 # FIX: Use INTER_LINEAR for smooth edges (matches your working script)
#                 mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
#                 instance_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
#                 if cv2.countNonZero(instance_mask) > 0:
#                     valid_masks_list.append(instance_mask)
            
#             return valid_masks_list, original_img, 0.0

#     def predict(self, image_source, prompt="chessboard"):
#         if prompt not in self.prompt_cache:
#             raise ValueError(f"Prompt '{prompt}' not cached.")

#         image_tensor, original_img = self._preprocess_image(image_source)
#         vision_res = self._run_inference(self.vision_ctx, self.vision_io, {'images': image_tensor})

#         decoder_input = {}
#         decoder_input.update(vision_res)
#         decoder_input['prompt_features'] = self.prompt_cache[prompt]['features']
#         decoder_input['prompt_mask'] = self.prompt_cache[prompt]['mask']
        
#         decoder_res = self._run_inference(self.decoder_ctx, self.decoder_io, decoder_input)

#         mask_key = next(k for k in decoder_res.keys() if 'pred_masks' in k)
#         score_key = next(k for k in decoder_res.keys() if 'pred_logits' in k)
#         pred_masks = decoder_res[mask_key] 
#         pred_scores = decoder_res[score_key]

#         if prompt == "chessboard":
#             best_idx = np.argmax(pred_scores[0])
#             best_score = float(pred_scores[0][best_idx])
#             raw_mask = pred_masks[0, best_idx]
            
#             # Clip and Sigmoid
#             raw_mask = np.clip(raw_mask, -80, 80)
#             prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))

#             h, w = original_img.shape[:2]
#             mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
#             binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
#             return binary_mask, original_img, best_score

#         else:
#             h, w = original_img.shape[:2]
#             valid_masks_list = []
            
#             # Apply Clip and Sigmoid to scores
#             flat_scores = 1.0 / (1.0 + np.exp(-np.clip(pred_scores[0], -80, 80)))
#             valid_indices = np.where(flat_scores > 0.35)[0] 
            
#             for idx in valid_indices:
#                 raw_mask = np.clip(pred_masks[0, idx], -80, 80)
#                 prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
                
#                 # FIX: Use INTER_LINEAR for smooth edges (matches your working script)
#                 mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
#                 instance_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
#                 if cv2.countNonZero(instance_mask) > 0:
#                     valid_masks_list.append(instance_mask)
            
#             return valid_masks_list, original_img, 0.0

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
        for inp in io_dict['inputs']:
            name = inp['name']
            if name in input_data:
                data = input_data[name]
                if data.dtype != inp['dtype']: data = data.astype(inp['dtype'])
                np.copyto(inp['cpu'], data.ravel().reshape(inp['shape']))
                cuda.memcpy_htod_async(inp['gpu'], inp['cpu'], self.stream)
                context.set_input_shape(name, inp['shape'])
            else:
                context.set_input_shape(name, inp['shape'])
            context.set_tensor_address(name, int(inp['gpu']))
        
        for out in io_dict['outputs']:
            context.set_tensor_address(out['name'], int(out['gpu']))
        
        if not context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError(f"Execution failed for {io_dict['inputs'][0]['name']}")
        
        results = {}
        for out in io_dict['outputs']:
            cuda.memcpy_dtoh_async(out['cpu'], out['gpu'], self.stream)
            results[out['name']] = out['cpu']
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
        """Run ONLY the Vision Encoder and return features."""
        image_tensor, original_img = self._preprocess_image(image_source)
        vision_res = self._run_inference(self.vision_ctx, self.vision_io, {'images': image_tensor})
        return vision_res, original_img

    def decode_from_features(self, vision_res, original_img, prompt="chessboard"):
        """Run ONLY the Decoder using cached vision features."""
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
            
            # Clip and Sigmoid
            raw_mask = np.clip(raw_mask, -80, 80)
            prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))

            h, w = original_img.shape[:2]
            mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            return binary_mask, original_img, best_score

        else:
            h, w = original_img.shape[:2]
            valid_masks_list = []
            
            # Apply Clip and Sigmoid to scores
            flat_scores = 1.0 / (1.0 + np.exp(-np.clip(pred_scores[0], -80, 80)))
            valid_indices = np.where(flat_scores > 0.35)[0] 
            
            for idx in valid_indices:
                raw_mask = np.clip(pred_masks[0, idx], -80, 80)
                prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
                
                # FIX: Use INTER_LINEAR for smooth edges (matches your working script)
                mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                instance_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
                if cv2.countNonZero(instance_mask) > 0:
                    valid_masks_list.append(instance_mask)
            
            return valid_masks_list, original_img, 0.0

    def predict(self, image_source, prompt="chessboard"):
        if prompt not in self.prompt_cache:
            raise ValueError(f"Prompt '{prompt}' not cached.")

        image_tensor, original_img = self._preprocess_image(image_source)
        vision_res = self._run_inference(self.vision_ctx, self.vision_io, {'images': image_tensor})

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
            
            # Clip and Sigmoid
            raw_mask = np.clip(raw_mask, -80, 80)
            prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))

            h, w = original_img.shape[:2]
            mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            return binary_mask, original_img, best_score

        else:
            h, w = original_img.shape[:2]
            valid_masks_list = []
            
            # Apply Clip and Sigmoid to scores
            flat_scores = 1.0 / (1.0 + np.exp(-np.clip(pred_scores[0], -80, 80)))
            valid_indices = np.where(flat_scores > 0.35)[0] 
            
            for idx in valid_indices:
                raw_mask = np.clip(pred_masks[0, idx], -80, 80)
                prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
                
                # FIX: Use INTER_LINEAR for smooth edges (matches your working script)
                mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                instance_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
                if cv2.countNonZero(instance_mask) > 0:
                    valid_masks_list.append(instance_mask)
            
            return valid_masks_list, original_img, 0.0

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

#         # 2. Load Engines
#         self.vision_engine = self._load_engine(vision_path)
#         self.text_engine = self._load_engine(text_path)
#         self.decoder_engine = self._load_engine(decoder_path)

#         # 3. Create Contexts
#         self.vision_ctx = self.vision_engine.create_execution_context()
#         self.text_ctx = self.text_engine.create_execution_context()
#         self.decoder_ctx = self.decoder_engine.create_execution_context()

#         # 4. Allocate Pinned Memory
#         self.vision_io = self._allocate_buffers(self.vision_engine)
#         self.text_io = self._allocate_buffers(self.text_engine)
#         self.decoder_io = self._allocate_buffers(self.decoder_engine)
        
#         # 5. Pre-compute Embeddings for ALL prompts
#         self.prompt_cache = {}
        
#         for p in prompts:
#             print(f"Caching embedding for: '{p}'")
#             encoded = self.tokenizer.encode(p)
#             input_ids = np.array([encoded.ids], dtype=np.int32)
#             attention_mask = np.array([encoded.attention_mask], dtype=np.int32)
            
#             # Run Inference to get features
#             text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
#             self._run_inference(self.text_ctx, self.text_io, text_inputs)
            
#             # Store the CPU copy of the features and the mask
#             # We copy specific output (assuming index 0 is features, check your engine if unsure)
#             # Usually outputs are: [text_features, ...]
#             features = np.array(self.text_io['outputs'][0]['cpu'], copy=True)
            
#             self.prompt_cache[p] = {
#                 'features': features,
#                 'mask': attention_mask
#             }

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
            
#             # Handle dynamic shapes (batch size 1)
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
#                 elif 'pred_boxes' in tensor_name: tensor_shape = [1, 200, 4]
#                 elif 'pred_logits' in tensor_name: tensor_shape = [1, 200]
#                 elif 'presence_logits' in tensor_name: tensor_shape = [1, 1]
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
#         for inp in io_dict['inputs']:
#             name = inp['name']
#             if name in input_data:
#                 data = input_data[name]
#                 if data.dtype != inp['dtype']: data = data.astype(inp['dtype'])
#                 np.copyto(inp['cpu'], data.ravel().reshape(inp['shape']))
#                 cuda.memcpy_htod_async(inp['gpu'], inp['cpu'], self.stream)
#                 context.set_input_shape(name, inp['shape'])
#             else:
#                 context.set_input_shape(name, inp['shape'])
#             context.set_tensor_address(name, int(inp['gpu']))
        
#         for out in io_dict['outputs']:
#             context.set_tensor_address(out['name'], int(out['gpu']))
        
#         if not context.execute_async_v3(stream_handle=self.stream.handle):
#             raise RuntimeError(f"Execution failed for {io_dict['inputs'][0]['name']}")
        
#         results = {}
#         for out in io_dict['outputs']:
#             cuda.memcpy_dtoh_async(out['cpu'], out['gpu'], self.stream)
#             results[out['name']] = out['cpu']
#         self.stream.synchronize()
#         return results

#     def _preprocess_image(self, image_source):
#         # Handle file path
#         if isinstance(image_source, str):
#             if not os.path.exists(image_source):
#                 raise FileNotFoundError(f"Image not found: {image_source}")
#             img = cv2.imread(image_source, cv2.IMREAD_COLOR)
#             if img is None: raise ValueError("Failed to decode image.")
#         # Handle numpy array (e.g., crop)
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

#     def predict(self, image_source, prompt="chessboard"):
#         if prompt not in self.prompt_cache:
#             raise ValueError(f"Prompt '{prompt}' not cached. Available: {list(self.prompt_cache.keys())}")

#         # 1. Vision
#         image_tensor, original_img = self._preprocess_image(image_source)
#         vision_res = self._run_inference(self.vision_ctx, self.vision_io, {'images': image_tensor})

#         # 2. Decoder
#         decoder_input = {}
#         decoder_input.update(vision_res)
#         decoder_input['prompt_features'] = self.prompt_cache[prompt]['features']
#         decoder_input['prompt_mask'] = self.prompt_cache[prompt]['mask']
        
#         decoder_res = self._run_inference(self.decoder_ctx, self.decoder_io, decoder_input)

#         # 3. Process Result
#         mask_key = next(k for k in decoder_res.keys() if 'pred_masks' in k)
#         score_key = next(k for k in decoder_res.keys() if 'pred_logits' in k)

#         pred_masks = decoder_res[mask_key] # [1, 200, 288, 288]
#         pred_scores = decoder_res[score_key] # [1, 200]

#         # For "chessboard", we want the single best mask
#         if prompt == "chessboard":
#             best_idx = np.argmax(pred_scores[0])
#             best_score = float(pred_scores[0][best_idx])
#             raw_mask = pred_masks[0, best_idx]
            
#             if raw_mask.max() > 1.0 or raw_mask.min() < 0.0:
#                 prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
#             else:
#                 prob_mask = raw_mask

#             h, w = original_img.shape[:2]
#             mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_NEAREST)
#             binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
#             return binary_mask, original_img, best_score

#         # For "chess pieces", we want ALL valid instances combined
#         else:
#             h, w = original_img.shape[:2]
#             combined_mask = np.zeros((h, w), dtype=np.uint8)
            
#             # Apply Sigmoid to scores
#             flat_scores = 1.0 / (1.0 + np.exp(-np.clip(pred_scores[0], -80, 80)))
#             valid_indices = np.where(flat_scores > 0.35)[0] # Threshold for pieces
            
#             for idx in valid_indices:
#                 raw_mask = np.clip(pred_masks[0, idx], -80, 80)
#                 prob_mask = 1.0 / (1.0 + np.exp(-raw_mask))
#                 mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_NEAREST)
#                 instance_mask = (mask_resized > 0.5).astype(np.uint8) * 255
#                 combined_mask = cv2.bitwise_or(combined_mask, instance_mask)
            
#             return combined_mask, original_img, 0.0 # Score irrelevant for multi-instance
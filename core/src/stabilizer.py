import numpy as np
import chess

class PositionStabilizer:
    def __init__(self, patience=3, confirmation_frames=2):
        """
        A stabilizer backed by a real Chess Engine.
        Includes logic to correct grid misalignment based on engine state.
        """
        self.board = chess.Board()
        self.counters = np.zeros(64, dtype=np.int32)
        self.PATIENCE = patience
        self.CONFIRMATION_FRAMES = confirmation_frames
        self.pending_move = None 
        self.pending_move_count = 0 

    # --- NEW: LOGIC BASED GRID CORRECTION ---
    def recover_grid_position(self, grid_points, raw_indices):
        """
        Detects if the grid is misaligned (e.g. shifted UP by one row) by comparing
        raw detections with the engine's known piece locations.
        
        Returns: (corrected_grid_points, corrected_indices_list, was_corrected_bool)
        """
        # 1. Get the "Truth" (Where the engine says pieces are)
        engine_indices = set()
        for sq in chess.SQUARES:
            if self.board.piece_at(sq):
                engine_indices.add(self._to_camera_index(sq))

        # 2. Get the "Observation"
        raw_set = set(raw_indices)
        current_score = len(raw_set.intersection(engine_indices))

        # 3. Define Shifts: key=index_offset, value=(x_multiplier, y_multiplier)
        # grid shape 9x9. Index 0 is Top-Left.
        # Shift -8: Index i becomes i-8. (Detected 8->Real 0). Grid was too High. Move Grid DOWN (+Y).
        # Shift +8: Index i becomes i+8. (Detected 0->Real 8). Grid was too Low. Move Grid UP (-Y).
        shifts = {
            0:  (0, 0),
            -8: (0, 1),   # Move Grid DOWN
            8:  (0, -1),  # Move Grid UP
            -1: (1, 0),   # Move Grid RIGHT
            1:  (-1, 0)   # Move Grid LEFT
        }

        best_shift = 0
        best_score = current_score

        # 4. Test all shifts
        for shift_idx, _ in shifts.items():
            if shift_idx == 0: continue
            
            # Apply shift to observed indices
            # (e.g. if we shift -8, we are checking if moving observations UP matches the board)
            shifted_set = {i + shift_idx for i in raw_set if 0 <= i + shift_idx < 64}
            score = len(shifted_set.intersection(engine_indices))

            # HEURISTIC: We need a significant improvement (+2 pieces or more) to justify a jump
            if score > best_score + 2:
                best_score = score
                best_shift = shift_idx

        # 5. Apply Correction if a better alignment was found
        if best_shift != 0:
            print(f"[Stabilizer] ⚠️ Grid Misalignment Detected (Index Shift {best_shift}). Correcting...")

            # A. Calculate Pixel Offset
            g = grid_points.reshape(9, 9, 2)
            # Average height of a cell (Bottom Row Y - Top Row Y) / 8
            cell_h = np.mean(g[8, :, 1] - g[0, :, 1]) / 8.0
            # Average width of a cell
            cell_w = np.mean(g[:, 8, 0] - g[:, 0, 0]) / 8.0

            x_mult, y_mult = shifts[best_shift]
            pixel_offset = np.array([x_mult * cell_w, y_mult * cell_h])

            # B. Apply to Grid Points
            corrected_grid = grid_points + pixel_offset

            # C. Apply to Indices (so the current frame processing is also correct)
            corrected_indices = [i + best_shift for i in raw_indices if 0 <= i + best_shift < 64]

            return corrected_grid, corrected_indices, True

        return grid_points, raw_indices, False

    def _to_chess_square(self, index):
        row, col = divmod(index, 8)
        chess_rank = 7 - row
        chess_file = col
        return chess.square(chess_file, chess_rank)

    def _to_camera_index(self, square):
        chess_file = chess.square_file(square)
        chess_rank = chess.square_rank(square)
        row = 7 - chess_rank
        col = chess_file
        return row * 8 + col
    
    def _is_central_grid(self, index):
        row, col = divmod(index, 8)
        return 2 <= row <= 5




    def update(self, raw_indices):
        """
        Robust update that ignores noise by checking if *any* legal move
        matches the subset of changed squares.
        """
        # 1. Get current reality according to the Engine
        engine_occupied = set()
        for sq in chess.SQUARES:
            if self.board.piece_at(sq):
                engine_occupied.add(self._to_camera_index(sq))

        # 2. Get camera reality
        camera_occupied = {int(x) for x in raw_indices}

        # 3. Calculate Diffs (The sets of changed squares)
        vanished = engine_occupied - camera_occupied
        appeared = camera_occupied - engine_occupied

        # 4. ROBUST MOVE DETECTOR
        # Instead of checking lengths, we check if a legal move exists 
        # that explains the changes we care about.
        
        detected_move = None
        move_type = None
        
        candidate_moves = []

        # Iterate through EVERY legal move available in the current position
        for move in self.board.legal_moves:
            src_idx = self._to_camera_index(move.from_square)
            dst_idx = self._to_camera_index(move.to_square)

            # Check Standard Move: Src must satisfy vanished, Dst must satisfy appeared
            if src_idx in vanished and dst_idx in appeared:
                candidate_moves.append((move, "STANDARD"))
            
            # Check Castling (King moves 2 squares, Rook moves automatically)
            # We only strictly check the King's start/end to be robust against Rook noise
            elif self.board.is_castling(move):
                if src_idx in vanished and dst_idx in appeared:
                     candidate_moves.append((move, "CASTLING"))

            # Check En Passant (Pawn moves to empty square, but captures pawn behind it)
            elif self.board.is_en_passant(move):
                 if src_idx in vanished and dst_idx in appeared:
                     candidate_moves.append((move, "EN_PASSANT"))

        # 5. Filter Candidates
        if len(candidate_moves) == 1:
            # We found exactly one legal move that matches the camera changes!
            detected_move, move_type = candidate_moves[0]
        
        elif len(candidate_moves) == 0:
            # Maybe it's a capture? (Piece vanishes, but nothing appears because it replaced a piece)
            # Logic: Src vanished, Dst is NOT in appeared (because it was already occupied), 
            # BUT Dst is NOT in vanished either (it didn't become empty).
            # Actually, standard CV Capture:
            # Attacker (Src) moves to Victim (Dst). 
            # Src vanishes. Dst stays occupied (symbol changes, but binary occupancy doesn't).
            # So: vanished = {Src}, appeared = {}
            
            for move in self.board.legal_moves:
                if self.board.is_capture(move):
                    src_idx = self._to_camera_index(move.from_square)
                    dst_idx = self._to_camera_index(move.to_square)
                    
                    # For a capture: Src must vanish.
                    # Dst must NOT be in vanished (it is still occupied).
                    if src_idx in vanished and dst_idx not in vanished:
                        # If we have multiple captures from same square, we can't distinguish yet
                        # But usually rare.
                        detected_move = move
                        move_type = "CAPTURE"
                        break

        # --- MOVE CONFIRMATION (With Grace Period) ---
        if detected_move is not None:
            if self.pending_move == detected_move:
                self.pending_move_count += 1
                print(f"[ChessEngine] 🔄 Confirming {move_type}: {self.board.san(detected_move)} ({self.pending_move_count}/{self.CONFIRMATION_FRAMES})")
                
                if self.pending_move_count >= self.CONFIRMATION_FRAMES:
                    print(f"[ChessEngine] ✅ CONFIRMED {move_type}: {self.board.san(detected_move)}")
                    self.board.push(detected_move)
                    self.pending_move = None
                    self.pending_move_count = 0
            else:
                self.pending_move = detected_move
                self.pending_move_count = 1
                print(f"[ChessEngine] 👁️  Detected {move_type}: {self.board.san(detected_move)}")
        
        else:
            # IMPROVEMENT: Don't kill the streak immediately on one bad frame.
            # Only kill it if we see a *different* move or silence for too long.
            # For now, we will just reduce the count instead of resetting to 0.
            if self.pending_move_count > 0:
                self.pending_move_count -= 1
                print(f"[ChessEngine] 📉 Signal lost, decay count: {self.pending_move_count}")
                if self.pending_move_count == 0:
                    self.pending_move = None

        return self._get_display_map()

    # def update(self, raw_indices):
    #     """
    #     Compares raw camera input vs. Chess Engine state.
    #     Returns: dict {index: piece_symbol} for all occupied squares.
    #     """
    #     # 1. Get current reality according to the Engine
    #     engine_occupied_indices = set()
    #     for sq in chess.SQUARES:
    #         if self.board.piece_at(sq):
    #             engine_occupied_indices.add(self._to_camera_index(sq))

    #     # 2. Get camera reality (Safe Cast to int)
    #     camera_occupied_indices = {int(x) for x in raw_indices}

    #     # 3. SANITY CHECK: Reject frames with too many missing pieces
    #     engine_piece_count = len(engine_occupied_indices)
    #     camera_piece_count = len(camera_occupied_indices)
    #     piece_difference = engine_piece_count - camera_piece_count
        
    #     if piece_difference > 5: # Slightly relaxed since we might correct it next frame
    #         print(f"[ChessEngine] ⚠️ Frame REJECTED: {piece_difference} pieces missing")
    #         self.pending_move = None
    #         self.pending_move_count = 0
    #         return self._get_display_map()

    #     # 4. Calculate Diffs
    #     vanished = list(engine_occupied_indices - camera_occupied_indices)
    #     appeared = list(camera_occupied_indices - engine_occupied_indices)

    #     # 5. Try to identify the move
    #     detected_move = None
    #     move_type = None

    #     # --- Check for moves in priority order ---
        
    #     # PRIORITY 1: Piece appeared on central grid
    #     vanished_central = [idx for idx in vanished if self._is_central_grid(idx)]
    #     appeared_central = [idx for idx in appeared if self._is_central_grid(idx)]
        
    #     if len(appeared_central) > 0 and detected_move is None:
    #         for dst_idx in appeared_central:
    #             dst_sq = self._to_chess_square(dst_idx)
    #             candidate_moves = []
    #             for move in self.board.legal_moves:
    #                 if move.to_square == dst_sq:
    #                     candidate_moves.append(move)
    #             if len(candidate_moves) == 1:
    #                 detected_move = candidate_moves[0]
    #                 move_type = "CENTRAL_GRID"
    #                 break
        
    #     # PRIORITY 2: Standard move on central grids
    #     if (len(vanished) == 1 and len(appeared) == 1 and
    #         vanished[0] in vanished_central and appeared[0] in appeared_central and
    #         detected_move is None):
    #         src_sq = self._to_chess_square(vanished[0])
    #         dst_sq = self._to_chess_square(appeared[0])
    #         move = chess.Move(src_sq, dst_sq)
    #         # Auto-Promote
    #         piece = self.board.piece_at(src_sq)
    #         if piece and piece.piece_type == chess.PAWN and chess.square_rank(dst_sq) in [0, 7]:
    #             move = chess.Move(src_sq, dst_sq, promotion=chess.QUEEN)
    #         if self.board.is_legal(move):
    #             detected_move = move
    #             move_type = "CENTRAL_MOVE"

    #     # PRIORITY 3: Standard move anywhere
    #     if len(vanished) == 1 and len(appeared) == 1 and detected_move is None:
    #         src_sq = self._to_chess_square(vanished[0])
    #         dst_sq = self._to_chess_square(appeared[0])
    #         move = chess.Move(src_sq, dst_sq)
    #         piece = self.board.piece_at(src_sq)
    #         if piece and piece.piece_type == chess.PAWN and chess.square_rank(dst_sq) in [0, 7]:
    #             move = chess.Move(src_sq, dst_sq, promotion=chess.QUEEN)
    #         if self.board.is_legal(move):
    #             detected_move = move
    #             move_type = "STANDARD"

    #     # PRIORITY 4: Capture
    #     elif len(vanished) == 1 and len(appeared) == 0 and detected_move is None:
    #         src_sq = self._to_chess_square(vanished[0])
    #         candidate_moves = [m for m in self.board.legal_moves if m.from_square == src_sq and self.board.is_capture(m)]
    #         if len(candidate_moves) == 1:
    #             detected_move = candidate_moves[0]
    #             move_type = "CAPTURE"

    #     # PRIORITY 5: Castling
    #     elif len(vanished) == 2 and len(appeared) == 2 and detected_move is None:
    #         for move in self.board.legal_moves:
    #             if self.board.is_castling(move):
    #                 detected_move = move
    #                 move_type = "CASTLING"
    #                 break

    #     # --- MOVE CONFIRMATION ---
    #     if detected_move is not None:
    #         if self.pending_move == detected_move:
    #             self.pending_move_count += 1
    #             print(f"[ChessEngine] 🔄 Confirming {move_type}: {self.board.san(detected_move)} ({self.pending_move_count}/{self.CONFIRMATION_FRAMES})")
    #             if self.pending_move_count >= self.CONFIRMATION_FRAMES:
    #                 print(f"[ChessEngine] ✅ CONFIRMED {move_type}: {self.board.san(detected_move)}")
    #                 self.board.push(detected_move)
    #                 self.counters.fill(0)
    #                 self.pending_move = None
    #                 self.pending_move_count = 0
    #                 return self._get_display_map()
    #         else:
    #             self.pending_move = detected_move
    #             self.pending_move_count = 1
    #             print(f"[ChessEngine] 👁️  Detected {move_type}: {self.board.san(detected_move)}")
    #     else:
    #         if self.pending_move is not None: print(f"[ChessEngine] ❌ Pending move lost")
    #         self.pending_move = None
    #         self.pending_move_count = 0

    #     return self._get_display_map()

    def _get_display_map(self):
        display = {}
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece: display[self._to_camera_index(sq)] = piece.symbol()
        return display




# import numpy as np
# import chess

# class PositionStabilizer:
#     def __init__(self, patience=3, confirmation_frames=2):
#         """
#         A stabilizer backed by a real Chess Engine.
#         It only updates the board state if the change visually detected
#         corresponds to a LEGAL chess move and is confirmed across multiple frames.
#         """
#         # 1. Initialize the Engine Board (Standard Starting Position)
#         self.board = chess.Board()
        
#         # 2. Counters for noise/fallback handling
#         self.counters = np.zeros(64, dtype=np.int32)
#         self.PATIENCE = patience
        
#         # 3. Move confirmation tracking
#         self.CONFIRMATION_FRAMES = confirmation_frames
#         self.pending_move = None  # Store the candidate move
#         self.pending_move_count = 0  # How many consecutive frames we've seen it

#     def _to_chess_square(self, index):
#         """
#         Maps Camera Index (0=TopLeft/A8) to python-chess Square (0=BottomLeft/A1).
#         """
#         row, col = divmod(index, 8)
#         # python-chess: rank 0 is row 7 (bottom), rank 7 is row 0 (top)
#         chess_rank = 7 - row
#         chess_file = col
#         return chess.square(chess_file, chess_rank)

#     def _to_camera_index(self, square):
#         """Maps python-chess Square back to Camera Index."""
#         chess_file = chess.square_file(square)
#         chess_rank = chess.square_rank(square)
#         row = 7 - chess_rank
#         col = chess_file
#         return row * 8 + col
    
#     def _is_central_grid(self, index):
#         """Check if index is on central lines (rows 3,4,5,6 or ranks 4,5,6,7 in chess notation)"""
#         row, col = divmod(index, 8)
#         # Rows 2,3,4,5 in 0-indexed = rows 3,4,5,6 in 1-indexed
#         return 2 <= row <= 5

#     def update(self, raw_indices):
#         """
#         Compares raw camera input vs. Chess Engine state.
#         Returns: dict {index: piece_symbol} for all occupied squares.
#         """
#         # 1. Get current reality according to the Engine
#         engine_occupied_indices = set()
#         for sq in chess.SQUARES:
#             if self.board.piece_at(sq):
#                 engine_occupied_indices.add(self._to_camera_index(sq))

#         # 2. Get camera reality (Safe Cast to int)
#         camera_occupied_indices = {int(x) for x in raw_indices}

#         # 3. SANITY CHECK: Reject frames with too many missing pieces
#         engine_piece_count = len(engine_occupied_indices)
#         camera_piece_count = len(camera_occupied_indices)
#         piece_difference = engine_piece_count - camera_piece_count
        
#         if piece_difference > 4:
#             print(f"[ChessEngine] ⚠️  Frame REJECTED: {piece_difference} pieces missing (Engine: {engine_piece_count}, Camera: {camera_piece_count})")
#             self.pending_move = None
#             self.pending_move_count = 0
#             return self._get_display_map()

#         # 4. Calculate Diffs
#         vanished = list(engine_occupied_indices - camera_occupied_indices)
#         appeared = list(camera_occupied_indices - engine_occupied_indices)

#         # 5. Try to identify the move
#         detected_move = None
#         move_type = None

#         # --- Check for moves in priority order ---
        
#         # PRIORITY 1: Piece appeared on central grid
#         vanished_central = [idx for idx in vanished if self._is_central_grid(idx)]
#         appeared_central = [idx for idx in appeared if self._is_central_grid(idx)]
        
#         if len(appeared_central) > 0 and detected_move is None:
#             for dst_idx in appeared_central:
#                 dst_sq = self._to_chess_square(dst_idx)
                
#                 # Find any legal move that goes to this destination square
#                 candidate_moves = []
#                 for move in self.board.legal_moves:
#                     if move.to_square == dst_sq:
#                         candidate_moves.append(move)
                
#                 # If there's exactly one legal move to this square
#                 if len(candidate_moves) == 1:
#                     detected_move = candidate_moves[0]
#                     move_type = "CENTRAL_GRID"
#                     break
        
#         # PRIORITY 2: Standard move on central grids (both source and dest detected)
#         if (len(vanished) == 1 and len(appeared) == 1 and
#             vanished[0] in vanished_central and appeared[0] in appeared_central and
#             detected_move is None):
            
#             src_idx = vanished[0]
#             dst_idx = appeared[0]
            
#             src_sq = self._to_chess_square(src_idx)
#             dst_sq = self._to_chess_square(dst_idx)
            
#             move = chess.Move(src_sq, dst_sq)
            
#             # Auto-Promote to Queen if pawn hits back rank
#             piece = self.board.piece_at(src_sq)
#             if piece and piece.piece_type == chess.PAWN:
#                 if chess.square_rank(dst_sq) in [0, 7]:
#                     move = chess.Move(src_sq, dst_sq, promotion=chess.QUEEN)

#             if self.board.is_legal(move):
#                 detected_move = move
#                 move_type = "CENTRAL_MOVE"

#         # PRIORITY 3: Standard move anywhere (1 Piece Moves)
#         if len(vanished) == 1 and len(appeared) == 1 and detected_move is None:
#             src_idx = vanished[0]
#             dst_idx = appeared[0]
            
#             src_sq = self._to_chess_square(src_idx)
#             dst_sq = self._to_chess_square(dst_idx)
            
#             move = chess.Move(src_sq, dst_sq)
            
#             # Auto-Promote to Queen if pawn hits back rank
#             piece = self.board.piece_at(src_sq)
#             if piece and piece.piece_type == chess.PAWN:
#                 if chess.square_rank(dst_sq) in [0, 7]:
#                     move = chess.Move(src_sq, dst_sq, promotion=chess.QUEEN)

#             if self.board.is_legal(move):
#                 detected_move = move
#                 move_type = "STANDARD"

#         # PRIORITY 4: Capture (1 Vanishes, 0 Appear)
#         elif len(vanished) == 1 and len(appeared) == 0 and detected_move is None:
#             src_idx = vanished[0]
#             src_sq = self._to_chess_square(src_idx)
            
#             # Find legal capture from this square
#             candidate_moves = []
#             for move in self.board.legal_moves:
#                 if move.from_square == src_sq and self.board.is_capture(move):
#                     candidate_moves.append(move)
            
#             if len(candidate_moves) == 1:
#                 detected_move = candidate_moves[0]
#                 move_type = "CAPTURE"

#         # PRIORITY 5: Castling (2 Vanish, 2 Appear)
#         elif len(vanished) == 2 and len(appeared) == 2 and detected_move is None:
#             for move in self.board.legal_moves:
#                 if self.board.is_castling(move):
#                     detected_move = move
#                     move_type = "CASTLING"
#                     break

#         # --- MOVE CONFIRMATION LOGIC ---
#         if detected_move is not None:
#             # Check if this is the same move we saw last frame
#             if self.pending_move == detected_move:
#                 self.pending_move_count += 1
#                 print(f"[ChessEngine] 🔄 Confirming {move_type}: {self.board.san(detected_move)} ({self.pending_move_count}/{self.CONFIRMATION_FRAMES})")
                
#                 # If we've seen it enough times, execute it
#                 if self.pending_move_count >= self.CONFIRMATION_FRAMES:
#                     san = self.board.san(detected_move)
#                     print(f"[ChessEngine] ✅ CONFIRMED {move_type}: {san}")
#                     self.board.push(detected_move)
#                     self.counters.fill(0)
#                     self.pending_move = None
#                     self.pending_move_count = 0
#                     return self._get_display_map()
#             else:
#                 # New move detected, start tracking it
#                 self.pending_move = detected_move
#                 self.pending_move_count = 1
#                 print(f"[ChessEngine] 👁️  Detected {move_type}: {self.board.san(detected_move)} (1/{self.CONFIRMATION_FRAMES})")
#         else:
#             # No valid move detected, reset pending
#             if self.pending_move is not None:
#                 print(f"[ChessEngine] ❌ Pending move lost")
#             self.pending_move = None
#             self.pending_move_count = 0

#         # Return current engine state (no changes yet)
#         return self._get_display_map()

#     def _get_display_map(self):
#         """
#         Returns a dictionary mapping camera indices to piece symbols.
#         White pieces: uppercase (K, Q, R, B, N, P)
#         Black pieces: lowercase (k, q, r, b, n, p)
#         """
#         display = {}
#         for sq in chess.SQUARES:
#             piece = self.board.piece_at(sq)
#             if piece:
#                 idx = self._to_camera_index(sq)
#                 # Get piece symbol (e.g., 'R' for white rook, 'r' for black rook)
#                 display[idx] = piece.symbol()
#         return display
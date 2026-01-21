from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import cv2

VALID_PIECES = set("PNBRQKpnbrqk")

@dataclass
class TileState:
    piece: Optional[str] = None
    conf: float = 0.0
    last_seen: int = -1
    pending_piece: Optional[str] = None 
    pending_count: int = 0

class ChessboardState:
    def __init__(self,
                 accept_k: int = 2,
                 min_obs_conf: float = 0.35,
                 inc: float = 0.20,
                 dec: float = 0.12,
                 miss_decay: float = 0.02,
                 clear_after_frames: int = 30,
                 clear_below_conf: float = 0.10,
    ):
        self.tiles: List[TileState] = [TileState() for _ in range(64)]
        self.accept_k = accept_k
        self.min_obs_conf = min_obs_conf
        self.inc = inc
        self.dec = dec
        self.miss_decay = miss_decay
        self.clear_after_frames = clear_after_frames
        self.clear_below_conf = clear_below_conf
        
    def init_standard(self):
        fen_rows = [
            "rnbqkbnr",
            "pppppppp",
            "8",
            "8",
            "8",
            "8",
            "PPPPPPPP",
            "RNBQKBNR",
        ]

        self.tiles = [TileState() for _ in range(64)]

        idx = 0
        for row in fen_rows:
            for ch in row:
                if ch.isdigit():
                    idx += int(ch)
                else:
                    self.tiles[idx].piece = ch
                    self.tiles[idx].conf = 0.95
                    self.tiles[idx].last_seen = 0
                    idx += 1


    def reset(self):
        self.tiles = [TileState() for _ in range(64)]

    def set_from_observation(self, obs, frame_idx: int = 0):
        """
        obs: {tile_idx: (fen_char or None, conf)}
        Hard set: state becomes exactly obs.

        ONLY FOR TESTING
        """
        for i in range(64):
            piece, conf = obs.get(i, (None, 0.0))
            self.tiles[i].piece = piece
            self.tiles[i].conf = float(conf)
            self.tiles[i].last_seen = frame_idx if piece is not None else -1
            self.tiles[i].pending_piece = None
            self.tiles[i].pending_count = 0


    def update(self,
               obs: Dict[int, Tuple[Optional[str],float]],
               frame_idx: int):
        """
        obs: {tile_idx: (fen_char or None, obs_conf)}
             IMPORTANT: passing None means "unknown", not "empty".
             We do NOT delete pieces from one observation.
        """
        for t in range(64):
            st = self.tiles[t]

            if t in obs:
                new_p, oconf = obs[t]

                if new_p is None:
                    if st.piece is not None:
                        st.last_seen = frame_idx 
                        st.conf = max(0.0, st.conf - 0.25 * oconf)
                        if st.conf < 0.15:
                            st.piece = None
                            st.conf = 0.0
                            st.pending_piece = None
                            st.pending_count = 0
                    continue
                if new_p not in VALID_PIECES:
                    continue

                oconf = float(oconf)

                if oconf < self.min_obs_conf:
                    continue 
            
                if st.piece == new_p:
                    st.conf = min(1.0, st.conf + self.inc * oconf)
                    st.last_seen = frame_idx
                    st.pending_piece = None
                    st.pending_count = 0

                else:
                    if st.pending_piece == new_p:
                        st.pending_count += 1
                    else:
                        st.pending_piece = new_p
                        st.pending_count = 1

                    st.conf = max(0.0, st.conf - self.dec * oconf)

                    if st.pending_count >= self.accept_k or st.conf < 0.25:
                        st.piece = new_p
                        st.conf = max(0.35, oconf)
                        st.last_seen = frame_idx
                        st.pending_piece = None
                        st.pending_count = 0
            
            else:
                # not observed: decay slowly
                if st.piece is not None:
                    st.conf = max(0.0, st.conf - self.miss_decay)

                    if (
                        st.conf < self.clear_below_conf
                        and st.last_seen >= 0
                        and (frame_idx - st.last_seen) > self.clear_after_frames
                    ):
                        st.piece = None
                        st.conf = 0.0
                        st.pending_piece = None
                        st.pending_count = 0
            

    def get_piece(self, tile_idx: int) -> Optional[str]:
        return self.tiles[tile_idx].piece

    def get_conf(self, tile_idx: int) -> float:
        return self.tiles[tile_idx].conf
    
    def to_fen_placement(self, tile_to_board: str = "row-major-top-left") -> str:
        """
        Returns ONLY the piece placement part of FEN (8 ranks separated by '/').
        To be used before passing through Stockfish


        tile_to_board:
          - "row-major-top-left": assumes tile_idx 0 is top-left, increases left->right, top->bottom.
        """
        if tile_to_board != "row-major-top-left":
            raise ValueError("Unsupported mapping for now")

        rows = []
        for r in range(8):
            empty = 0
            s = ""
            for c in range(8):
                idx = r * 8 + c
                p = self.tiles[idx].piece
                if p is None or p not in VALID_PIECES:
                    empty += 1
                else:
                    if empty > 0:
                        s += str(empty)
                        empty = 0
                    s += p
            if empty > 0:
                s += str(empty)
            rows.append(s)
        return "/".join(rows)
    
    def draw_overlay_visible(self, vis_img_bgr, tile_centers_global, obs):
        for idx, (piece, conf) in obs.items():
            if piece is None:
                continue

            x, y = tile_centers_global[idx].astype(int)
            txt = piece  

            cv2.putText(vis_img_bgr, txt, (x - 10, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(vis_img_bgr, txt, (x - 10, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    def enforce_max_counts(self, max_counts):
        by_piece = {p: [] for p in max_counts}
        for t, st in enumerate(self.tiles):
            p = st.piece
            if p in by_piece:
                by_piece[p].append(t)

        for p, tiles in by_piece.items():
            limit = max_counts[p]
            if len(tiles) <= limit:
                continue

            tiles_sorted = sorted(tiles, key=lambda t: self.tiles[t].conf)


            to_remove = tiles_sorted[: len(tiles) - limit]
            for t in to_remove:
                st = self.tiles[t]
                st.piece = None
                st.conf = 0.0
                st.last_seen = -1
                st.pending_piece = None
                st.pending_count = 0

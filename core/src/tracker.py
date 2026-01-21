from typing import Dict, List, Tuple
from .ChessboardState import ChessboardState

MAX_COUNTS = {
    'k': 1, 'K': 1,
    'q': 1, 'Q': 1,
    'p': 8, 'P': 8,
    'r': 2, 'R': 2,
    'n': 2, 'N': 2,
    'b': 2, 'B': 2,
}


def resolve_with_constraints(
    tile_cands: Dict[int, List[Tuple[str, float]]],
    min_keep: float = 0.35,
) -> Tuple[Dict[int, str], Dict[int, float]]:
    
    # 1) pick best per tile
    assign: Dict[int, str] = {}
    score: Dict[int, float] = {}

    for t, cands in tile_cands.items():
        if not cands:
            continue
        p, conf = cands[0]
        if conf >= min_keep:
            assign[t] = p
            score[t] = float(conf)

    def counts(assign_):
        c = {k: 0 for k in MAX_COUNTS}
        for p in assign_.values():
            if p in c:
                c[p] += 1
        return c

    # 2) repair overflow 
    while True:
        c = counts(assign)
        overflow = [p for p, n in c.items() if n > MAX_COUNTS[p]]
        if not overflow:
            break

        p_bad = overflow[0]
        tiles = [t for t, p in assign.items() if p == p_bad]

        best_fix = None  # (delta_loss, t, new_p_or_None, new_conf)

        for t in tiles:
            cur_conf = score[t]
            cands = tile_cands.get(t, [])

            # remove piece from that tile
            delta_remove = cur_conf
            cand_fix = (delta_remove, t, None, 0.0)
            if best_fix is None or cand_fix[0] < best_fix[0]:
                best_fix = cand_fix

            # change for another candidate
            for j in range(1, min(len(cands), 4)):
                p2, conf2 = cands[j]
                if conf2 < min_keep:
                    continue
                if c.get(p2, 0) + 1 > MAX_COUNTS.get(p2, 99):
                    continue
                delta = cur_conf - conf2
                cand_fix = (delta, t, p2, float(conf2))
                if best_fix is None or cand_fix[0] < best_fix[0]:
                    best_fix = cand_fix

        if best_fix is None:
            break

        _, t_fix, p_new, conf_new = best_fix
        if p_new is None:
            assign.pop(t_fix, None)
            score.pop(t_fix, None)
        else:
            assign[t_fix] = p_new
            score[t_fix] = conf_new

    return assign, score


class BoardTracker:
    """
      - it applies global constraints (MAX_COUNTS)
      - it compares obs vs memory
      - decides if accepts or not any frame
      - call to board.update
    """

    def __init__(
        self,
        board: ChessboardState,
        max_changed_tiles: int = 4,
        diff_conf_thr: float = 0.60,
        min_keep: float = 0.35,
    ):
        self.board = board
        self.max_changed_tiles = max_changed_tiles
        self.diff_conf_thr = diff_conf_thr
        self.min_keep = min_keep
        self.max_counts = MAX_COUNTS

    def _board_diff_tiles(
        self,
        obs: Dict[int, Tuple[str, float]],
    ) -> List[int]:
        diffs: List[int] = []
        for t in range(64):
            obs_p, obs_c = obs.get(t, (None, 0.0))
            mem_p = self.board.tiles[t].piece

            if obs_p is None:
                continue #unknown 

            if obs_c < self.diff_conf_thr:
                continue  

            if mem_p != obs_p:
                diffs.append(t)
        return diffs

    def step(
        self,
        tile_cands: Dict[int, List[Tuple[str, float]]],
        frame_idx: int,
    ) -> Tuple[bool, Dict[int, Tuple[str, float]], List[int]]:

        if not tile_cands:
            return False, {}, []

        assign, score_map = resolve_with_constraints(tile_cands, min_keep=self.min_keep)

        obs: Dict[int, Tuple[str, float]] = {
            t: (p, float(score_map[t])) for t, p in assign.items()
        }

        diffs = self._board_diff_tiles(obs)

        if len(diffs) <= self.max_changed_tiles:
            self.board.update(obs, frame_idx)
            self.board.enforce_max_counts(self.max_counts)
            accept_update = True
        else:
            accept_update = False

        return accept_update, obs, diffs

from stockfish import Stockfish


class StockfishModule:
    def __init__(self,
                 path_stockfish: str,
                 elo: int = 1500,
                 thr_frame: int = 10):
    
        
        self.path_stockfish = path_stockfish
        self.elo = elo
        self.thr_frame = thr_frame

        self.sf = Stockfish(path=self.path_stockfish)

        self.last_good_fen = None
        self.has_position = False

        self.best_move = None
        self.last_frame = 0
        
        try:
            self.sf.set_elo_rating(self.elo)
        except Exception:
            pass

    def set_position_from_state(self,state,side="w",castling="KQkq",ep="-",halfmove=0,fullmove=1):
        placement = state.to_fen_placement()
        fen = f"{placement} {side} {castling} {ep} {halfmove} {fullmove}"
        
        try:
            if self.sf.is_fen_valid(fen):
                self.sf.set_fen_position(fen)
                self.last_good_fen = fen
                self.has_position = True
                return True
        except Exception:
            pass

        self.has_position = self.last_good_fen is not None
        return False

    
    def get_best_move(self,actual_frame):
        #it returns None if there is no possible move
        if not self.has_position:
            return None

        need_refresh = (abs(self.last_frame - actual_frame) > self.thr_frame)
        
        if self.best_move is None:
            need_refresh = True
        else:
            try:
                if not self.sf.is_move_correct(self.best_move):
                    need_refresh = True
            except Exception:
                need_refresh = True

        if need_refresh:
            try:
                self.best_move = self.sf.get_best_move()  # can return None
            except Exception:
                return None
            self.last_frame = actual_frame

        return self.best_move
    

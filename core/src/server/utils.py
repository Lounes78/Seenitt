# src/server/utils.py

def print_ascii_board(tile_to_piece, frame_id):
    """Prints the 8x8 grid state to the terminal."""
    print(f"\n--- Frame {frame_id} Board State ---")
    print("   a b c d e f g h")
    print("  +-----------------+")
    for row in range(8):
        row_str = f"{8 - row} | " 
        for col in range(8):
            tile_idx = row * 8 + col
            symbol = "X" if tile_idx in tile_to_piece else "."
            row_str += f"{symbol} "
        row_str += f"| {8 - row}"
        print(row_str)
    print("  +-----------------+")
    print("   a b c d e f g h\n")

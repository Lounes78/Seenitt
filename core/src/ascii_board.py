def print_ascii_board(tile_to_piece, frame_id):
    """
    Prints the 8x8 grid state to the terminal.
    tile_to_piece: dict {tile_idx (0-63): piece_data}
    """
    print(f"\n--- Frame {frame_id} Board State ---")
    print("   a b c d e f g h")
    print("  +-----------------+")
    
    # Iterate rows 0 to 7
    for row in range(8):
        row_str = f"{8 - row} | " # Rank label (8 down to 1)
        for col in range(8):
            tile_idx = row * 8 + col
            
            # Check if this tile has a piece
            if tile_idx in tile_to_piece:
                # 'X' = Occupied, '.' = Empty
                symbol = "X" 
            else:
                symbol = "."
            
            row_str += f"{symbol} "
        
        row_str += f"| {8 - row}"
        print(row_str)
    
    print("  +-----------------+")
    print("   a b c d e f g h\n")

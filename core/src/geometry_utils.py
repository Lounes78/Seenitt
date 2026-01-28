import numpy as np

def correct_orientation(grid_points, tile_centers):
    """
    Forces the grid to align with gravity and standard reading order.
    Goal: 
      - Logical Row 0 should be at the visual TOP (Low Y).
      - Logical Col 0 should be at the visual LEFT (Low X).
    """
    g_reshaped = grid_points.reshape(9, 9, 2)
    c_reshaped = tile_centers.reshape(8, 8, 2)
    
    best_grid = None
    best_centers = None
    
    # We want to MINIMIZE the Y-coordinate of the first row (Top Row)
    # AND MINIMIZE the X-coordinate of the first column (Left Column)
    min_score = float('inf')

    # Try all 4 rotations (0, 90, 180, 270 degrees)
    for k in range(4):
        curr_g = np.rot90(g_reshaped, k)
        curr_c = np.rot90(c_reshaped, k)
        
        # 1. Check "Horizontal-ness"
        # The Top Row (Index 0-8) should be roughly horizontal.
        # So the variance in Y should be low.
        top_row_y = curr_g[0, :, 1]
        y_variance = np.var(top_row_y)
        
        # 2. Check "Top-ness"
        # The average Y of the top row should be small (close to 0)
        avg_y = np.mean(top_row_y)
        
        # 3. Check "Left-ness"
        # The average X of the first column should be small
        first_col_x = curr_g[:, 0, 0]
        avg_x = np.mean(first_col_x)

        # Composite Score:
        # We heavily penalize vertical orientation (high variance in Y for a row)
        # Then we penalize being at the bottom or right.
        score = (y_variance * 1000) + (avg_y * 10) + avg_x
        
        if score < min_score:
            min_score = score
            best_grid = curr_g
            best_centers = curr_c

    # Flatten back
    return np.ascontiguousarray(best_grid.reshape(-1, 2)), np.ascontiguousarray(best_centers.reshape(-1, 2))
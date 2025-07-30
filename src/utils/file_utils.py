"""File utility functions for the plant recognition pipeline."""

import os
from pathlib import Path
from typing import List, Optional
from PIL import Image


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_extension(filename: str) -> str:
    """Get file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (without dot)
    """
    return Path(filename).suffix.lower().lstrip('.')


def validate_image_file(file_path: str) -> bool:
    """Validate if file is a valid image.
    
    Args:
        file_path: Path to image file
        
    Returns:
        True if valid image, False otherwise
    """
    valid_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}
    
    if not Path(file_path).exists():
        return False
    
    extension = get_file_extension(file_path)
    if extension not in valid_extensions:
        return False
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_files(directory: str, recursive: bool = True) -> List[str]:
    """Get list of image files in directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if recursive:
        for ext in valid_extensions:
            image_files.extend(directory.rglob(f"*{ext}"))
            image_files.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in valid_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return [str(f) for f in image_files if validate_image_file(str(f))]


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    return safe_name
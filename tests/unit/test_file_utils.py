"""Unit tests for file utilities."""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

from src.utils.file_utils import (
    ensure_dir,
    get_file_extension,
    validate_image_file,
    get_image_files,
    safe_filename
)


class TestFileUtils:
    """Test file utility functions."""
    
    def test_ensure_dir(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "new" / "nested" / "directory"
            
            # Directory shouldn't exist initially
            assert not test_path.exists()
            
            # Create directory
            result_path = ensure_dir(str(test_path))
            
            # Check directory was created
            assert test_path.exists()
            assert test_path.is_dir()
            assert result_path == test_path
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension("image.jpg") == "jpg"
        assert get_file_extension("image.JPEG") == "jpeg"
        assert get_file_extension("document.pdf") == "pdf"
        assert get_file_extension("file_with_no_extension") == ""
        assert get_file_extension("/path/to/image.png") == "png"
    
    def test_validate_image_file(self):
        """Test image file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a valid image file
            valid_image_path = temp_path / "valid_image.jpg"
            img = Image.new('RGB', (100, 100), color='red')
            img.save(valid_image_path)
            
            # Create an invalid file with image extension
            invalid_image_path = temp_path / "invalid_image.jpg"
            with open(invalid_image_path, 'w') as f:
                f.write("not an image")
            
            # Create a text file
            text_file_path = temp_path / "text_file.txt"
            with open(text_file_path, 'w') as f:
                f.write("hello world")
            
            # Test validation
            assert validate_image_file(str(valid_image_path)) == True
            assert validate_image_file(str(invalid_image_path)) == False
            assert validate_image_file(str(text_file_path)) == False
            assert validate_image_file("nonexistent_file.jpg") == False
    
    def test_get_image_files(self):
        """Test image file discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            subdir = temp_path / "subdir"
            subdir.mkdir()
            
            # Create various files
            files_to_create = [
                temp_path / "image1.jpg",
                temp_path / "image2.PNG",
                temp_path / "document.pdf",
                temp_path / "text.txt",
                subdir / "image3.jpeg",
                subdir / "image4.bmp"
            ]
            
            # Create valid images
            for file_path in files_to_create:
                if file_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
                    img = Image.new('RGB', (50, 50), color='blue')
                    img.save(file_path)
                else:
                    with open(file_path, 'w') as f:
                        f.write("content")
            
            # Test recursive search
            image_files = get_image_files(str(temp_path), recursive=True)
            image_files = [Path(f).name for f in image_files]
            
            expected_images = {"image1.jpg", "image2.PNG", "image3.jpeg", "image4.bmp"}
            actual_images = set(image_files)
            
            assert expected_images == actual_images
            
            # Test non-recursive search
            image_files_non_recursive = get_image_files(str(temp_path), recursive=False)
            image_files_non_recursive = [Path(f).name for f in image_files_non_recursive]
            
            expected_non_recursive = {"image1.jpg", "image2.PNG"}
            actual_non_recursive = set(image_files_non_recursive)
            
            assert expected_non_recursive == actual_non_recursive
    
    def test_safe_filename(self):
        """Test safe filename generation."""
        # Test with invalid characters
        unsafe_name = 'file<name>with:invalid"chars/and\\pipes|?*'
        safe_name = safe_filename(unsafe_name)
        
        # All invalid characters should be replaced with underscores
        expected = 'file_name_with_invalid_chars_and_pipes___'
        assert safe_name == expected
        
        # Test with already safe filename
        already_safe = 'already_safe_filename.jpg'
        assert safe_filename(already_safe) == already_safe
        
        # Test with empty string
        assert safe_filename('') == ''
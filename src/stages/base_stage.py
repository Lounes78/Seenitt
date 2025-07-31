"""Base class for all processing stages."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from ..utils import setup_logger


class BaseStage(ABC):
    """Base class for all processing stages in the pipeline."""
    
    def __init__(self, config: Dict[str, Any], stage_name: str):
        """Initialize base stage.
        
        Args:
            config: Configuration dictionary
            stage_name: Name of the processing stage
        """
        self.config = config
        self.stage_name = stage_name
        self.logger = setup_logger(f"Stage_{stage_name}")
        
    @abstractmethod
    def process(self, input_data: Any, output_dir: str) -> Dict[str, Any]:
        """Process input data and return results.
        
        Args:
            input_data: Input data for processing
            output_dir: Directory to save output files
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> str:
        """Save results to JSON file.
        
        Args:
            results: Results dictionary to save
            output_dir: Output directory
            
        Returns:
            Path to saved results file
        """
        output_path = Path(output_dir) / f"{self.stage_name}_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = self._make_json_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        return str(output_path)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # NumPy scalar
            return obj.item()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        else:
            return obj
    
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load results from JSON file.
        
        Args:
            results_file: Path to results file
            
        Returns:
            Loaded results dictionary
        """
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True  # Override in subclasses for specific validation
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for this stage.
        
        Returns:
            Dictionary of metrics
        """
        return {}  # Override in subclasses to provide metrics
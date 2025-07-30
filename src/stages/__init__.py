"""Processing stages for the smart glasses plant recognition pipeline."""

from .stage1_detection import YOLODetectionStage
from .stage2_filtering import TrackingIDFilterStage
from .stage3_intra_track import IntraTrackFilterStage
from .stage4_similarity import SimilarityDetectionStage
from .stage5_quality import QualityAssessmentStage
from .stage6_validation import FinalValidationStage

__all__ = [
    'YOLODetectionStage',
    'TrackingIDFilterStage',
    'IntraTrackFilterStage',
    'SimilarityDetectionStage',
    'QualityAssessmentStage',
    'FinalValidationStage'
]
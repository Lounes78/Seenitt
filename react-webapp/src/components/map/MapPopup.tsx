import React, { useCallback } from 'react';
import { ImageData } from '../../types/image';
import { formatRelativeTime, formatConfidence, formatCoordinates, formatFileSize } from '../../utils/formatters';
import { STATUS_COLORS, MAP_SETTINGS } from '../../utils/constants';

interface MapPopupProps {
  image: ImageData;
  onClose: () => void;
  onViewImage: (image: ImageData) => void;
  onViewGallery?: () => void;
  position?: { x: number; y: number };
  className?: string;
}

export const MapPopup: React.FC<MapPopupProps> = ({
  image,
  onClose,
  onViewImage,
  onViewGallery,
  position,
  className = ''
}) => {
  const handleViewImage = useCallback(() => {
    onViewImage(image);
  }, [onViewImage, image]);

  const handleViewGallery = useCallback(() => {
    onViewGallery?.();
  }, [onViewGallery]);

  const handleBackdropClick = useCallback((e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  }, [onClose]);

  return (
    <div 
      className={`map-popup ${className}`}
      onClick={handleBackdropClick}
      style={{
        position: position ? 'fixed' : 'absolute',
        left: position?.x,
        top: position?.y,
        zIndex: MAP_SETTINGS.popupMaxWidth
      }}
    >
      <div className="popup-container">
        {/* Header */}
        <div className="popup-header">
          <div className="popup-title">
            <span className="title-text">Image #{image.id.slice(-8)}</span>
            <div 
              className="status-badge"
              style={{ backgroundColor: STATUS_COLORS[image.status] }}
            >
              {image.status}
            </div>
          </div>
          
          <button 
            className="popup-close"
            onClick={onClose}
            title="Close popup"
          >
            
          </button>
        </div>

        {/* Image preview */}
        <div className="popup-image">
          <img
            src={image.thumbnailUrl || image.imageUrl}
            alt={`Image from ${formatRelativeTime(image.createdAt)}`}
            className="popup-image-thumb"
            onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
              const container = target.parentElement;
              if (container) {
                container.innerHTML = `
                  <div class="image-error">
                    <span class="error-icon">=�</span>
                    <span class="error-text">Image not available</span>
                  </div>
                `;
              }
            }}
          />
          
          {/* Image overlay info */}
          <div className="image-overlay">
            {image.metadata.confidence !== undefined && (
              <div className="overlay-badge overlay-badge--confidence">
                {formatConfidence(image.metadata.confidence)}
              </div>
            )}
            
            {image.metadata.objects && image.metadata.objects.length > 0 && (
              <div className="overlay-badge overlay-badge--objects">
                = {image.metadata.objects.length}
              </div>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="popup-content">
          {/* Basic info */}
          <div className="info-section">
            <div className="info-item">
              <span className="info-label">Created:</span>
              <span className="info-value">{formatRelativeTime(image.createdAt)}</span>
            </div>
            
            {image.location && (
              <div className="info-item">
                <span className="info-label">Location:</span>
                <span className="info-value">
                  {formatCoordinates(image.location.latitude, image.location.longitude)}
                </span>
              </div>
            )}
            
            {image.metadata.fileSize && (
              <div className="info-item">
                <span className="info-label">Size:</span>
                <span className="info-value">{formatFileSize(image.metadata.fileSize)}</span>
              </div>
            )}
            
            {image.metadata.imageSize && (
              <div className="info-item">
                <span className="info-label">Dimensions:</span>
                <span className="info-value">
                  {image.metadata.imageSize.width} � {image.metadata.imageSize.height}
                </span>
              </div>
            )}
          </div>

          {/* Detected objects */}
          {image.metadata.objects && image.metadata.objects.length > 0 && (
            <div className="info-section">
              <h4 className="section-title">Detected Objects</h4>
              <div className="objects-list">
                {image.metadata.objects.slice(0, 4).map((obj, index) => (
                  <div key={index} className="object-item">
                    <span className="object-name">{obj.name}</span>
                    <span className="object-confidence">
                      {formatConfidence(obj.confidence)}
                    </span>
                  </div>
                ))}
                {image.metadata.objects.length > 4 && (
                  <div className="objects-more">
                    +{image.metadata.objects.length - 4} more objects
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Tags */}
          {image.metadata.tags && image.metadata.tags.length > 0 && (
            <div className="info-section">
              <h4 className="section-title">Tags</h4>
              <div className="tags-list">
                {image.metadata.tags.slice(0, 6).map((tag, index) => (
                  <span key={index} className="tag">
                    {tag}
                  </span>
                ))}
                {image.metadata.tags.length > 6 && (
                  <span className="tag tag--more">
                    +{image.metadata.tags.length - 6}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Location details */}
          {image.location && (
            <div className="info-section">
              <h4 className="section-title">Location Details</h4>
              <div className="location-details">
                <div className="info-item">
                  <span className="info-label">Latitude:</span>
                  <span className="info-value">{image.location.latitude.toFixed(6)}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Longitude:</span>
                  <span className="info-value">{image.location.longitude.toFixed(6)}</span>
                </div>
                {image.location.accuracy && (
                  <div className="info-item">
                    <span className="info-label">Accuracy:</span>
                    <span className="info-value">{image.location.accuracy}m</span>
                  </div>
                )}
                {image.location.altitude && (
                  <div className="info-item">
                    <span className="info-label">Altitude:</span>
                    <span className="info-value">{image.location.altitude}m</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="popup-actions">
          <button 
            className="action-btn action-btn--primary"
            onClick={handleViewImage}
          >
            View Full Image
          </button>
          
          {onViewGallery && (
            <button 
              className="action-btn action-btn--secondary"
              onClick={handleViewGallery}
            >
              Gallery View
            </button>
          )}
        </div>
      </div>

      <style>{`
        .map-popup {
          position: absolute;
          z-index: 1100;
          pointer-events: auto;
        }

        .popup-container {
          background: white;
          border-radius: 12px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
          border: 1px solid #e9ecef;
          width: 320px;
          max-height: 500px;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .popup-header {
          padding: 16px 20px;
          border-bottom: 1px solid #e9ecef;
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: #f8f9fa;
        }

        .popup-title {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .title-text {
          font-size: 16px;
          font-weight: 600;
          color: #2d3748;
        }

        .status-badge {
          padding: 4px 10px;
          border-radius: 12px;
          color: white;
          font-size: 11px;
          font-weight: 500;
          text-transform: uppercase;
        }

        .popup-close {
          width: 28px;
          height: 28px;
          border: none;
          border-radius: 6px;
          background: #e9ecef;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 14px;
          color: #6c757d;
          transition: all 0.2s;
        }

        .popup-close:hover {
          background: #dc3545;
          color: white;
        }

        .popup-image {
          position: relative;
          height: 150px;
          overflow: hidden;
          background: #f8f9fa;
        }

        .popup-image-thumb {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .image-error {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          color: #6c757d;
        }

        .error-icon {
          font-size: 32px;
          margin-bottom: 8px;
        }

        .error-text {
          font-size: 14px;
        }

        .image-overlay {
          position: absolute;
          top: 8px;
          right: 8px;
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .overlay-badge {
          background: rgba(0, 0, 0, 0.7);
          color: white;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 11px;
          font-weight: 500;
        }

        .overlay-badge--confidence {
          background: rgba(40, 167, 69, 0.9);
        }

        .overlay-badge--objects {
          background: rgba(0, 123, 255, 0.9);
        }

        .popup-content {
          flex: 1;
          padding: 16px 20px;
          overflow-y: auto;
          max-height: 250px;
        }

        .info-section {
          margin-bottom: 16px;
        }

        .info-section:last-child {
          margin-bottom: 0;
        }

        .section-title {
          font-size: 14px;
          font-weight: 600;
          color: #2d3748;
          margin: 0 0 8px 0;
        }

        .info-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
          font-size: 13px;
        }

        .info-label {
          color: #6c757d;
          font-weight: 500;
        }

        .info-value {
          color: #2d3748;
          text-align: right;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          max-width: 60%;
        }

        .objects-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .object-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 4px 8px;
          background: #f8f9fa;
          border-radius: 4px;
          font-size: 12px;
        }

        .object-name {
          color: #2d3748;
          font-weight: 500;
        }

        .object-confidence {
          color: #6c757d;
          font-size: 11px;
        }

        .objects-more {
          font-size: 11px;
          color: #6c757d;
          font-style: italic;
          text-align: center;
          padding: 4px;
        }

        .tags-list {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }

        .tag {
          background: #007bff;
          color: white;
          padding: 3px 8px;
          border-radius: 10px;
          font-size: 11px;
          font-weight: 500;
        }

        .tag--more {
          background: #6c757d;
        }

        .location-details {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .popup-actions {
          padding: 16px 20px;
          border-top: 1px solid #e9ecef;
          display: flex;
          gap: 8px;
        }

        .action-btn {
          flex: 1;
          padding: 10px 16px;
          border: none;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .action-btn--primary {
          background: #007bff;
          color: white;
        }

        .action-btn--primary:hover {
          background: #0056b3;
          transform: translateY(-1px);
        }

        .action-btn--secondary {
          background: #e9ecef;
          color: #495057;
          border: 1px solid #dee2e6;
        }

        .action-btn--secondary:hover {
          background: #dee2e6;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
          .popup-container {
            width: 280px;
            max-height: 400px;
          }

          .popup-header {
            padding: 12px 16px;
          }

          .popup-content {
            padding: 12px 16px;
            max-height: 200px;
          }

          .popup-actions {
            padding: 12px 16px;
            flex-direction: column;
          }

          .info-item {
            flex-direction: column;
            align-items: flex-start;
            gap: 2px;
          }

          .info-value {
            text-align: left;
            max-width: 100%;
          }
        }

        /* Animation */
        .popup-container {
          animation: popupSlideIn 0.3s ease-out;
        }

        @keyframes popupSlideIn {
          from {
            opacity: 0;
            transform: translateY(20px) scale(0.9);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
      `}</style>
    </div>
  );
};
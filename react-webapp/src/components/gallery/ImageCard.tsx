import React, { useState, useCallback } from 'react';
import { ImageData } from '../../types/image';
import { formatRelativeTime, formatConfidence, formatFileSize, formatObjectCount } from '../../utils/formatters';
import { STATUS_COLORS } from '../../utils/constants';

interface ImageCardProps {
  image: ImageData;
  size: 'small' | 'medium' | 'large';
  viewMode: 'grid' | 'list';
  selected: boolean;
  selectionMode: boolean;
  onClick: () => void;
  onDoubleClick: () => void;
  className?: string;
}

export const ImageCard: React.FC<ImageCardProps> = ({
  image,
  size,
  viewMode,
  selected,
  selectionMode,
  onClick,
  onDoubleClick,
  className = ''
}) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
  }, []);

  const handleImageError = useCallback(() => {
    setImageError(true);
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onClick();
    }
  }, [onClick]);

  const sizeConfig = {
    small: { height: 150, padding: 8 },
    medium: { height: 200, padding: 12 },
    large: { height: 250, padding: 16 }
  };

  const currentConfig = sizeConfig[size];
  const isListView = viewMode === 'list';

  return (
    <div 
      className={`image-card ${className} ${selected ? 'image-card--selected' : ''} ${selectionMode ? 'image-card--selection-mode' : ''} ${isListView ? 'image-card--list' : 'image-card--grid'}`}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      role="button"
      aria-label={`Image ${image.id} - ${formatRelativeTime(image.createdAt)}`}
      style={{
        height: isListView ? 'auto' : currentConfig.height,
        padding: currentConfig.padding
      }}
    >
      {/* Selection checkbox */}
      {selectionMode && (
        <div className="image-card__selection">
          <input
            type="checkbox"
            checked={selected}
            onChange={() => {}} // Handled by parent onClick
            tabIndex={-1}
            aria-hidden="true"
          />
        </div>
      )}

      <div className="image-card__content">
        {/* Image container */}
        <div className="image-card__image-container">
          {!imageLoaded && !imageError && (
            <div className="image-card__loading">
              <div className="loading-spinner"></div>
            </div>
          )}

          {imageError ? (
            <div className="image-card__error">
              <span className="error-icon">�</span>
              <span className="error-text">Failed to load</span>
            </div>
          ) : (
            <img
              src={image.thumbnailUrl || image.imageUrl}
              alt={`Processed image from ${formatRelativeTime(image.createdAt)}`}
              className="image-card__image"
              onLoad={handleImageLoad}
              onError={handleImageError}
              style={{
                opacity: imageLoaded ? 1 : 0,
                transition: 'opacity 0.3s ease'
              }}
            />
          )}

          {/* Status indicator */}
          <div 
            className="image-card__status"
            style={{ backgroundColor: STATUS_COLORS[image.status] }}
          >
            {image.status === 'processing' && '�'}
            {image.status === 'completed' && ''}
            {image.status === 'error' && 'L'}
            {image.status === 'pending' && '�'}
            {image.status === 'cancelled' && '=�'}
          </div>

          {/* Confidence badge */}
          {image.metadata.confidence !== undefined && (
            <div className="image-card__confidence">
              {formatConfidence(image.metadata.confidence)}
            </div>
          )}

          {/* Location indicator */}
          {image.location && (
            <div className="image-card__location">
              =�
            </div>
          )}

          {/* Object count */}
          {image.metadata.objects && image.metadata.objects.length > 0 && (
            <div className="image-card__objects">
              = {image.metadata.objects.length}
            </div>
          )}
        </div>

        {/* Metadata (for list view or large grid) */}
        {(isListView || size === 'large') && (
          <div className="image-card__metadata">
            <div className="image-card__title">
              Image #{image.id.slice(-8)}
            </div>

            <div className="image-card__details">
              <div className="detail-item">
                <span className="detail-label">Time:</span>
                <span className="detail-value">{formatRelativeTime(image.createdAt)}</span>
              </div>

              {image.metadata.confidence !== undefined && (
                <div className="detail-item">
                  <span className="detail-label">Confidence:</span>
                  <span className="detail-value">{formatConfidence(image.metadata.confidence)}</span>
                </div>
              )}

              {image.location && (
                <div className="detail-item">
                  <span className="detail-label">Location:</span>
                  <span className="detail-value">
                    {image.location.latitude.toFixed(4)}, {image.location.longitude.toFixed(4)}
                  </span>
                </div>
              )}

              {image.metadata.objects && image.metadata.objects.length > 0 && (
                <div className="detail-item">
                  <span className="detail-label">Objects:</span>
                  <span className="detail-value">{formatObjectCount(image.metadata.objects.length)}</span>
                </div>
              )}

              {image.metadata.fileSize && (
                <div className="detail-item">
                  <span className="detail-label">Size:</span>
                  <span className="detail-value">{formatFileSize(image.metadata.fileSize)}</span>
                </div>
              )}

              {image.metadata.imageSize && (
                <div className="detail-item">
                  <span className="detail-label">Dimensions:</span>
                  <span className="detail-value">
                    {image.metadata.imageSize.width} � {image.metadata.imageSize.height}
                  </span>
                </div>
              )}
            </div>

            {/* Tags */}
            {image.metadata.tags && image.metadata.tags.length > 0 && (
              <div className="image-card__tags">
                {image.metadata.tags.slice(0, 3).map((tag, index) => (
                  <span key={index} className="tag">
                    {tag}
                  </span>
                ))}
                {image.metadata.tags.length > 3 && (
                  <span className="tag tag--more">
                    +{image.metadata.tags.length - 3}
                  </span>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <style>{`
        .image-card {
          position: relative;
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          transition: all 0.2s ease;
          cursor: pointer;
          overflow: hidden;
          border: 2px solid transparent;
        }

        .image-card:hover {
          box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
          transform: translateY(-2px);
        }

        .image-card:focus {
          outline: none;
          border-color: #007bff;
        }

        .image-card--selected {
          border-color: #007bff;
          box-shadow: 0 4px 16px rgba(0, 123, 255, 0.3);
        }

        .image-card--selection-mode {
          cursor: default;
        }

        .image-card--list {
          display: flex;
          flex-direction: row;
          height: auto !important;
          padding: 16px !important;
        }

        .image-card--list .image-card__content {
          display: flex;
          flex-direction: row;
          gap: 16px;
          width: 100%;
        }

        .image-card--list .image-card__image-container {
          flex-shrink: 0;
          width: 120px;
          height: 80px;
        }

        .image-card--list .image-card__metadata {
          flex: 1;
          margin-top: 0;
        }

        .image-card__selection {
          position: absolute;
          top: 8px;
          left: 8px;
          z-index: 10;
          background: rgba(255, 255, 255, 0.9);
          border-radius: 4px;
          padding: 2px;
        }

        .image-card__content {
          height: 100%;
          display: flex;
          flex-direction: column;
        }

        .image-card__image-container {
          position: relative;
          flex: 1;
          min-height: 0;
          border-radius: 6px;
          overflow: hidden;
          background: #f8f9fa;
        }

        .image-card__loading {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
        }

        .loading-spinner {
          width: 24px;
          height: 24px;
          border: 2px solid #e9ecef;
          border-top: 2px solid #007bff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .image-card__error {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          text-align: center;
          color: #6c757d;
        }

        .error-icon {
          display: block;
          font-size: 24px;
          margin-bottom: 4px;
        }

        .error-text {
          font-size: 12px;
        }

        .image-card__image {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .image-card__status {
          position: absolute;
          top: 6px;
          right: 6px;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          color: white;
        }

        .image-card__confidence {
          position: absolute;
          bottom: 6px;
          right: 6px;
          background: rgba(0, 0, 0, 0.7);
          color: white;
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 500;
        }

        .image-card__location {
          position: absolute;
          bottom: 6px;
          left: 6px;
          background: rgba(0, 123, 255, 0.9);
          color: white;
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 11px;
        }

        .image-card__objects {
          position: absolute;
          top: 6px;
          left: 6px;
          background: rgba(40, 167, 69, 0.9);
          color: white;
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 500;
        }

        .image-card__metadata {
          margin-top: 8px;
        }

        .image-card__title {
          font-weight: 600;
          font-size: 14px;
          color: #2d3748;
          margin-bottom: 6px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .image-card__details {
          font-size: 12px;
          color: #6c757d;
          line-height: 1.4;
        }

        .detail-item {
          display: flex;
          justify-content: space-between;
          margin-bottom: 2px;
        }

        .detail-label {
          font-weight: 500;
        }

        .detail-value {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          max-width: 60%;
        }

        .image-card__tags {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          margin-top: 8px;
        }

        .tag {
          background: #e9ecef;
          color: #495057;
          padding: 2px 6px;
          border-radius: 12px;
          font-size: 10px;
          font-weight: 500;
        }

        .tag--more {
          background: #007bff;
          color: white;
        }

        /* Grid size variations */
        .image-card--grid.image-card {
          height: ${currentConfig.height}px;
          padding: ${currentConfig.padding}px;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
          .image-card--list {
            padding: 12px !important;
          }
          
          .image-card--list .image-card__content {
            gap: 12px;
          }
          
          .image-card--list .image-card__image-container {
            width: 80px;
            height: 60px;
          }
          
          .detail-item {
            flex-direction: column;
            align-items: flex-start;
          }
          
          .detail-value {
            max-width: 100%;
          }
        }
      `}</style>
    </div>
  );
};
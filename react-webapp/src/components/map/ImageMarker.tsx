import React, { useState, useCallback } from 'react';
import { ImageData } from '../../types/image';
import { formatRelativeTime, formatConfidence } from '../../utils/formatters';
import { STATUS_COLORS, MAP_SETTINGS } from '../../utils/constants';

interface ImageMarkerProps {
  image: ImageData;
  size?: 'small' | 'medium' | 'large';
  selected?: boolean;
  onClick?: (image: ImageData) => void;
  onDoubleClick?: (image: ImageData) => void;
  showPopup?: boolean;
  className?: string;
}

export const ImageMarker: React.FC<ImageMarkerProps> = ({
  image,
  size = 'medium',
  selected = false,
  onClick,
  onDoubleClick,
  showPopup = false,
  className = ''
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onClick?.(image);
  }, [onClick, image]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onDoubleClick?.(image);
  }, [onDoubleClick, image]);

  const handleMouseEnter = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
  }, []);

  const markerSize = MAP_SETTINGS.markerSize[size];
  const thumbnailSize = markerSize - 4; // Account for border

  if (!image.location) {
    return null;
  }

  return (
    <div
      className={`image-marker ${className} ${selected ? 'image-marker--selected' : ''} ${isHovered ? 'image-marker--hovered' : ''}`}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{
        width: markerSize,
        height: markerSize,
        transform: selected || isHovered ? 'scale(1.2)' : 'scale(1)',
        zIndex: selected ? 1000 : isHovered ? 999 : 1
      }}
    >
      {/* Marker container */}
      <div className="marker-container">
        {/* Thumbnail */}
        <div 
          className="marker-thumbnail"
          style={{
            width: thumbnailSize,
            height: thumbnailSize
          }}
        >
          <img
            src={image.thumbnailUrl || image.imageUrl}
            alt={`Image from ${formatRelativeTime(image.createdAt)}`}
            className="thumbnail-image"
            onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
              const parent = target.parentElement;
              if (parent) {
                parent.innerHTML = '=�';
                parent.style.display = 'flex';
                parent.style.alignItems = 'center';
                parent.style.justifyContent = 'center';
                parent.style.fontSize = `${thumbnailSize * 0.4}px`;
                parent.style.color = '#6c757d';
              }
            }}
          />
        </div>

        {/* Status indicator */}
        <div 
          className="marker-status"
          style={{ backgroundColor: STATUS_COLORS[image.status] }}
        >
          {image.status === 'processing' && '�'}
          {image.status === 'completed' && ''}
          {image.status === 'error' && 'L'}
          {image.status === 'pending' && '�'}
          {image.status === 'cancelled' && '=�'}
        </div>

        {/* Confidence indicator */}
        {image.metadata.confidence !== undefined && image.metadata.confidence > 0.7 && (
          <div className="marker-confidence">
            P
          </div>
        )}

        {/* Object count indicator */}
        {image.metadata.objects && image.metadata.objects.length > 0 && (
          <div className="marker-objects">
            {image.metadata.objects.length}
          </div>
        )}
      </div>

      {/* Popup tooltip */}
      {(showPopup || isHovered) && (
        <div className="marker-popup">
          <div className="popup-content">
            <div className="popup-header">
              <span className="popup-title">
                Image #{image.id.slice(-6)}
              </span>
              <span 
                className="popup-status"
                style={{ backgroundColor: STATUS_COLORS[image.status] }}
              >
                {image.status}
              </span>
            </div>
            
            <div className="popup-details">
              <div className="popup-item">
                <span className="popup-label">Time:</span>
                <span className="popup-value">{formatRelativeTime(image.createdAt)}</span>
              </div>
              
              {image.metadata.confidence !== undefined && (
                <div className="popup-item">
                  <span className="popup-label">Confidence:</span>
                  <span className="popup-value">{formatConfidence(image.metadata.confidence)}</span>
                </div>
              )}
              
              {image.location && (
                <div className="popup-item">
                  <span className="popup-label">Location:</span>
                  <span className="popup-value">
                    {image.location.latitude.toFixed(4)}, {image.location.longitude.toFixed(4)}
                  </span>
                </div>
              )}

              {image.metadata.objects && image.metadata.objects.length > 0 && (
                <div className="popup-item">
                  <span className="popup-label">Objects:</span>
                  <span className="popup-value">
                    {image.metadata.objects.slice(0, 2).map(obj => obj.name).join(', ')}
                    {image.metadata.objects.length > 2 && ` +${image.metadata.objects.length - 2}`}
                  </span>
                </div>
              )}

              {image.metadata.tags && image.metadata.tags.length > 0 && (
                <div className="popup-item">
                  <span className="popup-label">Tags:</span>
                  <span className="popup-value">
                    {image.metadata.tags.slice(0, 2).join(', ')}
                    {image.metadata.tags.length > 2 && ` +${image.metadata.tags.length - 2}`}
                  </span>
                </div>
              )}
            </div>

            {/* Action hint */}
            <div className="popup-hint">
              Click to select " Double-click to view
            </div>
          </div>
          
          {/* Popup arrow */}
          <div className="popup-arrow"></div>
        </div>
      )}

      <style>{`
        .image-marker {
          position: absolute;
          cursor: pointer;
          transition: all 0.3s ease;
          border-radius: 50%;
          overflow: visible;
        }

        .image-marker--selected {
          z-index: 1000 !important;
        }

        .image-marker--hovered {
          z-index: 999 !important;
        }

        .marker-container {
          position: relative;
          width: 100%;
          height: 100%;
          border-radius: 50%;
          overflow: hidden;
          border: 2px solid white;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
          background: white;
        }

        .image-marker--selected .marker-container {
          border-color: #007bff;
          box-shadow: 0 4px 16px rgba(0, 123, 255, 0.5);
        }

        .image-marker--hovered .marker-container {
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }

        .marker-thumbnail {
          width: 100%;
          height: 100%;
          border-radius: 50%;
          overflow: hidden;
          background: #f8f9fa;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .thumbnail-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          border-radius: 50%;
        }

        .marker-status {
          position: absolute;
          top: -2px;
          right: -2px;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 8px;
          color: white;
          border: 1px solid white;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }

        .marker-confidence {
          position: absolute;
          bottom: -2px;
          right: -2px;
          width: 12px;
          height: 12px;
          background: #ffc107;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 6px;
          border: 1px solid white;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }

        .marker-objects {
          position: absolute;
          bottom: -2px;
          left: -2px;
          width: 14px;
          height: 14px;
          background: #28a745;
          color: white;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 8px;
          font-weight: bold;
          border: 1px solid white;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }

        .marker-popup {
          position: absolute;
          bottom: 100%;
          left: 50%;
          transform: translateX(-50%);
          margin-bottom: 10px;
          z-index: 1001;
          pointer-events: none;
          animation: popupFadeIn 0.2s ease-out;
        }

        @keyframes popupFadeIn {
          from {
            opacity: 0;
            transform: translateX(-50%) translateY(5px);
          }
          to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
          }
        }

        .popup-content {
          background: white;
          border-radius: 8px;
          padding: 12px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
          border: 1px solid #e9ecef;
          min-width: 200px;
          max-width: 280px;
        }

        .popup-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 8px;
          padding-bottom: 6px;
          border-bottom: 1px solid #e9ecef;
        }

        .popup-title {
          font-weight: 600;
          font-size: 13px;
          color: #2d3748;
        }

        .popup-status {
          padding: 2px 6px;
          border-radius: 10px;
          color: white;
          font-size: 10px;
          font-weight: 500;
          text-transform: uppercase;
        }

        .popup-details {
          margin-bottom: 8px;
        }

        .popup-item {
          display: flex;
          justify-content: space-between;
          margin-bottom: 4px;
          font-size: 12px;
          line-height: 1.4;
        }

        .popup-label {
          color: #6c757d;
          font-weight: 500;
          margin-right: 8px;
        }

        .popup-value {
          color: #2d3748;
          text-align: right;
          flex: 1;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .popup-hint {
          font-size: 10px;
          color: #6c757d;
          text-align: center;
          font-style: italic;
          padding-top: 6px;
          border-top: 1px solid #f8f9fa;
        }

        .popup-arrow {
          position: absolute;
          top: 100%;
          left: 50%;
          transform: translateX(-50%);
          width: 0;
          height: 0;
          border-left: 6px solid transparent;
          border-right: 6px solid transparent;
          border-top: 6px solid white;
        }

        .popup-arrow::before {
          content: '';
          position: absolute;
          top: -7px;
          left: -6px;
          width: 0;
          height: 0;
          border-left: 6px solid transparent;
          border-right: 6px solid transparent;
          border-top: 6px solid #e9ecef;
        }

        /* Size variations */
        .image-marker--small {
          width: ${MAP_SETTINGS.markerSize.small}px;
          height: ${MAP_SETTINGS.markerSize.small}px;
        }

        .image-marker--medium {
          width: ${MAP_SETTINGS.markerSize.medium}px;
          height: ${MAP_SETTINGS.markerSize.medium}px;
        }

        .image-marker--large {
          width: ${MAP_SETTINGS.markerSize.large}px;
          height: ${MAP_SETTINGS.markerSize.large}px;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
          .popup-content {
            min-width: 180px;
            max-width: 220px;
            padding: 10px;
          }

          .popup-item {
            flex-direction: column;
            align-items: flex-start;
            margin-bottom: 6px;
          }

          .popup-value {
            text-align: left;
            margin-top: 2px;
          }

          .marker-status,
          .marker-confidence,
          .marker-objects {
            width: 12px;
            height: 12px;
            font-size: 7px;
          }
        }
      `}</style>
    </div>
  );
};
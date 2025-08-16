import React, { useState, useCallback, useMemo } from 'react';
import { ImageData } from '../../types/image';
import { formatRelativeTime } from '../../utils/formatters';
import { MAP_SETTINGS, STATUS_COLORS } from '../../utils/constants';

interface ImageClusterProps {
  images: ImageData[];
  center: { lat: number; lng: number };
  size?: 'small' | 'medium' | 'large';
  onClick?: (images: ImageData[]) => void;
  onImageClick?: (image: ImageData) => void;
  expanded?: boolean;
  className?: string;
}

export const ImageCluster: React.FC<ImageClusterProps> = ({
  images,
  center,
  size = 'medium',
  onClick,
  onImageClick,
  expanded = false,
  className = ''
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [showPreview, setShowPreview] = useState(false);

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (expanded) {
      // When expanded, handle individual image clicks
      return;
    }
    onClick?.(images);
  }, [onClick, images, expanded]);

  const handleMouseEnter = useCallback(() => {
    setIsHovered(true);
    setTimeout(() => setShowPreview(true), 300);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
    setShowPreview(false);
  }, []);

  const handleImageClick = useCallback((e: React.MouseEvent, image: ImageData) => {
    e.stopPropagation();
    onImageClick?.(image);
  }, [onImageClick]);

  // Calculate cluster statistics
  const stats = useMemo(() => {
    const total = images.length;
    const withLocation = images.filter(img => img.location).length;
    const completed = images.filter(img => img.status === 'completed').length;
    const processing = images.filter(img => img.status === 'processing').length;
    const errors = images.filter(img => img.status === 'error').length;
    const avgConfidence = images.reduce((sum, img) => sum + (img.metadata.confidence || 0), 0) / total;
    
    return {
      total,
      withLocation,
      completed,
      processing,
      errors,
      avgConfidence,
      recentCount: images.filter(img => 
        Date.now() - new Date(img.createdAt).getTime() < 24 * 60 * 60 * 1000
      ).length
    };
  }, [images]);

  const clusterSize = expanded ? 60 : MAP_SETTINGS.markerSize[size] * 1.5;
  const previewImages = images.slice(0, 4);

  // Get cluster color based on status distribution
  const getClusterColor = () => {
    if (stats.errors > stats.total * 0.3) return STATUS_COLORS.error;
    if (stats.processing > stats.total * 0.3) return STATUS_COLORS.processing;
    if (stats.completed === stats.total) return STATUS_COLORS.completed;
    return '#007bff';
  };

  return (
    <div
      className={`image-cluster ${className} ${expanded ? 'image-cluster--expanded' : ''} ${isHovered ? 'image-cluster--hovered' : ''}`}
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{
        width: clusterSize,
        height: clusterSize,
        transform: isHovered && !expanded ? 'scale(1.1)' : 'scale(1)',
        zIndex: isHovered ? 999 : expanded ? 1000 : 1
      }}
    >
      {expanded ? (
        /* Expanded cluster showing individual images */
        <div className="cluster-expanded">
          {images.slice(0, 9).map((image, index) => (
            <div
              key={image.id}
              className="cluster-image"
              onClick={(e) => handleImageClick(e, image)}
              style={{
                transform: `translate(${(index % 3) * 25 - 25}px, ${Math.floor(index / 3) * 25 - 25}px)`,
                zIndex: index + 1
              }}
            >
              <img
                src={image.thumbnailUrl || image.imageUrl}
                alt={`Image ${index + 1}`}
                className="cluster-image-thumb"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                }}
              />
              <div 
                className="cluster-image-status"
                style={{ backgroundColor: STATUS_COLORS[image.status] }}
              />
            </div>
          ))}
          {images.length > 9 && (
            <div className="cluster-more">
              +{images.length - 9}
            </div>
          )}
        </div>
      ) : (
        /* Compact cluster */
        <div className="cluster-compact">
          {/* Background circle */}
          <div 
            className="cluster-background"
            style={{ backgroundColor: getClusterColor() }}
          />
          
          {/* Preview thumbnails */}
          <div className="cluster-thumbnails">
            {previewImages.map((image, index) => (
              <div
                key={image.id}
                className="cluster-thumbnail"
                style={{
                  transform: `rotate(${index * 90}deg) translateY(-${clusterSize * 0.15}px)`,
                  zIndex: previewImages.length - index
                }}
              >
                <img
                  src={image.thumbnailUrl || image.imageUrl}
                  alt={`Preview ${index + 1}`}
                  className="cluster-thumbnail-image"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.style.display = 'none';
                  }}
                />
              </div>
            ))}
          </div>

          {/* Count */}
          <div className="cluster-count">
            {stats.total}
          </div>

          {/* Status indicators */}
          <div className="cluster-indicators">
            {stats.processing > 0 && (
              <div 
                className="cluster-indicator"
                style={{ backgroundColor: STATUS_COLORS.processing }}
                title={`${stats.processing} processing`}
              >
                �
              </div>
            )}
            {stats.errors > 0 && (
              <div 
                className="cluster-indicator"
                style={{ backgroundColor: STATUS_COLORS.error }}
                title={`${stats.errors} errors`}
              >
                L
              </div>
            )}
            {stats.recentCount > 0 && (
              <div 
                className="cluster-indicator cluster-indicator--new"
                title={`${stats.recentCount} recent`}
              >
                
              </div>
            )}
          </div>
        </div>
      )}

      {/* Detailed popup on hover */}
      {showPreview && !expanded && (
        <div className="cluster-popup">
          <div className="popup-content">
            <div className="popup-header">
              <span className="popup-title">
                {stats.total} Image{stats.total !== 1 ? 's' : ''}
              </span>
              <span className="popup-time">
                {formatRelativeTime(Math.min(...images.map(img => new Date(img.createdAt).getTime())))}
              </span>
            </div>

            <div className="popup-stats">
              <div className="stat-item">
                <span className="stat-icon"></span>
                <span className="stat-label">Completed:</span>
                <span className="stat-value">{stats.completed}</span>
              </div>
              
              {stats.processing > 0 && (
                <div className="stat-item">
                  <span className="stat-icon">�</span>
                  <span className="stat-label">Processing:</span>
                  <span className="stat-value">{stats.processing}</span>
                </div>
              )}
              
              {stats.errors > 0 && (
                <div className="stat-item">
                  <span className="stat-icon">L</span>
                  <span className="stat-label">Errors:</span>
                  <span className="stat-value">{stats.errors}</span>
                </div>
              )}
              
              <div className="stat-item">
                <span className="stat-icon">=�</span>
                <span className="stat-label">With location:</span>
                <span className="stat-value">{stats.withLocation}</span>
              </div>
              
              {stats.avgConfidence > 0 && (
                <div className="stat-item">
                  <span className="stat-icon">P</span>
                  <span className="stat-label">Avg confidence:</span>
                  <span className="stat-value">{Math.round(stats.avgConfidence * 100)}%</span>
                </div>
              )}
            </div>

            <div className="popup-preview">
              {previewImages.map((image, index) => (
                <div key={image.id} className="preview-image">
                  <img
                    src={image.thumbnailUrl || image.imageUrl}
                    alt={`Preview ${index + 1}`}
                    className="preview-thumb"
                  />
                </div>
              ))}
              {images.length > 4 && (
                <div className="preview-more">
                  +{images.length - 4}
                </div>
              )}
            </div>

            <div className="popup-hint">
              Click to expand cluster
            </div>
          </div>
          
          <div className="popup-arrow"></div>
        </div>
      )}

      <style>{`
        .image-cluster {
          position: absolute;
          cursor: pointer;
          transition: all 0.3s ease;
          border-radius: 50%;
          overflow: visible;
        }

        .image-cluster--hovered {
          z-index: 999 !important;
        }

        .image-cluster--expanded {
          z-index: 1000 !important;
          cursor: default;
        }

        .cluster-compact {
          position: relative;
          width: 100%;
          height: 100%;
          border-radius: 50%;
        }

        .cluster-background {
          width: 100%;
          height: 100%;
          border-radius: 50%;
          border: 3px solid white;
          box-shadow: 0 3px 12px rgba(0, 0, 0, 0.4);
          opacity: 0.9;
        }

        .image-cluster--hovered .cluster-background {
          box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        }

        .cluster-thumbnails {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
        }

        .cluster-thumbnail {
          position: absolute;
          top: 50%;
          left: 50%;
          width: 20px;
          height: 20px;
          margin: -10px 0 0 -10px;
          border-radius: 50%;
          overflow: hidden;
          border: 1px solid white;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
          transform-origin: center center;
        }

        .cluster-thumbnail-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .cluster-count {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          color: white;
          font-weight: bold;
          font-size: ${clusterSize > 40 ? '14px' : '12px'};
          text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
        }

        .cluster-indicators {
          position: absolute;
          top: -5px;
          right: -5px;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .cluster-indicator {
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

        .cluster-indicator--new {
          background: #ffc107;
          color: black;
        }

        .cluster-expanded {
          position: relative;
          width: 100%;
          height: 100%;
        }

        .cluster-image {
          position: absolute;
          top: 50%;
          left: 50%;
          width: 24px;
          height: 24px;
          margin: -12px 0 0 -12px;
          border-radius: 50%;
          overflow: hidden;
          border: 2px solid white;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
          cursor: pointer;
          transition: all 0.2s ease;
          background: white;
        }

        .cluster-image:hover {
          transform: scale(1.2) !important;
          z-index: 1000 !important;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }

        .cluster-image-thumb {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .cluster-image-status {
          position: absolute;
          top: -2px;
          right: -2px;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          border: 1px solid white;
        }

        .cluster-more {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: rgba(0, 0, 0, 0.7);
          color: white;
          border-radius: 50%;
          width: 20px;
          height: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          font-weight: bold;
        }

        .cluster-popup {
          position: absolute;
          bottom: 100%;
          left: 50%;
          transform: translateX(-50%);
          margin-bottom: 15px;
          z-index: 1001;
          pointer-events: none;
          animation: popupFadeIn 0.2s ease-out;
        }

        @keyframes popupFadeIn {
          from {
            opacity: 0;
            transform: translateX(-50%) translateY(10px);
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
          min-width: 220px;
          max-width: 300px;
        }

        .popup-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
          padding-bottom: 6px;
          border-bottom: 1px solid #e9ecef;
        }

        .popup-title {
          font-weight: 600;
          font-size: 14px;
          color: #2d3748;
        }

        .popup-time {
          font-size: 11px;
          color: #6c757d;
        }

        .popup-stats {
          margin-bottom: 12px;
        }

        .stat-item {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 4px;
          font-size: 12px;
        }

        .stat-icon {
          font-size: 10px;
          width: 12px;
        }

        .stat-label {
          color: #6c757d;
          flex: 1;
        }

        .stat-value {
          color: #2d3748;
          font-weight: 500;
        }

        .popup-preview {
          display: flex;
          gap: 4px;
          margin-bottom: 8px;
          align-items: center;
        }

        .preview-image {
          width: 24px;
          height: 24px;
          border-radius: 4px;
          overflow: hidden;
          border: 1px solid #e9ecef;
        }

        .preview-thumb {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .preview-more {
          background: #e9ecef;
          color: #6c757d;
          font-size: 10px;
          font-weight: 500;
          padding: 2px 6px;
          border-radius: 4px;
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
          border-left: 8px solid transparent;
          border-right: 8px solid transparent;
          border-top: 8px solid white;
        }

        .popup-arrow::before {
          content: '';
          position: absolute;
          top: -9px;
          left: -8px;
          width: 0;
          height: 0;
          border-left: 8px solid transparent;
          border-right: 8px solid transparent;
          border-top: 8px solid #e9ecef;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
          .popup-content {
            min-width: 200px;
            max-width: 250px;
            padding: 10px;
          }

          .cluster-image {
            width: 20px;
            height: 20px;
            margin: -10px 0 0 -10px;
          }

          .cluster-thumbnail {
            width: 16px;
            height: 16px;
            margin: -8px 0 0 -8px;
          }
        }
      `}</style>
    </div>
  );
};
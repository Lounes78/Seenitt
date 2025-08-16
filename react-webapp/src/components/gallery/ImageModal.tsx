import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ImageData } from '../../types/image';
import { formatRelativeTime, formatConfidence, formatFileSize, formatCoordinates, formatDimensions } from '../../utils/formatters';
import { STATUS_COLORS, Z_INDEX } from '../../utils/constants';

interface ImageModalProps {
  image: ImageData;
  onClose: () => void;
  onNext?: () => void;
  onPrevious?: () => void;
  hasNext?: boolean;
  hasPrevious?: boolean;
}

export const ImageModal: React.FC<ImageModalProps> = ({
  image,
  onClose,
  onNext,
  onPrevious,
  hasNext = false,
  hasPrevious = false
}) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [showMetadata, setShowMetadata] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  const modalRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          if (hasPrevious && onPrevious) {
            onPrevious();
          }
          break;
        case 'ArrowRight':
          if (hasNext && onNext) {
            onNext();
          }
          break;
        case 'i':
        case 'I':
          setShowMetadata(!showMetadata);
          break;
        case '+':
        case '=':
          handleZoomIn();
          break;
        case '-':
          handleZoomOut();
          break;
        case '0':
          handleResetZoom();
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose, onNext, onPrevious, hasNext, hasPrevious, showMetadata]);

  // Handle image load/error
  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
    setImageError(false);
  }, []);

  const handleImageError = useCallback(() => {
    setImageError(true);
    setImageLoaded(false);
  }, []);

  // Reset image state when image changes
  useEffect(() => {
    setImageLoaded(false);
    setImageError(false);
    setZoomLevel(1);
    setPanPosition({ x: 0, y: 0 });
  }, [image.id]);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    setZoomLevel(prev => Math.min(prev * 1.2, 5));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.5));
  }, []);

  const handleResetZoom = useCallback(() => {
    setZoomLevel(1);
    setPanPosition({ x: 0, y: 0 });
  }, []);

  // Pan controls
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoomLevel > 1) {
      setIsDragging(true);
      setLastMousePos({ x: e.clientX, y: e.clientY });
    }
  }, [zoomLevel]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging && zoomLevel > 1) {
      const deltaX = e.clientX - lastMousePos.x;
      const deltaY = e.clientY - lastMousePos.y;
      
      setPanPosition(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      
      setLastMousePos({ x: e.clientX, y: e.clientY });
    }
  }, [isDragging, lastMousePos, zoomLevel]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomLevel(prev => Math.max(0.5, Math.min(5, prev * delta)));
  }, []);

  // Click outside to close
  const handleBackdropClick = useCallback((e: React.MouseEvent) => {
    if (e.target === modalRef.current) {
      onClose();
    }
  }, [onClose]);

  const imageTransform = `scale(${zoomLevel}) translate(${panPosition.x / zoomLevel}px, ${panPosition.y / zoomLevel}px)`;

  return (
    <div 
      ref={modalRef}
      className="image-modal"
      onClick={handleBackdropClick}
      style={{ zIndex: Z_INDEX.modal }}
    >
      <div className="image-modal__content">
        {/* Header */}
        <div className="image-modal__header">
          <div className="image-modal__title">
            <span className="title-text">Image #{image.id.slice(-8)}</span>
            <div 
              className="status-badge"
              style={{ backgroundColor: STATUS_COLORS[image.status] }}
            >
              {image.status}
            </div>
          </div>
          
          <div className="image-modal__controls">
            <button 
              className="control-btn"
              onClick={() => setShowMetadata(!showMetadata)}
              title="Toggle metadata (i)"
            >
              9
            </button>
            <button 
              className="control-btn"
              onClick={handleResetZoom}
              title="Reset zoom (0)"
            >
              =
            </button>
            <button 
              className="control-btn control-btn--close"
              onClick={onClose}
              title="Close (Esc)"
            >
              
            </button>
          </div>
        </div>

        {/* Main content */}
        <div className="image-modal__body">
          {/* Image container */}
          <div className="image-modal__image-container">
            {!imageLoaded && !imageError && (
              <div className="image-modal__loading">
                <div className="loading-spinner"></div>
                <p>Loading image...</p>
              </div>
            )}

            {imageError ? (
              <div className="image-modal__error">
                <span className="error-icon">�</span>
                <h3>Failed to load image</h3>
                <p>The image could not be loaded. Please try again later.</p>
              </div>
            ) : (
              <div 
                className="image-wrapper"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={handleWheel}
                style={{ cursor: zoomLevel > 1 ? 'grab' : 'zoom-in' }}
              >
                <img
                  ref={imageRef}
                  src={image.imageUrl}
                  alt={`Processed image from ${formatRelativeTime(image.createdAt)}`}
                  className="image-modal__image"
                  onLoad={handleImageLoad}
                  onError={handleImageError}
                  style={{
                    transform: imageTransform,
                    cursor: isDragging ? 'grabbing' : 'inherit'
                  }}
                  onClick={zoomLevel === 1 ? handleZoomIn : undefined}
                />

                {/* Zoom controls */}
                <div className="zoom-controls">
                  <button 
                    className="zoom-btn"
                    onClick={handleZoomOut}
                    disabled={zoomLevel <= 0.5}
                    title="Zoom out (-)"
                  >
                    -
                  </button>
                  <span className="zoom-level">{Math.round(zoomLevel * 100)}%</span>
                  <button 
                    className="zoom-btn"
                    onClick={handleZoomIn}
                    disabled={zoomLevel >= 5}
                    title="Zoom in (+)"
                  >
                    +
                  </button>
                </div>
              </div>
            )}

            {/* Navigation arrows */}
            {hasPrevious && (
              <button 
                className="nav-arrow nav-arrow--prev"
                onClick={onPrevious}
                title="Previous image (�)"
              >
                9
              </button>
            )}
            
            {hasNext && (
              <button 
                className="nav-arrow nav-arrow--next"
                onClick={onNext}
                title="Next image (�)"
              >
                :
              </button>
            )}
          </div>

          {/* Metadata panel */}
          {showMetadata && (
            <div className="image-modal__metadata">
              <div className="metadata-section">
                <h3>Image Information</h3>
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span className="metadata-label">Created:</span>
                    <span className="metadata-value">{formatRelativeTime(image.createdAt)}</span>
                  </div>
                  
                  {image.metadata.confidence !== undefined && (
                    <div className="metadata-item">
                      <span className="metadata-label">Confidence:</span>
                      <span className="metadata-value">{formatConfidence(image.metadata.confidence)}</span>
                    </div>
                  )}
                  
                  {image.metadata.fileSize && (
                    <div className="metadata-item">
                      <span className="metadata-label">File Size:</span>
                      <span className="metadata-value">{formatFileSize(image.metadata.fileSize)}</span>
                    </div>
                  )}
                  
                  {image.metadata.imageSize && (
                    <div className="metadata-item">
                      <span className="metadata-label">Dimensions:</span>
                      <span className="metadata-value">
                        {formatDimensions(image.metadata.imageSize.width, image.metadata.imageSize.height)}
                      </span>
                    </div>
                  )}
                  
                  {image.metadata.processingTime && (
                    <div className="metadata-item">
                      <span className="metadata-label">Processing Time:</span>
                      <span className="metadata-value">{image.metadata.processingTime}ms</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Location information */}
              {image.location && (
                <div className="metadata-section">
                  <h3>Location</h3>
                  <div className="metadata-grid">
                    <div className="metadata-item">
                      <span className="metadata-label">Coordinates:</span>
                      <span className="metadata-value">
                        {formatCoordinates(image.location.latitude, image.location.longitude)}
                      </span>
                    </div>
                    
                    {image.location.accuracy && (
                      <div className="metadata-item">
                        <span className="metadata-label">Accuracy:</span>
                        <span className="metadata-value">{image.location.accuracy}m</span>
                      </div>
                    )}
                    
                    {image.location.altitude && (
                      <div className="metadata-item">
                        <span className="metadata-label">Altitude:</span>
                        <span className="metadata-value">{image.location.altitude}m</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Detected objects */}
              {image.metadata.objects && image.metadata.objects.length > 0 && (
                <div className="metadata-section">
                  <h3>Detected Objects ({image.metadata.objects.length})</h3>
                  <div className="objects-list">
                    {image.metadata.objects.map((obj, index) => (
                      <div key={index} className="object-item">
                        <span className="object-name">{obj.name}</span>
                        <span className="object-confidence">{formatConfidence(obj.confidence)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tags */}
              {image.metadata.tags && image.metadata.tags.length > 0 && (
                <div className="metadata-section">
                  <h3>Tags</h3>
                  <div className="tags-list">
                    {image.metadata.tags.map((tag, index) => (
                      <span key={index} className="tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <style>{`
        .image-modal {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.9);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
        }

        .image-modal__content {
          background: white;
          border-radius: 12px;
          max-width: 95vw;
          max-height: 95vh;
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .image-modal__header {
          padding: 16px 20px;
          border-bottom: 1px solid #e9ecef;
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: #f8f9fa;
        }

        .image-modal__title {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .title-text {
          font-size: 18px;
          font-weight: 600;
          color: #2d3748;
        }

        .status-badge {
          padding: 4px 12px;
          border-radius: 16px;
          color: white;
          font-size: 12px;
          font-weight: 500;
          text-transform: uppercase;
        }

        .image-modal__controls {
          display: flex;
          gap: 8px;
        }

        .control-btn {
          width: 32px;
          height: 32px;
          border: none;
          border-radius: 6px;
          background: #e9ecef;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.2s;
        }

        .control-btn:hover {
          background: #dee2e6;
        }

        .control-btn--close {
          background: #dc3545;
          color: white;
        }

        .control-btn--close:hover {
          background: #c82333;
        }

        .image-modal__body {
          flex: 1;
          display: flex;
          overflow: hidden;
        }

        .image-modal__image-container {
          flex: 1;
          position: relative;
          background: #f8f9fa;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
        }

        .image-modal__loading {
          text-align: center;
          color: #6c757d;
        }

        .loading-spinner {
          width: 48px;
          height: 48px;
          border: 4px solid #e9ecef;
          border-top: 4px solid #007bff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 16px;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .image-modal__error {
          text-align: center;
          color: #6c757d;
        }

        .error-icon {
          font-size: 48px;
          display: block;
          margin-bottom: 16px;
        }

        .image-wrapper {
          position: relative;
          width: 100%;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
        }

        .image-modal__image {
          max-width: 100%;
          max-height: 100%;
          object-fit: contain;
          transition: transform 0.1s ease;
          user-select: none;
        }

        .zoom-controls {
          position: absolute;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          gap: 8px;
          background: rgba(0, 0, 0, 0.7);
          padding: 8px 12px;
          border-radius: 20px;
          color: white;
        }

        .zoom-btn {
          width: 28px;
          height: 28px;
          border: none;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 16px;
          font-weight: bold;
        }

        .zoom-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.3);
        }

        .zoom-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .zoom-level {
          font-size: 12px;
          font-weight: 500;
          min-width: 40px;
          text-align: center;
        }

        .nav-arrow {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          width: 48px;
          height: 48px;
          border: none;
          border-radius: 50%;
          background: rgba(0, 0, 0, 0.7);
          color: white;
          font-size: 24px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .nav-arrow:hover {
          background: rgba(0, 0, 0, 0.8);
          transform: translateY(-50%) scale(1.1);
        }

        .nav-arrow--prev {
          left: 20px;
        }

        .nav-arrow--next {
          right: 20px;
        }

        .image-modal__metadata {
          width: 320px;
          background: white;
          border-left: 1px solid #e9ecef;
          overflow-y: auto;
          padding: 20px;
        }

        .metadata-section {
          margin-bottom: 24px;
        }

        .metadata-section h3 {
          font-size: 16px;
          font-weight: 600;
          color: #2d3748;
          margin: 0 0 12px 0;
        }

        .metadata-grid {
          display: grid;
          gap: 8px;
        }

        .metadata-item {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          align-items: center;
        }

        .metadata-label {
          font-size: 13px;
          color: #6c757d;
          font-weight: 500;
        }

        .metadata-value {
          font-size: 13px;
          color: #2d3748;
          text-align: right;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .objects-list {
          display: grid;
          gap: 6px;
        }

        .object-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 6px 8px;
          background: #f8f9fa;
          border-radius: 4px;
        }

        .object-name {
          font-size: 13px;
          color: #2d3748;
          font-weight: 500;
        }

        .object-confidence {
          font-size: 12px;
          color: #6c757d;
        }

        .tags-list {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }

        .tag {
          background: #007bff;
          color: white;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
        }

        /* Responsive design */
        @media (max-width: 768px) {
          .image-modal {
            padding: 10px;
          }

          .image-modal__content {
            flex-direction: column;
          }

          .image-modal__metadata {
            width: 100%;
            max-height: 40%;
            border-left: none;
            border-top: 1px solid #e9ecef;
          }

          .nav-arrow {
            width: 40px;
            height: 40px;
            font-size: 20px;
          }

          .nav-arrow--prev {
            left: 10px;
          }

          .nav-arrow--next {
            right: 10px;
          }
        }
      `}</style>
    </div>
  );
};
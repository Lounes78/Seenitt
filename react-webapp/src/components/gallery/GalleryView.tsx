import React, { useState, useMemo, useCallback } from 'react';
import { ImageData, ImageFilter, SortOption } from '../../types/image';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ImageCard } from './ImageCard';
import { ImageModal } from './ImageModal';
import { GalleryFilters } from './GalleryFilters';

interface GalleryViewProps {
  images: ImageData[];
  isLoading: boolean;
  onImageSelect: (image: ImageData) => void;
  onViewChange: () => void;
  className?: string;
}

export const GalleryView: React.FC<GalleryViewProps> = ({
  images,
  isLoading,
  onImageSelect,
  onViewChange,
  className = ''
}) => {
  // State management
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [gridSize, setGridSize] = useState<'small' | 'medium' | 'large'>('medium');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filter, setFilter] = useState<ImageFilter>({
    sessionId: null,
    dateRange: null,
    hasLocation: null,
    tags: [],
    confidence: null
  });
  const [sortBy, setSortBy] = useState<SortOption>('newest');
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());
  const [isSelectionMode, setIsSelectionMode] = useState(false);

  // Filtered and sorted images
  const processedImages = useMemo(() => {
    let filtered = [...images];

    // Apply filters
    if (filter.sessionId) {
      filtered = filtered.filter(img => img.sessionId === filter.sessionId);
    }

    if (filter.hasLocation !== null) {
      filtered = filtered.filter(img => 
        filter.hasLocation ? !!img.location : !img.location
      );
    }

    if (filter.tags && filter.tags.length > 0) {
      filtered = filtered.filter(img =>
        filter.tags!.some(tag => img.metadata.tags?.includes(tag))
      );
    }

    if (filter.confidence !== null) {
      filtered = filtered.filter(img =>
        img.metadata.confidence !== undefined &&
        img.metadata.confidence >= filter.confidence!
      );
    }

    if (filter.dateRange) {
      const { start, end } = filter.dateRange;
      filtered = filtered.filter(img => {
        const imgDate = new Date(img.createdAt);
        return imgDate >= start && imgDate <= end;
      });
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'newest':
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
        case 'oldest':
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        case 'confidence':
          const aConf = a.metadata.confidence || 0;
          const bConf = b.metadata.confidence || 0;
          return bConf - aConf;
        case 'location':
          if (!a.location && !b.location) return 0;
          if (!a.location) return 1;
          if (!b.location) return -1;
          return b.location.latitude - a.location.latitude;
        default:
          return 0;
      }
    });

    return filtered;
  }, [images, filter, sortBy]);

  // Event handlers
  const handleImageClick = useCallback((image: ImageData) => {
    if (isSelectionMode) {
      const newSelected = new Set(selectedImages);
      if (newSelected.has(image.id)) {
        newSelected.delete(image.id);
      } else {
        newSelected.add(image.id);
      }
      setSelectedImages(newSelected);
    } else {
      setSelectedImage(image);
      onImageSelect(image);
    }
  }, [isSelectionMode, selectedImages, onImageSelect]);

  const handleImageDoubleClick = useCallback((image: ImageData) => {
    setSelectedImage(image);
  }, []);

  const handleSelectAll = useCallback(() => {
    if (selectedImages.size === processedImages.length) {
      setSelectedImages(new Set());
    } else {
      setSelectedImages(new Set(processedImages.map(img => img.id)));
    }
  }, [selectedImages.size, processedImages]);

  const handleClearSelection = useCallback(() => {
    setSelectedImages(new Set());
    setIsSelectionMode(false);
  }, []);

  // Grid size configurations
  const gridConfig = {
    small: { columns: 6, gap: 8, imageSize: 150 },
    medium: { columns: 4, gap: 12, imageSize: 200 },
    large: { columns: 3, gap: 16, imageSize: 250 }
  };

  const currentConfig = gridConfig[gridSize];

  if (isLoading && images.length === 0) {
    return (
      <div className="gallery-loading">
        <LoadingSpinner size="large" text="Loading images..." />
      </div>
    );
  }

  return (
    <div className={`gallery-view ${className}`}>
      {/* Header Controls */}
      <div className="gallery-header">
        <div className="gallery-header__info">
          <h2 className="gallery-title">
            Image Gallery
            <span className="image-count">({processedImages.length})</span>
          </h2>
          
          {isSelectionMode && selectedImages.size > 0 && (
            <div className="selection-info">
              {selectedImages.size} selected
            </div>
          )}
        </div>

        <div className="gallery-header__controls">
          {/* View Mode Toggle */}
          <div className="view-controls">
            <button
              className={`view-btn ${viewMode === 'grid' ? 'view-btn--active' : ''}`}
              onClick={() => setViewMode('grid')}
              title="Grid view"
            >
              ⊞
            </button>
            <button
              className={`view-btn ${viewMode === 'list' ? 'view-btn--active' : ''}`}
              onClick={() => setViewMode('list')}
              title="List view"
            >
              ☰
            </button>
          </div>

          {/* Grid Size Controls */}
          {viewMode === 'grid' && (
            <div className="size-controls">
              <button
                className={`size-btn ${gridSize === 'small' ? 'size-btn--active' : ''}`}
                onClick={() => setGridSize('small')}
                title="Small thumbnails"
              >
                ⚏
              </button>
              <button
                className={`size-btn ${gridSize === 'medium' ? 'size-btn--active' : ''}`}
                onClick={() => setGridSize('medium')}
                title="Medium thumbnails"
              >
                ⚏
              </button>
              <button
                className={`size-btn ${gridSize === 'large' ? 'size-btn--active' : ''}`}
                onClick={() => setGridSize('large')}
                title="Large thumbnails"
              >
                ⚏
              </button>
            </div>
          )}

          {/* Selection Controls */}
          <div className="selection-controls">
            <button
              className={`selection-btn ${isSelectionMode ? 'selection-btn--active' : ''}`}
              onClick={() => setIsSelectionMode(!isSelectionMode)}
            >
              Select
            </button>
            
            {isSelectionMode && (
              <>
                <button
                  className="selection-btn"
                  onClick={handleSelectAll}
                >
                  {selectedImages.size === processedImages.length ? 'None' : 'All'}
                </button>
                <button
                  className="selection-btn selection-btn--clear"
                  onClick={handleClearSelection}
                >
                  Clear
                </button>
              </>
            )}
          </div>

          {/* Map View Button */}
          <button
            className="map-view-btn"
            onClick={onViewChange}
            disabled={images.filter(img => img.location).length === 0}
          >
            🗺️ Map View
          </button>
        </div>
      </div>

      {/* Filters */}
      <GalleryFilters
        filter={filter}
        sortBy={sortBy}
        onFilterChange={setFilter}
        onSortChange={setSortBy}
        availableTags={getAvailableTags(images)}
        availableSessions={getAvailableSessions(images)}
        className="gallery-filters"
      />

      {/* Selection Actions */}
      {isSelectionMode && selectedImages.size > 0 && (
        <div className="selection-actions">
          <div className="selection-actions__info">
            {selectedImages.size} image{selectedImages.size !== 1 ? 's' : ''} selected
          </div>
          <div className="selection-actions__buttons">
            <button className="action-btn action-btn--download">
              ⬇ Download
            </button>
            <button className="action-btn action-btn--delete">
              🗑️ Delete
            </button>
            <button className="action-btn action-btn--export">
              📤 Export
            </button>
          </div>
        </div>
      )}

      {/* Images Container */}
      <div className="gallery-content">
        {processedImages.length === 0 ? (
          <div className="empty-gallery">
            <div className="empty-icon">📷</div>
            <h3>No images found</h3>
            <p>
              {images.length === 0
                ? 'Start processing images to see them here'
                : 'Try adjusting your filters to see more results'
              }
            </p>
          </div>
        ) : (
          <div 
            className={`gallery-grid gallery-grid--${viewMode} gallery-grid--${gridSize}`}
            style={{
              gridTemplateColumns: viewMode === 'grid' 
                ? `repeat(auto-fill, minmax(${currentConfig.imageSize}px, 1fr))`
                : '1fr',
              gap: `${currentConfig.gap}px`
            }}
          >
            {processedImages.map((image) => (
              <ImageCard
                key={image.id}
                image={image}
                size={gridSize}
                viewMode={viewMode}
                selected={selectedImages.has(image.id)}
                selectionMode={isSelectionMode}
                onClick={() => handleImageClick(image)}
                onDoubleClick={() => handleImageDoubleClick(image)}
                className="gallery-item"
              />
            ))}
          </div>
        )}

        {/* Loading more indicator */}
        {isLoading && images.length > 0 && (
          <div className="loading-more">
            <LoadingSpinner size="medium" text="Loading more images..." />
          </div>
        )}
      </div>

      {/* Image Modal */}
      {selectedImage && (
        <ImageModal
          image={selectedImage}
          onClose={() => setSelectedImage(null)}
          onNext={() => {
            const currentIndex = processedImages.findIndex(img => img.id === selectedImage.id);
            const nextImage = processedImages[currentIndex + 1];
            if (nextImage) setSelectedImage(nextImage);
          }}
          onPrevious={() => {
            const currentIndex = processedImages.findIndex(img => img.id === selectedImage.id);
            const prevImage = processedImages[currentIndex - 1];
            if (prevImage) setSelectedImage(prevImage);
          }}
          hasNext={processedImages.findIndex(img => img.id === selectedImage.id) < processedImages.length - 1}
          hasPrevious={processedImages.findIndex(img => img.id === selectedImage.id) > 0}
        />
      )}

      <style>{`
        .gallery-view {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: #f8f9fa;
        }

        .gallery-loading {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 400px;
        }

        .gallery-header {
          background: white;
          padding: 20px;
          border-bottom: 1px solid #e9ecef;
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 16px;
        }

        .gallery-header__info {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .gallery-title {
          font-size: 24px;
          font-weight: 600;
          margin: 0;
          color: #2d3748;
        }

        .image-count {
          color: #6c757d;
          font-weight: 400;
          font-size: 18px;
        }

        .selection-info {
          background: #007bff;
          color: white;
          padding: 4px 12px;
          border-radius: 16px;
          font-size: 14px;
          font-weight: 500;
        }

        .gallery-header__controls {
          display: flex;
          align-items: center;
          gap: 16px;
          flex-wrap: wrap;
        }

        .view-controls, .size-controls, .selection-controls {
          display: flex;
          gap: 4px;
          background: #f8f9fa;
          padding: 4px;
          border-radius: 8px;
        }

        .view-btn, .size-btn, .selection-btn {
          padding: 8px 12px;
          border: none;
          background: none;
          cursor: pointer;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 500;
          color: #6c757d;
          transition: all 0.2s;
        }

        .view-btn--active, .size-btn--active, .selection-btn--active {
          background: white;
          color: #007bff;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .selection-btn--clear {
          color: #dc3545;
        }

        .map-view-btn {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          padding: 10px 16px;
          border-radius: 8px;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }

        .map-view-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }

        .map-view-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .selection-actions {
          background: #007bff;
          color: white;
          padding: 12px 20px;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .selection-actions__buttons {
          display: flex;
          gap: 12px;
        }

        .action-btn {
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: none;
          padding: 6px 12px;
          border-radius: 6px;
          font-size: 13px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .action-btn:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .gallery-content {
          flex: 1;
          padding: 20px;
          overflow-y: auto;
        }

        .empty-gallery {
          text-align: center;
          padding: 60px 20px;
          color: #6c757d;
        }

        .empty-icon {
          font-size: 48px;
          margin-bottom: 16px;
        }

        .empty-gallery h3 {
          font-size: 20px;
          margin: 0 0 8px 0;
          color: #495057;
        }

        .gallery-grid {
          display: grid;
          width: 100%;
        }

        .gallery-grid--list {
          gap: 16px !important;
        }

        .loading-more {
          padding: 20px;
          text-align: center;
        }

        @media (max-width: 768px) {
          .gallery-header {
            flex-direction: column;
            align-items: stretch;
          }

          .gallery-header__controls {
            justify-content: center;
          }

          .gallery-content {
            padding: 16px;
          }

          .gallery-grid {
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)) !important;
          }
        }
      `}</style>
    </div>
  );
};

// Helper functions
function getAvailableTags(images: ImageData[]): string[] {
  const tags = new Set<string>();
  images.forEach(img => {
    img.metadata.tags?.forEach(tag => tags.add(tag));
  });
  return Array.from(tags).sort();
}

function getAvailableSessions(images: ImageData[]): string[] {
  const sessions = new Set(images.map(img => img.sessionId));
  return Array.from(sessions).sort();
}

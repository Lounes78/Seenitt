import { useState, useCallback, useMemo } from 'react';
import { ImageData, ImageFilter, SortOption } from '../types/image';

interface ImageDataState {
  images: ImageData[];
  selectedImageId: string | null;
  filter: ImageFilter;
  sortBy: SortOption;
  isLoading: boolean;
}

/**
 * Custom hook for managing image data state and operations
 * Handles images received from your Python processing pipeline
 */
export const useImageData = () => {
  // Main state
  const [state, setState] = useState<ImageDataState>({
    images: [],
    selectedImageId: null,
    filter: {
      sessionId: null,
      dateRange: null,
      hasLocation: null,
      tags: [],
      confidence: null
    },
    sortBy: 'newest',
    isLoading: false
  });

  /**
   * Add a new image to the collection
   */
  const addImage = useCallback((imageData: ImageData) => {
    setState(prev => {
      // Check if image already exists to prevent duplicates
      const existingIndex = prev.images.findIndex(img => img.id === imageData.id);
      if (existingIndex !== -1) {
        console.log(`Image ${imageData.id} already exists, skipping duplicate`);
        return prev;
      }
      
      return {
        ...prev,
        images: [imageData, ...prev.images] // Add to beginning for newest first
      };
    });
  }, []);

  /**
   * Update an existing image
   */
  const updateImage = useCallback((imageId: string, updates: Partial<ImageData>) => {
    setState(prev => ({
      ...prev,
      images: prev.images.map(img => 
        img.id === imageId ? { ...img, ...updates } : img
      )
    }));
  }, []);

  /**
   * Remove an image
   */
  const removeImage = useCallback((imageId: string) => {
    setState(prev => ({
      ...prev,
      images: prev.images.filter(img => img.id !== imageId),
      selectedImageId: prev.selectedImageId === imageId ? null : prev.selectedImageId
    }));
  }, []);

  /**
   * Clear all images
   */
  const clearImages = useCallback(() => {
    setState(prev => ({
      ...prev,
      images: [],
      selectedImageId: null
    }));
  }, []);

  /**
   * Select an image
   */
  const selectImage = useCallback((imageId: string | null) => {
    setState(prev => ({
      ...prev,
      selectedImageId: imageId
    }));
  }, []);

  /**
   * Update filter
   */
  const updateFilter = useCallback((newFilter: Partial<ImageFilter>) => {
    setState(prev => ({
      ...prev,
      filter: { ...prev.filter, ...newFilter }
    }));
  }, []);

  /**
   * Update sort option
   */
  const updateSort = useCallback((sortBy: SortOption) => {
    setState(prev => ({
      ...prev,
      sortBy
    }));
  }, []);

  /**
   * Set loading state
   */
  const setLoading = useCallback((isLoading: boolean) => {
    setState(prev => ({
      ...prev,
      isLoading
    }));
  }, []);

  /**
   * Get filtered and sorted images
   */
  const filteredImages = useMemo(() => {
    let filtered = [...state.images];

    // Apply filters
    const { filter } = state;

    // Filter by session
    if (filter.sessionId) {
      filtered = filtered.filter(img => img.sessionId === filter.sessionId);
    }

    // Filter by date range
    if (filter.dateRange) {
      const { start, end } = filter.dateRange;
      filtered = filtered.filter(img => {
        const imgDate = new Date(img.createdAt);
        return imgDate >= start && imgDate <= end;
      });
    }

    // Filter by location availability
    if (filter.hasLocation !== null) {
      filtered = filtered.filter(img => 
        filter.hasLocation ? !!img.location : !img.location
      );
    }

    // Filter by tags
    if (filter.tags && filter.tags.length > 0) {
      filtered = filtered.filter(img => 
        filter.tags!.some(tag => img.metadata.tags?.includes(tag))
      );
    }

    // Filter by confidence
    if (filter.confidence !== null) {
      filtered = filtered.filter(img => 
        img.metadata.confidence !== undefined && 
        img.metadata.confidence >= filter.confidence!
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (state.sortBy) {
        case 'newest':
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
        
        case 'oldest':
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        
        case 'confidence':
          const aConf = a.metadata.confidence || 0;
          const bConf = b.metadata.confidence || 0;
          return bConf - aConf;
        
        case 'location':
          // Sort by location availability, then by latitude
          if (!a.location && !b.location) return 0;
          if (!a.location) return 1;
          if (!b.location) return -1;
          return b.location.latitude - a.location.latitude;
        
        default:
          return 0;
      }
    });

    return filtered;
  }, [state.images, state.filter, state.sortBy]);

  /**
   * Get images by session ID
   */
  const getImagesBySession = useCallback((sessionId: string) => {
    return state.images.filter(img => img.sessionId === sessionId);
  }, [state.images]);

  /**
   * Get images with location data
   */
  const getImagesWithLocation = useCallback(() => {
    return state.images.filter(img => img.location);
  }, [state.images]);

  /**
   * Get unique sessions
   */
  const getUniqueSessions = useCallback(() => {
    const sessions = new Set(state.images.map(img => img.sessionId));
    return Array.from(sessions);
  }, [state.images]);

  /**
   * Get unique tags
   */
  const getUniqueTags = useCallback(() => {
    const tags = new Set<string>();
    state.images.forEach(img => {
      img.metadata.tags?.forEach(tag => tags.add(tag));
    });
    return Array.from(tags).sort();
  }, [state.images]);

  /**
   * Get statistics
   */
  const getStats = useMemo(() => {
    const total = state.images.length;
    const withLocation = state.images.filter(img => img.location).length;
    const withObjects = state.images.filter(img => 
      img.metadata.objects && img.metadata.objects.length > 0
    ).length;
    
    const avgConfidence = total > 0 
      ? state.images
          .filter(img => img.metadata.confidence !== undefined)
          .reduce((sum, img) => sum + (img.metadata.confidence || 0), 0) / total
      : 0;

    const sessions = getUniqueSessions().length;
    const tags = getUniqueTags().length;

    return {
      total,
      withLocation,
      withObjects,
      avgConfidence: Math.round(avgConfidence * 100) / 100,
      sessions,
      tags,
      locationPercentage: total > 0 ? Math.round((withLocation / total) * 100) : 0
    };
  }, [state.images, getUniqueSessions, getUniqueTags]);

  /**
   * Find similar images (by objects or tags)
   */
  const findSimilarImages = useCallback((imageId: string, limit = 5) => {
    const targetImage = state.images.find(img => img.id === imageId);
    if (!targetImage) return [];

    const targetObjects = targetImage.metadata.objects || [];
    const targetTags = targetImage.metadata.tags || [];

    return state.images
      .filter(img => img.id !== imageId)
      .map(img => {
        const objects = img.metadata.objects || [];
        const tags = img.metadata.tags || [];
        
        // Calculate similarity score
        const objectMatches = targetObjects.filter(obj => 
          objects.some(o => o.name === obj.name)
        ).length;
        const tagMatches = targetTags.filter(tag => tags.includes(tag)).length;
        
        const similarity = (objectMatches + tagMatches) / 
          (targetObjects.length + targetTags.length + objects.length + tags.length - objectMatches - tagMatches);

        return { image: img, similarity };
      })
      .filter(item => item.similarity > 0)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .map(item => item.image);
  }, [state.images]);

  /**
   * Get recent images (last 24 hours)
   */
  const getRecentImages = useCallback((hours = 24) => {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    return state.images.filter(img => new Date(img.createdAt) > cutoff);
  }, [state.images]);

  // Return hook interface
  return {
    // State
    images: filteredImages,
    allImages: state.images,
    selectedImage: state.images.find(img => img.id === state.selectedImageId) || null,
    selectedImageId: state.selectedImageId,
    filter: state.filter,
    sortBy: state.sortBy,
    isLoading: state.isLoading,

    // Actions
    addImage,
    updateImage,
    removeImage,
    clearImages,
    selectImage,
    updateFilter,
    updateSort,
    setLoading,

    // Queries
    getImagesBySession,
    getImagesWithLocation,
    getUniqueSessions,
    getUniqueTags,
    findSimilarImages,
    getRecentImages,

    // Stats
    totalImages: state.images.length,
    filteredCount: filteredImages.length,
    stats: getStats
  };
};

export default useImageData;

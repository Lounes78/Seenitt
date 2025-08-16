// Image store for managing image data state

import { ImageData, ImageFilter, SortOption, ImageStats } from '../types/image';

/**
 * Image store interface for state management
 */
export interface ImageStore {
  // State
  images: ImageData[];
  selectedImages: Set<string>;
  currentFilter: ImageFilter;
  currentSort: SortOption;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  addImage: (image: ImageData) => void;
  addImages: (images: ImageData[]) => void;
  updateImage: (id: string, updates: Partial<ImageData>) => void;
  removeImage: (id: string) => void;
  removeImages: (ids: string[]) => void;
  clearImages: () => void;
  
  // Selection
  selectImage: (id: string) => void;
  deselectImage: (id: string) => void;
  selectAll: () => void;
  deselectAll: () => void;
  toggleSelection: (id: string) => void;
  
  // Filtering and sorting
  setFilter: (filter: ImageFilter) => void;
  setSort: (sort: SortOption) => void;
  resetFilters: () => void;
  
  // Getters
  getImage: (id: string) => ImageData | undefined;
  getImagesBySession: (sessionId: string) => ImageData[];
  getFilteredImages: () => ImageData[];
  getSelectedImages: () => ImageData[];
  getStats: () => ImageStats;
  
  // State setters
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

/**
 * Create a default image filter
 */
export const createDefaultFilter = (): ImageFilter => ({
  sessionId: null,
  dateRange: null,
  hasLocation: null,
  tags: [],
  confidence: null,
  categories: [],
  objects: []
});

/**
 * Image store implementation using a simple state pattern
 */
export class SimpleImageStore implements ImageStore {
  private state = {
    images: [] as ImageData[],
    selectedImages: new Set<string>(),
    currentFilter: createDefaultFilter(),
    currentSort: 'newest' as SortOption,
    isLoading: false,
    error: null as string | null
  };

  private listeners: Array<() => void> = [];

  // Subscribe to state changes
  subscribe(listener: () => void): () => void {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  // Notify all listeners of state changes
  private notify(): void {
    this.listeners.forEach(listener => listener());
  }

  // Getters
  get images(): ImageData[] {
    return this.state.images;
  }

  get selectedImages(): Set<string> {
    return new Set(this.state.selectedImages);
  }

  get currentFilter(): ImageFilter {
    return { ...this.state.currentFilter };
  }

  get currentSort(): SortOption {
    return this.state.currentSort;
  }

  get isLoading(): boolean {
    return this.state.isLoading;
  }

  get error(): string | null {
    return this.state.error;
  }

  // Image management
  addImage = (image: ImageData): void => {
    const existingIndex = this.state.images.findIndex(img => img.id === image.id);
    
    if (existingIndex >= 0) {
      // Update existing image
      this.state.images[existingIndex] = image;
    } else {
      // Add new image
      this.state.images.push(image);
    }
    
    this.notify();
  };

  addImages = (images: ImageData[]): void => {
    images.forEach(image => {
      const existingIndex = this.state.images.findIndex(img => img.id === image.id);
      
      if (existingIndex >= 0) {
        this.state.images[existingIndex] = image;
      } else {
        this.state.images.push(image);
      }
    });
    
    this.notify();
  };

  updateImage = (id: string, updates: Partial<ImageData>): void => {
    const index = this.state.images.findIndex(img => img.id === id);
    
    if (index >= 0) {
      this.state.images[index] = {
        ...this.state.images[index],
        ...updates,
        updatedAt: new Date()
      };
      this.notify();
    }
  };

  removeImage = (id: string): void => {
    this.state.images = this.state.images.filter(img => img.id !== id);
    this.state.selectedImages.delete(id);
    this.notify();
  };

  removeImages = (ids: string[]): void => {
    const idsSet = new Set(ids);
    this.state.images = this.state.images.filter(img => !idsSet.has(img.id));
    ids.forEach(id => this.state.selectedImages.delete(id));
    this.notify();
  };

  clearImages = (): void => {
    this.state.images = [];
    this.state.selectedImages.clear();
    this.notify();
  };

  // Selection management
  selectImage = (id: string): void => {
    this.state.selectedImages.add(id);
    this.notify();
  };

  deselectImage = (id: string): void => {
    this.state.selectedImages.delete(id);
    this.notify();
  };

  selectAll = (): void => {
    const filteredImages = this.getFilteredImages();
    filteredImages.forEach(img => this.state.selectedImages.add(img.id));
    this.notify();
  };

  deselectAll = (): void => {
    this.state.selectedImages.clear();
    this.notify();
  };

  toggleSelection = (id: string): void => {
    if (this.state.selectedImages.has(id)) {
      this.state.selectedImages.delete(id);
    } else {
      this.state.selectedImages.add(id);
    }
    this.notify();
  };

  // Filtering and sorting
  setFilter = (filter: ImageFilter): void => {
    this.state.currentFilter = { ...filter };
    this.notify();
  };

  setSort = (sort: SortOption): void => {
    this.state.currentSort = sort;
    this.notify();
  };

  resetFilters = (): void => {
    this.state.currentFilter = createDefaultFilter();
    this.notify();
  };

  // Getters
  getImage = (id: string): ImageData | undefined => {
    return this.state.images.find(img => img.id === id);
  };

  getImagesBySession = (sessionId: string): ImageData[] => {
    return this.state.images.filter(img => img.sessionId === sessionId);
  };

  getFilteredImages = (): ImageData[] => {
    let filtered = [...this.state.images];
    const filter = this.state.currentFilter;

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

    if (filter.categories && filter.categories.length > 0) {
      filtered = filtered.filter(img =>
        filter.categories!.some(category => img.metadata.categories?.includes(category))
      );
    }

    if (filter.objects && filter.objects.length > 0) {
      filtered = filtered.filter(img =>
        img.metadata.objects?.some(obj => 
          filter.objects!.includes(obj.name)
        )
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (this.state.currentSort) {
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
        case 'name':
          return a.id.localeCompare(b.id);
        case 'size':
          const aSize = a.metadata.fileSize || 0;
          const bSize = b.metadata.fileSize || 0;
          return bSize - aSize;
        case 'processing-time':
          const aTime = a.metadata.processingTime || 0;
          const bTime = b.metadata.processingTime || 0;
          return bTime - aTime;
        default:
          return 0;
      }
    });

    return filtered;
  };

  getSelectedImages = (): ImageData[] => {
    return this.state.images.filter(img => this.state.selectedImages.has(img.id));
  };

  getStats = (): ImageStats => {
    const images = this.state.images;
    const total = images.length;
    
    if (total === 0) {
      return {
        total: 0,
        withLocation: 0,
        withObjects: 0,
        avgConfidence: 0,
        sessions: 0,
        tags: 0,
        locationPercentage: 0,
        byStatus: {
          processing: 0,
          completed: 0,
          error: 0,
          pending: 0,
          cancelled: 0
        },
        byCategory: {},
        recentCount: 0
      };
    }

    const withLocation = images.filter(img => !!img.location).length;
    const withObjects = images.filter(img => img.metadata.objects && img.metadata.objects.length > 0).length;
    
    const confidenceSum = images.reduce((sum, img) => {
      return sum + (img.metadata.confidence || 0);
    }, 0);
    const avgConfidence = confidenceSum / total;

    const sessions = new Set(images.map(img => img.sessionId)).size;
    
    const allTags = new Set<string>();
    images.forEach(img => {
      img.metadata.tags?.forEach(tag => allTags.add(tag));
    });

    const byStatus = images.reduce((acc, img) => {
      acc[img.status] = (acc[img.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const byCategory = images.reduce((acc, img) => {
      img.metadata.categories?.forEach(category => {
        acc[category] = (acc[category] || 0) + 1;
      });
      return acc;
    }, {} as Record<string, number>);

    const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const recentCount = images.filter(img => new Date(img.createdAt) > twentyFourHoursAgo).length;

    return {
      total,
      withLocation,
      withObjects,
      avgConfidence,
      sessions,
      tags: allTags.size,
      locationPercentage: total > 0 ? (withLocation / total) * 100 : 0,
      byStatus: {
        processing: byStatus.processing || 0,
        completed: byStatus.completed || 0,
        error: byStatus.error || 0,
        pending: byStatus.pending || 0,
        cancelled: byStatus.cancelled || 0
      },
      byCategory,
      recentCount
    };
  };

  // State setters
  setLoading = (loading: boolean): void => {
    this.state.isLoading = loading;
    this.notify();
  };

  setError = (error: string | null): void => {
    this.state.error = error;
    this.notify();
  };
}

// Create global store instance
export const imageStore = new SimpleImageStore();

// Export store instance as default
export default imageStore;
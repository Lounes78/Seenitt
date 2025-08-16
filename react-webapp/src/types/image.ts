// Core image data structure
export interface ImageData {
  id: string;
  sessionId: string;
  imageUrl: string;
  thumbnailUrl?: string;
  location?: ImageLocation;
  metadata: ImageMetadata;
  status: ProcessingStatus;
  createdAt: Date;
  updatedAt?: Date;
}

// Geographic location data
export interface ImageLocation {
  latitude: number;
  longitude: number;
  accuracy?: number; // GPS accuracy in meters
  altitude?: number;
  heading?: number; // Compass direction
  speed?: number; // Speed in m/s
  timestamp?: number; // GPS timestamp
}

// Rich metadata from your Python processing pipeline
export interface ImageMetadata {
  // Processing info
  timestamp: number;
  processingTime?: number; // Processing duration in ms
  confidence?: number; // Overall confidence score (0-1)
  
  // Object detection results
  objects?: DetectedObject[];
  
  // Image classification
  tags?: string[];
  categories?: string[];
  
  // Technical details
  imageSize?: {
    width: number;
    height: number;
  };
  fileSize?: number; // In bytes
  format?: string; // 'jpg', 'png', etc.
  
  // Camera/device info
  deviceInfo?: {
    model?: string;
    manufacturer?: string;
    os?: string;
  };
  
  // Custom metadata from your pipeline
  [key: string]: any;
}

// Object detection result
export interface DetectedObject {
  id: string;
  name: string;
  confidence: number; // 0-1
  boundingBox: BoundingBox;
  category?: string;
  attributes?: ObjectAttribute[];
}

// Bounding box for detected objects
export interface BoundingBox {
  x: number; // Left edge (0-1 normalized)
  y: number; // Top edge (0-1 normalized) 
  width: number; // Width (0-1 normalized)
  height: number; // Height (0-1 normalized)
}

// Object attributes (color, size, etc.)
export interface ObjectAttribute {
  name: string;
  value: string;
  confidence?: number;
}

// Processing status from your backend
export type ProcessingStatus = 
  | 'processing' 
  | 'completed' 
  | 'error' 
  | 'pending'
  | 'cancelled';

// Image filtering options
export interface ImageFilter {
  sessionId?: string | null;
  dateRange?: DateRange | null;
  hasLocation?: boolean | null;
  tags?: string[];
  confidence?: number | null; // Minimum confidence threshold
  categories?: string[];
  objects?: string[]; // Filter by detected object names
  boundingBox?: BoundingBox | null; // Geographic bounding box
}

// Date range filter
export interface DateRange {
  start: Date;
  end: Date;
}

// Sorting options for image list
export type SortOption = 
  | 'newest'
  | 'oldest' 
  | 'confidence'
  | 'location'
  | 'name'
  | 'size'
  | 'processing-time';

// Image selection state
export interface ImageSelection {
  selectedIds: string[];
  lastSelectedId?: string;
  selectionMode: 'single' | 'multiple';
}

// Image display options
export interface ImageDisplayOptions {
  showMetadata: boolean;
  showConfidence: boolean;
  showBoundingBoxes: boolean;
  showCoordinates: boolean;
  thumbnailSize: 'small' | 'medium' | 'large';
  gridColumns?: number;
}

// Image processing queue item
export interface ImageProcessingItem {
  id: string;
  sessionId: string;
  imageUrl: string;
  status: ProcessingStatus;
  priority: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  error?: string;
}

// Image statistics
export interface ImageStats {
  total: number;
  withLocation: number;
  withObjects: number;
  avgConfidence: number;
  sessions: number;
  tags: number;
  locationPercentage: number;
  byStatus: Record<ProcessingStatus, number>;
  byCategory: Record<string, number>;
  recentCount: number; // Last 24h
}

// Image search result
export interface ImageSearchResult {
  images: ImageData[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
  searchTime: number; // Search duration in ms
}

// Image similarity result
export interface SimilarImage {
  image: ImageData;
  similarity: number; // 0-1 similarity score
  matchingObjects?: string[];
  matchingTags?: string[];
}

// Bulk operations
export type BulkOperation = 
  | 'delete'
  | 'tag' 
  | 'untag'
  | 'move-session'
  | 'export'
  | 'reprocess';

export interface BulkOperationResult {
  operation: BulkOperation;
  successful: string[]; // Image IDs
  failed: string[]; // Image IDs
  errors: Record<string, string>; // imageId -> error message
  totalProcessed: number;
}

// Image export options
export interface ImageExportOptions {
  format: 'json' | 'csv' | 'zip';
  includeMetadata: boolean;
  includeImages: boolean;
  includeThumbnails: boolean;
  filterOptions?: ImageFilter;
}

// Image import result
export interface ImageImportResult {
  imported: number;
  skipped: number;
  errors: number;
  duplicates: number;
  details: {
    successful: string[];
    failed: { file: string; error: string }[];
    duplicated: string[];
  };
}

// Real-time image update
export interface ImageUpdate {
  type: 'added' | 'updated' | 'deleted' | 'status-changed';
  imageId: string;
  sessionId: string;
  data?: Partial<ImageData>;
  timestamp: number;
}

// Image annotation (for manual tagging/corrections)
export interface ImageAnnotation {
  id: string;
  imageId: string;
  type: 'tag' | 'object' | 'correction' | 'note';
  data: {
    tags?: string[];
    objects?: DetectedObject[];
    note?: string;
    correction?: string;
  };
  userId?: string;
  createdAt: Date;
}

// Image quality metrics
export interface ImageQualityMetrics {
  sharpness: number; // 0-1
  brightness: number; // 0-1
  contrast: number; // 0-1
  saturation: number; // 0-1
  noise: number; // 0-1 (higher = more noise)
  overallQuality: number; // 0-1 computed score
}

// Utility type for partial image updates
export type ImageUpdatePayload = Partial<Omit<ImageData, 'id' | 'createdAt'>>;

// Type guards
export const isImageData = (obj: any): obj is ImageData => {
  return obj && 
    typeof obj.id === 'string' &&
    typeof obj.sessionId === 'string' &&
    typeof obj.imageUrl === 'string' &&
    obj.metadata &&
    obj.status &&
    obj.createdAt instanceof Date;
};

export const hasLocation = (image: ImageData): image is ImageData & { location: ImageLocation } => {
  return image.location !== undefined;
};

export const isProcessed = (image: ImageData): boolean => {
  return image.status === 'completed';
};

export const hasObjects = (image: ImageData): boolean => {
  return image.metadata.objects !== undefined && image.metadata.objects.length > 0;
};

// Default values
export const createDefaultImageFilter = (): ImageFilter => ({
  sessionId: null,
  dateRange: null,
  hasLocation: null,
  tags: [],
  confidence: null,
  categories: [],
  objects: []
});

export const createDefaultDisplayOptions = (): ImageDisplayOptions => ({
  showMetadata: true,
  showConfidence: true,
  showBoundingBoxes: true,
  showCoordinates: false,
  thumbnailSize: 'medium',
  gridColumns: 4
});
import { ImageData, ProcessingStatus } from './image';
import { LocationData } from './location';

// Connection status for SSE and general connectivity
export type ConnectionStatus = 
  | 'disconnected' 
  | 'connecting' 
  | 'connected' 
  | 'reconnecting'
  | 'error';

// View modes for the application
export type ViewMode = 'dashboard' | 'map' | 'gallery';

// Backend health check response
export interface HealthCheckResponse {
  status: 'ok' | 'degraded' | 'error';
  timestamp: number;
  processingSessions: number;
  uptime?: number;
  version?: string;
  services?: {
    database?: 'healthy' | 'degraded' | 'down';
    python?: 'healthy' | 'degraded' | 'down';
    camera?: 'healthy' | 'degraded' | 'down';
    storage?: 'healthy' | 'degraded' | 'down';
  };
  metrics?: {
    memoryUsage: number;
    cpuUsage: number;
    diskSpace: number;
    activeConnections: number;
  };
}

// Processing result from your Python pipeline (matches your backend interface)
export interface ProcessingResult {
  sessionId: string;
  timestamp: number;
  results: ProcessingResultData;
  status: ProcessingStatus;
  message?: string;
  processingTime?: number;
  error?: string;
}

// Detailed processing result data from Python
export interface ProcessingResultData {
  // Image paths/URLs
  imageUrl?: string;
  image_path?: string;
  thumbnailUrl?: string;
  thumbnail_path?: string;
  
  // Location data (flexible format from Python)
  location?: {
    lat?: number;
    latitude?: number;
    lng?: number;
    longitude?: number;
    accuracy?: number;
    timestamp?: number;
  };
  
  // Object detection results
  objects?: Array<{
    id?: string;
    name: string;
    confidence: number;
    bbox?: [number, number, number, number]; // [x, y, width, height]
    bounding_box?: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
    category?: string;
  }>;
  
  // Classification results
  tags?: string[];
  categories?: string[];
  confidence?: number;
  
  // Technical metadata
  processing_time?: number;
  image_size?: {
    width: number;
    height: number;
  };
  file_size?: number;
  format?: string;
  
  // Custom metadata from your pipeline
  metadata?: Record<string, any>;
  
  // Error information
  error?: string;
  warnings?: string[];
}

// SSE message types from your backend
export interface SSEMessage {
  type: SSEMessageType;
  timestamp: number;
  userId?: string;
  sessionId?: string;
  data?: any;
}

export type SSEMessageType =
  | 'connected'
  | 'session_started'
  | 'processing_result'
  | 'processing_error'
  | 'session_ended'
  | 'heartbeat'
  | 'reconnect'
  | 'error';

// Session started message data
export interface SessionStartedData {
  sessionId: string;
  streamUrl: string;
  userId: string;
  timestamp: number;
  cameraInfo?: {
    resolution?: string;
    fps?: number;
    codec?: string;
  };
}

// Processing error message data  
export interface ProcessingErrorData {
  sessionId: string;
  message: string;
  error: string;
  timestamp: number;
  recoverable?: boolean;
}

// Session management
export interface AppSession {
  sessionId: string;
  userId: string;
  startTime: number;
  endTime?: number;
  streamUrl?: string;
  status: 'active' | 'ended' | 'error';
  imageCount: number;
  lastActivity: number;
}

// API request/response interfaces
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: number;
  requestId?: string;
}

// Get results API response
export interface GetResultsResponse {
  sessionId: string;
  results: ProcessingResult[];
  total: number;
  hasMore: boolean;
  nextCursor?: string;
}

// Manual processing request
export interface ProcessingRequest {
  streamUrl: string;
  sessionId: string;
  options?: {
    enableObjectDetection?: boolean;
    enableGeotagging?: boolean;
    confidenceThreshold?: number;
    outputFormat?: 'json' | 'xml';
    maxProcessingTime?: number;
  };
}

// API error response
export interface ApiError {
  code: string;
  message: string;
  details?: any;
  timestamp: number;
  requestId?: string;
  stack?: string; // Only in development
}

// Common API error codes
export enum ApiErrorCode {
  UNAUTHORIZED = 'UNAUTHORIZED',
  FORBIDDEN = 'FORBIDDEN', 
  NOT_FOUND = 'NOT_FOUND',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  PROCESSING_ERROR = 'PROCESSING_ERROR',
  STREAM_ERROR = 'STREAM_ERROR',
  SESSION_NOT_FOUND = 'SESSION_NOT_FOUND',
  RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED',
  INTERNAL_ERROR = 'INTERNAL_ERROR',
  SERVICE_UNAVAILABLE = 'SERVICE_UNAVAILABLE'
}

// WebSocket/SSE connection configuration
export interface ConnectionConfig {
  baseUrl?: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
  timeout: number;
  withCredentials: boolean;
}

// API client configuration
export interface ApiClientConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  headers?: Record<string, string>;
  auth?: {
    type: 'bearer' | 'api-key' | 'basic';
    token?: string;
    apiKey?: string;
    username?: string;
    password?: string;
  };
}

// File upload progress
export interface UploadProgress {
  sessionId: string;
  fileName: string;
  loaded: number;
  total: number;
  percentage: number;
  speed: number; // bytes per second
  timeRemaining: number; // seconds
  status: 'uploading' | 'completed' | 'error' | 'cancelled';
  error?: string;
}

// Batch processing request
export interface BatchProcessingRequest {
  files: File[];
  sessionId: string;
  options?: ProcessingRequest['options'];
  onProgress?: (progress: UploadProgress) => void;
  onComplete?: (results: ProcessingResult[]) => void;
  onError?: (error: ApiError) => void;
}

// Real-time metrics
export interface RealtimeMetrics {
  timestamp: number;
  connections: number;
  activeSessions: number;
  processingQueue: number;
  avgProcessingTime: number;
  errorRate: number;
  throughput: number; // images per minute
  systemLoad: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
}

// Pagination parameters
export interface PaginationParams {
  page?: number;
  limit?: number;
  cursor?: string;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

// Search parameters
export interface SearchParams extends PaginationParams {
  query?: string;
  filters?: {
    sessionId?: string;
    dateFrom?: Date;
    dateTo?: Date;
    hasLocation?: boolean;
    minConfidence?: number;
    tags?: string[];
    status?: ProcessingStatus[];
  };
}

// Export request
export interface ExportRequest {
  format: 'json' | 'csv' | 'zip';
  sessionIds?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  includeImages?: boolean;
  includeThumbnails?: boolean;
  includeMetadata?: boolean;
}

// Export response
export interface ExportResponse {
  downloadUrl: string;
  fileName: string;
  fileSize: number;
  expiresAt: Date;
  format: string;
  itemCount: number;
}

// Statistics request
export interface StatsRequest {
  sessionIds?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  groupBy?: 'session' | 'day' | 'hour' | 'tag' | 'location';
}

// Statistics response
export interface StatsResponse {
  total: number;
  processed: number;
  errors: number;
  avgProcessingTime: number;
  locationCoverage: number;
  topTags: Array<{ tag: string; count: number }>;
  topObjects: Array<{ object: string; count: number }>;
  processingSessions: number;
  timeRange: {
    start: Date;
    end: Date;
  };
  breakdown?: Record<string, number>;
}

// WebSocket message format (if using WebSocket instead of SSE)
export interface WebSocketMessage {
  id: string;
  type: string;
  payload: any;
  timestamp: number;
  sessionId?: string;
}

// Retry configuration
export interface RetryConfig {
  maxAttempts: number;
  baseDelay: number; // milliseconds
  maxDelay: number; // milliseconds  
  backoffFactor: number;
  retryOn: number[]; // HTTP status codes to retry
}

// Cache configuration
export interface CacheConfig {
  enabled: boolean;
  ttl: number; // Time to live in seconds
  maxSize: number; // Maximum cache entries
  strategy: 'lru' | 'fifo' | 'lifo';
}

// Type guards
export const isApiError = (obj: any): obj is ApiError => {
  return obj && typeof obj.code === 'string' && typeof obj.message === 'string';
};

export const isProcessingResult = (obj: any): obj is ProcessingResult => {
  return obj && 
    typeof obj.sessionId === 'string' &&
    typeof obj.timestamp === 'number' &&
    obj.results &&
    obj.status;
};

export const isSSEMessage = (obj: any): obj is SSEMessage => {
  return obj &&
    typeof obj.type === 'string' &&
    typeof obj.timestamp === 'number';
};

// Default configurations
export const DEFAULT_CONNECTION_CONFIG: ConnectionConfig = {
  reconnectInterval: 3000,
  maxReconnectAttempts: 10,
  heartbeatInterval: 30000,
  timeout: 10000,
  withCredentials: true
};

export const DEFAULT_API_CONFIG: ApiClientConfig = {
  baseUrl: '/api',
  timeout: 30000,
  retries: 3,
  headers: {
    'Content-Type': 'application/json'
  }
};

export const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxAttempts: 3,
  baseDelay: 1000,
  maxDelay: 10000,
  backoffFactor: 2,
  retryOn: [408, 429, 500, 502, 503, 504]
};

// HTTP method types
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

// Request configuration
export interface RequestConfig {
  method: HttpMethod;
  url: string;
  data?: any;
  params?: Record<string, any>;
  headers?: Record<string, string>;
  timeout?: number;
  retryConfig?: RetryConfig;
  cacheConfig?: CacheConfig;
}

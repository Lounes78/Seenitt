// Application constants and configuration

// Environment variables
export const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN || '';
export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '/api';
export const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';

// Application settings
export const APP_NAME = 'SeenittApp';
export const APP_VERSION = '1.0.0';

// Map configuration
export const MAP_DEFAULTS = {
  center: {
    lat: 37.7749,
    lng: -122.4194
  },
  zoom: 12,
  maxZoom: 20,
  minZoom: 1,
  style: 'mapbox://styles/mapbox/streets-v12',
  projection: 'mercator' as const
};

// Map interaction settings
export const MAP_SETTINGS = {
  clusterRadius: 50,
  clusterMaxZoom: 14,
  popupMaxWidth: 300,
  popupOffset: 10,
  markerSize: {
    small: 20,
    medium: 30,
    large: 40
  },
  animationDuration: 300
};

// Gallery settings
export const GALLERY_SETTINGS = {
  defaultPageSize: 20,
  maxPageSize: 100,
  thumbnailSizes: {
    small: 150,
    medium: 200,
    large: 250
  },
  gridColumns: {
    small: 6,
    medium: 4,
    large: 3
  },
  loadingThreshold: 200 // px from bottom to trigger load more
};

// Image processing settings
export const PROCESSING_SETTINGS = {
  supportedFormats: ['jpg', 'jpeg', 'png', 'webp', 'gif'],
  maxFileSize: 10 * 1024 * 1024, // 10MB
  maxWidth: 4096,
  maxHeight: 4096,
  qualityThreshold: 0.5, // Minimum confidence threshold
  compressionQuality: 0.8
};

// API settings
export const API_SETTINGS = {
  timeout: 30000, // 30 seconds
  retryAttempts: 3,
  retryDelay: 1000, // 1 second
  maxRetryDelay: 10000, // 10 seconds
  backoffFactor: 2
};

// SSE settings
export const SSE_SETTINGS = {
  endpoint: '/api/stream',
  reconnectInterval: 3000, // 3 seconds
  maxReconnectAttempts: 10,
  heartbeatInterval: 30000, // 30 seconds
  connectionTimeout: 10000 // 10 seconds
};

// Local storage keys
export const STORAGE_KEYS = {
  currentView: 'seenitt:currentView',
  mapSettings: 'seenitt:mapSettings',
  gallerySettings: 'seenitt:gallerySettings',
  userPreferences: 'seenitt:userPreferences',
  sessionData: 'seenitt:sessionData',
  imageFilters: 'seenitt:imageFilters'
};

// Color scheme
export const COLORS = {
  primary: '#007bff',
  secondary: '#6c757d',
  success: '#28a745',
  warning: '#ffc107',
  danger: '#dc3545',
  info: '#17a2b8',
  light: '#f8f9fa',
  dark: '#343a40',
  white: '#ffffff',
  black: '#000000'
};

// Status colors for different states
export const STATUS_COLORS = {
  processing: '#ffc107',
  completed: '#28a745',
  error: '#dc3545',
  pending: '#6c757d',
  cancelled: '#6c757d',
  connected: '#28a745',
  disconnected: '#dc3545',
  connecting: '#ffc107',
  reconnecting: '#17a2b8'
};

// Animation settings
export const ANIMATION = {
  fast: 150,
  normal: 300,
  slow: 500,
  spring: {
    tension: 280,
    friction: 60
  },
  easing: {
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)'
  }
};

// Breakpoints for responsive design
export const BREAKPOINTS = {
  xs: 576,
  sm: 768,
  md: 992,
  lg: 1200,
  xl: 1400
};

// Media queries
export const MEDIA_QUERIES = {
  xs: `(max-width: ${BREAKPOINTS.xs - 1}px)`,
  sm: `(max-width: ${BREAKPOINTS.sm - 1}px)`,
  md: `(max-width: ${BREAKPOINTS.md - 1}px)`,
  lg: `(max-width: ${BREAKPOINTS.lg - 1}px)`,
  xl: `(max-width: ${BREAKPOINTS.xl - 1}px)`
};

// Z-index layers
export const Z_INDEX = {
  base: 1,
  overlay: 10,
  dropdown: 100,
  modal: 1000,
  popup: 1100,
  tooltip: 1200,
  notification: 1300
};

// File type icons mapping
export const FILE_TYPE_ICONS = {
  jpg: '=¼',
  jpeg: '=¼',
  png: '=¼',
  gif: '<ž',
  webp: '=¼',
  svg: '<¨',
  pdf: '=Ä',
  doc: '=Ý',
  docx: '=Ý',
  txt: '=Ä',
  json: '=Ë',
  csv: '=Ê',
  zip: '=æ',
  default: '=Á'
};

// Error messages
export const ERROR_MESSAGES = {
  network: 'Network error. Please check your connection.',
  timeout: 'Request timed out. Please try again.',
  unauthorized: 'You are not authorized to perform this action.',
  forbidden: 'Access denied.',
  notFound: 'Resource not found.',
  serverError: 'Internal server error. Please try again later.',
  unknown: 'An unknown error occurred.',
  mapboxToken: 'Mapbox token is not configured. Please check your environment variables.',
  fileSize: 'File size exceeds the maximum limit.',
  fileType: 'File type is not supported.',
  invalidImage: 'Invalid image file.',
  processingFailed: 'Image processing failed.',
  uploadFailed: 'File upload failed.',
  sessionNotFound: 'Session not found.',
  connectionLost: 'Connection to server lost. Attempting to reconnect...'
};

// Success messages
export const SUCCESS_MESSAGES = {
  imageUploaded: 'Image uploaded successfully.',
  imageProcessed: 'Image processed successfully.',
  sessionCreated: 'Session created successfully.',
  dataExported: 'Data exported successfully.',
  settingsSaved: 'Settings saved successfully.',
  connectionRestored: 'Connection restored.',
  imageDeleted: 'Image deleted successfully.',
  sessionDeleted: 'Session deleted successfully.'
};

// Feature flags
export const FEATURES = {
  enableMapClustering: true,
  enableImageAnnotations: true,
  enableBatchProcessing: true,
  enableExport: true,
  enableNotifications: true,
  enableOfflineMode: false,
  enableAnalytics: false,
  enableDebugMode: process.env.NODE_ENV === 'development'
};

// Development settings
export const DEV_SETTINGS = {
  showPerformanceMetrics: process.env.NODE_ENV === 'development',
  enableMockData: false,
  logLevel: process.env.NODE_ENV === 'development' ? 'debug' : 'warn',
  enableReduxDevTools: process.env.NODE_ENV === 'development'
};

// Regular expressions
export const REGEX = {
  email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  url: /^https?:\/\/.+/,
  sessionId: /^[a-zA-Z0-9-_]{8,}$/,
  imageId: /^[a-zA-Z0-9-_]+$/,
  coordinates: /^-?\d+\.?\d*,-?\d+\.?\d*$/
};

// Date/time formats
export const DATE_FORMATS = {
  short: 'MM/dd/yyyy',
  medium: 'MMM dd, yyyy',
  long: 'MMMM dd, yyyy',
  dateTime: 'MM/dd/yyyy HH:mm',
  time: 'HH:mm:ss',
  iso: "yyyy-MM-dd'T'HH:mm:ss.SSSxxx"
};

// Export all constants as default
export default {
  MAPBOX_TOKEN,
  API_BASE_URL,
  BACKEND_URL,
  APP_NAME,
  APP_VERSION,
  MAP_DEFAULTS,
  MAP_SETTINGS,
  GALLERY_SETTINGS,
  PROCESSING_SETTINGS,
  API_SETTINGS,
  SSE_SETTINGS,
  STORAGE_KEYS,
  COLORS,
  STATUS_COLORS,
  ANIMATION,
  BREAKPOINTS,
  MEDIA_QUERIES,
  Z_INDEX,
  FILE_TYPE_ICONS,
  ERROR_MESSAGES,
  SUCCESS_MESSAGES,
  FEATURES,
  DEV_SETTINGS,
  REGEX,
  DATE_FORMATS
};
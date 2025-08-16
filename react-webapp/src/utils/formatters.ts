// Utility functions for formatting data display

import { DATE_FORMATS, FILE_TYPE_ICONS } from './constants';

/**
 * Format file size in bytes to human readable format
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Format duration in milliseconds to readable format
 */
export const formatDuration = (milliseconds: number): string => {
  if (milliseconds < 1000) {
    return `${milliseconds}ms`;
  }
  
  const seconds = Math.floor(milliseconds / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  
  if (minutes < 60) {
    return remainingSeconds > 0 
      ? `${minutes}m ${remainingSeconds}s`
      : `${minutes}m`;
  }
  
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  
  return remainingMinutes > 0
    ? `${hours}h ${remainingMinutes}m`
    : `${hours}h`;
};

/**
 * Format confidence score as percentage
 */
export const formatConfidence = (confidence: number | undefined): string => {
  if (confidence === undefined || confidence === null) {
    return 'N/A';
  }
  
  return `${Math.round(confidence * 100)}%`;
};

/**
 * Format coordinates to readable string
 */
export const formatCoordinates = (
  latitude: number, 
  longitude: number,
  precision: number = 4
): string => {
  const lat = parseFloat(latitude.toFixed(precision));
  const lng = parseFloat(longitude.toFixed(precision));
  
  const latDir = lat >= 0 ? 'N' : 'S';
  const lngDir = lng >= 0 ? 'E' : 'W';
  
  return `${Math.abs(lat)}°${latDir}, ${Math.abs(lng)}°${lngDir}`;
};

/**
 * Format date with various options
 */
export const formatDate = (
  date: Date | string | number,
  format: keyof typeof DATE_FORMATS = 'medium'
): string => {
  const dateObj = new Date(date);
  
  if (isNaN(dateObj.getTime())) {
    return 'Invalid Date';
  }
  
  const options: Intl.DateTimeFormatOptions = {};
  
  switch (format) {
    case 'short':
      return dateObj.toLocaleDateString('en-US', {
        month: '2-digit',
        day: '2-digit',
        year: 'numeric'
      });
    
    case 'medium':
      return dateObj.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    
    case 'long':
      return dateObj.toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric'
      });
    
    case 'dateTime':
      return dateObj.toLocaleDateString('en-US', {
        month: '2-digit',
        day: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    
    case 'time':
      return dateObj.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    
    case 'iso':
      return dateObj.toISOString();
    
    default:
      return dateObj.toLocaleDateString();
  }
};

/**
 * Format relative time (e.g., "2 minutes ago")
 */
export const formatRelativeTime = (date: Date | string | number): string => {
  const dateObj = new Date(date);
  const now = new Date();
  const diffMs = now.getTime() - dateObj.getTime();
  
  if (diffMs < 0) {
    return 'In the future';
  }
  
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffSeconds < 60) {
    return 'Just now';
  } else if (diffMinutes < 60) {
    return `${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''} ago`;
  } else if (diffHours < 24) {
    return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  } else if (diffDays < 7) {
    return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  } else {
    return formatDate(dateObj, 'medium');
  }
};

/**
 * Format session ID for display (shorter version)
 */
export const formatSessionId = (sessionId: string, length: number = 8): string => {
  if (!sessionId) return 'Unknown';
  return sessionId.length > length 
    ? `${sessionId.substring(0, length)}...`
    : sessionId;
};

/**
 * Format processing status for display
 */
export const formatProcessingStatus = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'processing':
      return 'Processing...';
    case 'completed':
      return 'Completed';
    case 'error':
      return 'Failed';
    case 'pending':
      return 'Pending';
    case 'cancelled':
      return 'Cancelled';
    default:
      return status.charAt(0).toUpperCase() + status.slice(1);
  }
};

/**
 * Format image dimensions
 */
export const formatDimensions = (width: number, height: number): string => {
  return `${width} × ${height}`;
};

/**
 * Format image resolution category
 */
export const formatResolution = (width: number, height: number): string => {
  const totalPixels = width * height;
  
  if (totalPixels >= 8000000) return '4K+';
  if (totalPixels >= 2000000) return 'HD';
  if (totalPixels >= 1000000) return 'Standard';
  return 'Low';
};

/**
 * Format object count for display
 */
export const formatObjectCount = (count: number): string => {
  if (count === 0) return 'No objects';
  if (count === 1) return '1 object';
  return `${count} objects`;
};

/**
 * Format tag list for display
 */
export const formatTags = (tags: string[], maxTags: number = 3): string => {
  if (!tags || tags.length === 0) return 'No tags';
  
  if (tags.length <= maxTags) {
    return tags.join(', ');
  }
  
  const displayTags = tags.slice(0, maxTags);
  const remainingCount = tags.length - maxTags;
  
  return `${displayTags.join(', ')} +${remainingCount} more`;
};

/**
 * Format file extension with icon
 */
export const formatFileType = (filename: string): { extension: string; icon: string } => {
  const extension = filename.split('.').pop()?.toLowerCase() || '';
  const icon = FILE_TYPE_ICONS[extension as keyof typeof FILE_TYPE_ICONS] || FILE_TYPE_ICONS.default;
  
  return { extension: extension.toUpperCase(), icon };
};

/**
 * Format number with proper locale formatting
 */
export const formatNumber = (
  num: number,
  options: Intl.NumberFormatOptions = {}
): string => {
  return new Intl.NumberFormat('en-US', options).format(num);
};

/**
 * Format percentage with proper locale formatting
 */
export const formatPercentage = (
  value: number,
  total: number,
  decimals: number = 1
): string => {
  if (total === 0) return '0%';
  
  const percentage = (value / total) * 100;
  return `${percentage.toFixed(decimals)}%`;
};

/**
 * Format connection status for display
 */
export const formatConnectionStatus = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'connected':
      return 'Connected';
    case 'disconnected':
      return 'Disconnected';
    case 'connecting':
      return 'Connecting...';
    case 'reconnecting':
      return 'Reconnecting...';
    case 'error':
      return 'Connection Error';
    default:
      return 'Unknown';
  }
};

/**
 * Truncate text with ellipsis
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (!text || text.length <= maxLength) return text;
  return text.substring(0, maxLength).trim() + '...';
};

/**
 * Format error message for display
 */
export const formatErrorMessage = (error: Error | string): string => {
  if (typeof error === 'string') {
    return error;
  }
  
  return error.message || 'An unknown error occurred';
};

/**
 * Format API response time
 */
export const formatResponseTime = (timeMs: number): string => {
  if (timeMs < 1000) {
    return `${Math.round(timeMs)}ms`;
  }
  
  return `${(timeMs / 1000).toFixed(1)}s`;
};

/**
 * Format memory usage
 */
export const formatMemoryUsage = (bytes: number): string => {
  return formatFileSize(bytes);
};

/**
 * Format accuracy for GPS coordinates
 */
export const formatAccuracy = (accuracy: number | undefined): string => {
  if (accuracy === undefined || accuracy === null) {
    return 'Unknown';
  }
  
  if (accuracy < 1) {
    return `${Math.round(accuracy * 1000)}mm`;
  } else if (accuracy < 1000) {
    return `${Math.round(accuracy)}m`;
  } else {
    return `${(accuracy / 1000).toFixed(1)}km`;
  }
};

/**
 * Format processing time for display
 */
export const formatProcessingTime = (timeMs: number | undefined): string => {
  if (timeMs === undefined || timeMs === null) {
    return 'Unknown';
  }
  
  return formatDuration(timeMs);
};

/**
 * Format image quality score
 */
export const formatQualityScore = (score: number): string => {
  if (score >= 0.9) return 'Excellent';
  if (score >= 0.8) return 'Very Good';
  if (score >= 0.7) return 'Good';
  if (score >= 0.6) return 'Fair';
  if (score >= 0.5) return 'Poor';
  return 'Very Poor';
};

/**
 * Format list of items with proper grammar
 */
export const formatList = (
  items: string[], 
  conjunction: 'and' | 'or' = 'and'
): string => {
  if (items.length === 0) return '';
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} ${conjunction} ${items[1]}`;
  
  const lastItem = items[items.length - 1];
  const firstItems = items.slice(0, -1);
  
  return `${firstItems.join(', ')}, ${conjunction} ${lastItem}`;
};

/**
 * Format speed for display
 */
export const formatSpeed = (speedMs: number | undefined): string => {
  if (speedMs === undefined || speedMs === null) {
    return 'Unknown';
  }
  
  const speedKmh = speedMs * 3.6;
  
  if (speedKmh < 0.1) {
    return 'Stationary';
  }
  
  return `${speedKmh.toFixed(1)} km/h`;
};

/**
 * Format altitude for display
 */
export const formatAltitude = (altitude: number | undefined): string => {
  if (altitude === undefined || altitude === null) {
    return 'Unknown';
  }
  
  return `${Math.round(altitude)}m`;
};

/**
 * Format compass heading
 */
export const formatHeading = (heading: number | undefined): string => {
  if (heading === undefined || heading === null) {
    return 'Unknown';
  }
  
  const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  const index = Math.round(heading / 22.5) % 16;
  
  return `${Math.round(heading)}° ${directions[index]}`;
};
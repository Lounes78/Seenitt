import { 
  ImageData, 
  DetectedObject, 
  BoundingBox as ImageBoundingBox,
  ImageMetadata,
  ImageStats
} from '../types/image';
import { 
  ProcessingResult, 
  ProcessingResultData,
  SSEMessage 
} from '../types/api';
import { 
  Coordinates, 
  LocationData, 
  BoundingBox as GeoBoundingBox,
  DistanceResult,
  DistanceUnit
} from '../types/location';

/**
 * Data Processing Service
 * Handles conversion, validation, and processing of data between formats
 */
class DataProcessor {

  /**
   * Convert ProcessingResult from backend to ImageData for frontend
   */
  convertProcessingResultToImageData(result: ProcessingResult): ImageData {
    const resultData = result.results;

    // Extract location data (handle different formats from Python)
    const location: LocationData | undefined = this.extractLocation(resultData);

    // Convert objects array
    const objects: DetectedObject[] | undefined = this.convertObjects(resultData.objects);

    // Create metadata
    const metadata: ImageMetadata = {
      timestamp: result.timestamp,
      processingTime: result.processingTime || resultData.processing_time,
      confidence: resultData.confidence,
      objects,
      tags: resultData.tags || [],
      categories: resultData.categories || [],
      imageSize: resultData.image_size,
      fileSize: resultData.file_size,
      format: resultData.format,
      ...resultData.metadata
    };

    return {
      id: `${result.sessionId}-${result.timestamp}`,
      sessionId: result.sessionId,
      imageUrl: resultData.imageUrl || resultData.image_path || '',
      thumbnailUrl: resultData.thumbnailUrl || resultData.thumbnail_path,
      location,
      metadata,
      status: result.status,
      createdAt: new Date(result.timestamp),
      updatedAt: new Date()
    };
  }

  /**
   * Extract location data from various formats
   */
  private extractLocation(data: ProcessingResultData): LocationData | undefined {
    const loc = data.location;
    if (!loc) return undefined;

    const latitude = loc.lat || loc.latitude;
    const longitude = loc.lng || loc.longitude;

    if (typeof latitude !== 'number' || typeof longitude !== 'number') {
      return undefined;
    }

    return {
      latitude,
      longitude,
      accuracy: loc.accuracy,
      timestamp: loc.timestamp
    };
  }

  /**
   * Convert object detection results to standardized format
   */
  private convertObjects(objects?: any[]): DetectedObject[] | undefined {
    if (!objects || !Array.isArray(objects)) return undefined;

    return objects.map((obj, index) => {
      // Handle different bounding box formats
      let boundingBox: ImageBoundingBox;
      
      if (obj.bbox && Array.isArray(obj.bbox) && obj.bbox.length === 4) {
        // Format: [x, y, width, height]
        const [x, y, width, height] = obj.bbox;
        boundingBox = { x, y, width, height };
      } else if (obj.bounding_box) {
        // Format: {x, y, width, height}
        boundingBox = obj.bounding_box;
      } else {
        // Default bounding box
        boundingBox = { x: 0, y: 0, width: 1, height: 1 };
      }

      return {
        id: obj.id || `obj-${index}`,
        name: obj.name,
        confidence: obj.confidence,
        boundingBox,
        category: obj.category
      };
    });
  }

  /**
   * Validate image data integrity
   */
  validateImageData(imageData: ImageData): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Required fields validation
    if (!imageData.id) errors.push('Missing image ID');
    if (!imageData.sessionId) errors.push('Missing session ID');
    if (!imageData.imageUrl) errors.push('Missing image URL');
    if (!imageData.metadata) errors.push('Missing metadata');
    if (!imageData.createdAt) errors.push('Missing creation date');

    // URL validation
    if (imageData.imageUrl && !this.isValidUrl(imageData.imageUrl)) {
      errors.push('Invalid image URL format');
    }

    // Location validation
    if (imageData.location && !this.isValidCoordinates(imageData.location)) {
      errors.push('Invalid coordinates');
    }

    // Confidence validation
    if (imageData.metadata?.confidence !== undefined) {
      const conf = imageData.metadata.confidence;
      if (conf < 0 || conf > 1) {
        errors.push('Confidence must be between 0 and 1');
      }
    }

    // Objects validation
    if (imageData.metadata?.objects) {
      imageData.metadata.objects.forEach((obj, index) => {
        if (!obj.name) errors.push(`Object ${index}: Missing name`);
        if (obj.confidence < 0 || obj.confidence > 1) {
          errors.push(`Object ${index}: Invalid confidence`);
        }
      });
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Calculate statistics from image array
   */
  calculateImageStats(images: ImageData[]): ImageStats {
    const total = images.length;
    const withLocation = images.filter(img => img.location).length;
    const withObjects = images.filter(img => 
      img.metadata.objects && img.metadata.objects.length > 0
    ).length;

    // Calculate average confidence
    const confidenceImages = images.filter(img => 
      img.metadata.confidence !== undefined
    );
    const avgConfidence = confidenceImages.length > 0
      ? confidenceImages.reduce((sum, img) => sum + img.metadata.confidence!, 0) / confidenceImages.length
      : 0;

    // Count unique sessions
    const sessions = new Set(images.map(img => img.sessionId)).size;

    // Count unique tags
    const allTags = new Set<string>();
    images.forEach(img => {
      img.metadata.tags?.forEach(tag => allTags.add(tag));
    });

    // Status breakdown
    const byStatus = images.reduce((acc, img) => {
      acc[img.status] = (acc[img.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Category breakdown
    const byCategory = images.reduce((acc, img) => {
      img.metadata.categories?.forEach(cat => {
        acc[cat] = (acc[cat] || 0) + 1;
      });
      return acc;
    }, {} as Record<string, number>);

    // Recent count (last 24 hours)
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const recentCount = images.filter(img => 
      new Date(img.createdAt) > oneDayAgo
    ).length;

    return {
      total,
      withLocation,
      withObjects,
      avgConfidence: Math.round(avgConfidence * 100) / 100,
      sessions,
      tags: allTags.size,
      locationPercentage: total > 0 ? Math.round((withLocation / total) * 100) : 0,
      byStatus,
      byCategory,
      recentCount
    };
  }

  /**
   * Calculate distance between two coordinates
   */
  calculateDistance(
    coord1: Coordinates, 
    coord2: Coordinates,
    unit: DistanceUnit = 'meters'
  ): DistanceResult {
    const R = 6371000; // Earth's radius in meters
    
    const lat1Rad = this.toRadians(coord1.latitude);
    const lat2Rad = this.toRadians(coord2.latitude);
    const deltaLatRad = this.toRadians(coord2.latitude - coord1.latitude);
    const deltaLngRad = this.toRadians(coord2.longitude - coord1.longitude);

    const a = Math.sin(deltaLatRad / 2) * Math.sin(deltaLatRad / 2) +
              Math.cos(lat1Rad) * Math.cos(lat2Rad) *
              Math.sin(deltaLngRad / 2) * Math.sin(deltaLngRad / 2);
    
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const distanceMeters = R * c;

    // Calculate bearing
    const y = Math.sin(deltaLngRad) * Math.cos(lat2Rad);
    const x = Math.cos(lat1Rad) * Math.sin(lat2Rad) - 
              Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(deltaLngRad);
    const bearingRad = Math.atan2(y, x);
    const bearing = (this.toDegrees(bearingRad) + 360) % 360;

    // Convert to requested unit
    const distance = this.convertDistance(distanceMeters, 'meters', unit);

    return {
      distance,
      unit,
      bearing
    };
  }

  /**
   * Calculate bounding box from coordinates array
   */
  calculateBoundingBox(coordinates: Coordinates[]): GeoBoundingBox {
    if (coordinates.length === 0) {
      return { north: 0, south: 0, east: 0, west: 0 };
    }

    let north = coordinates[0].latitude;
    let south = coordinates[0].latitude;
    let east = coordinates[0].longitude;
    let west = coordinates[0].longitude;

    coordinates.forEach(coord => {
      north = Math.max(north, coord.latitude);
      south = Math.min(south, coord.latitude);
      east = Math.max(east, coord.longitude);
      west = Math.min(west, coord.longitude);
    });

    return { north, south, east, west };
  }

  /**
   * Check if coordinates are within bounding box
   */
  isWithinBounds(coords: Coordinates, bounds: GeoBoundingBox): boolean {
    return coords.latitude >= bounds.south &&
           coords.latitude <= bounds.north &&
           coords.longitude >= bounds.west &&
           coords.longitude <= bounds.east;
  }

  /**
   * Cluster nearby coordinates
   */
  clusterCoordinates(
    coordinates: Array<{ coords: Coordinates; data: any }>,
    maxDistance: number = 100 // meters
  ): Array<{ center: Coordinates; items: any[]; count: number }> {
    const clusters: Array<{ center: Coordinates; items: any[]; count: number }> = [];
    const processed = new Set<number>();

    coordinates.forEach((coord, index) => {
      if (processed.has(index)) return;

      const cluster = {
        center: coord.coords,
        items: [coord.data],
        count: 1
      };

      // Find nearby coordinates
      coordinates.forEach((otherCoord, otherIndex) => {
        if (index === otherIndex || processed.has(otherIndex)) return;

        const distance = this.calculateDistance(coord.coords, otherCoord.coords);
        if (distance.distance <= maxDistance) {
          cluster.items.push(otherCoord.data);
          cluster.count++;
          processed.add(otherIndex);
        }
      });

      // Recalculate center if multiple items
      if (cluster.count > 1) {
        const centerLat = cluster.items.reduce((sum: number, item: any) => 
          sum + item.location.latitude, 0) / cluster.count;
        const centerLng = cluster.items.reduce((sum: number, item: any) => 
          sum + item.location.longitude, 0) / cluster.count;
        cluster.center = { latitude: centerLat, longitude: centerLng };
      }

      clusters.push(cluster);
      processed.add(index);
    });

    return clusters;
  }

  /**
   * Sanitize and format data for export
   */
  prepareExportData(images: ImageData[], format: 'json' | 'csv'): any {
    const sanitizedData = images.map(img => ({
      id: img.id,
      sessionId: img.sessionId,
      imageUrl: img.imageUrl,
      thumbnailUrl: img.thumbnailUrl,
      latitude: img.location?.latitude,
      longitude: img.location?.longitude,
      accuracy: img.location?.accuracy,
      timestamp: img.createdAt.toISOString(),
      status: img.status,
      confidence: img.metadata.confidence,
      tags: img.metadata.tags?.join(';'),
      categories: img.metadata.categories?.join(';'),
      objectCount: img.metadata.objects?.length || 0,
      objects: img.metadata.objects?.map(obj => `${obj.name}:${obj.confidence}`).join(';'),
      processingTime: img.metadata.processingTime,
      imageWidth: img.metadata.imageSize?.width,
      imageHeight: img.metadata.imageSize?.height,
      fileSize: img.metadata.fileSize
    }));

    if (format === 'json') {
      return {
        exportDate: new Date().toISOString(),
        totalImages: images.length,
        images: sanitizedData
      };
    }

    // CSV format
    if (sanitizedData.length === 0) return '';
    
    const headers = Object.keys(sanitizedData[0]).join(',');
    const rows = sanitizedData.map(row => 
      Object.values(row).map(value => 
        typeof value === 'string' && value.includes(',') 
          ? `"${value}"` 
          : value || ''
      ).join(',')
    );

    return [headers, ...rows].join('\n');
  }

  /**
   * Utility methods
   */
  private isValidUrl(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  private isValidCoordinates(coords: Coordinates): boolean {
    return coords.latitude >= -90 && coords.latitude <= 90 &&
           coords.longitude >= -180 && coords.longitude <= 180;
  }

  private toRadians(degrees: number): number {
    return degrees * Math.PI / 180;
  }

  private toDegrees(radians: number): number {
    return radians * 180 / Math.PI;
  }

  private convertDistance(value: number, from: DistanceUnit, to: DistanceUnit): number {
    const toMeters: Record<DistanceUnit, number> = {
      'meters': 1,
      'kilometers': 1000,
      'miles': 1609.344,
      'nautical-miles': 1852
    };

    const meters = value * toMeters[from];
    return meters / toMeters[to];
  }
}

// Create singleton instance
export const dataProcessor = new DataProcessor();
export default dataProcessor;

// Base coordinate system
export interface Coordinates {
  latitude: number;
  longitude: number;
}

// Extended location with additional GPS data
export interface LocationData extends Coordinates {
  accuracy?: number; // GPS accuracy in meters
  altitude?: number; // Altitude in meters
  altitudeAccuracy?: number; // Altitude accuracy in meters
  heading?: number; // Compass direction (0-360 degrees)
  speed?: number; // Speed in m/s
  timestamp?: number; // GPS timestamp
}

// Geographic bounding box for area queries
export interface BoundingBox {
  north: number; // Northern latitude boundary
  south: number; // Southern latitude boundary
  east: number; // Eastern longitude boundary
  west: number; // Western longitude boundary
}

// Map viewport state
export interface MapViewport {
  latitude: number;
  longitude: number;
  zoom: number;
  bearing?: number; // Map rotation
  pitch?: number; // Map tilt
}

// Map bounds (what's currently visible)
export interface MapBounds extends BoundingBox {
  center: Coordinates;
  zoom: number;
}

// Mapbox style configuration
export type MapStyle = 
  | 'streets-v11'
  | 'outdoors-v11' 
  | 'light-v10'
  | 'dark-v10'
  | 'satellite-v9'
  | 'satellite-streets-v11'
  | 'navigation-day-v1'
  | 'navigation-night-v1';

// Map layer configuration
export interface MapLayer {
  id: string;
  type: 'circle' | 'symbol' | 'line' | 'fill' | 'heatmap' | 'cluster';
  source: string;
  paint?: Record<string, any>;
  layout?: Record<string, any>;
  filter?: any[];
  minzoom?: number;
  maxzoom?: number;
}

// Image marker on the map
export interface ImageMarker {
  id: string;
  imageId: string;
  position: Coordinates;
  clustered?: boolean;
  clusterSize?: number;
  selected?: boolean;
  confidence?: number;
  thumbnailUrl?: string;
  sessionId: string;
  timestamp: number;
}

// Marker cluster
export interface MarkerCluster {
  id: string;
  position: Coordinates;
  count: number;
  imageIds: string[];
  bounds: BoundingBox;
  zoom: number;
  selected?: boolean;
}

// Map popup content
export interface MapPopup {
  imageId: string;
  position: Coordinates;
  isOpen: boolean;
  content?: {
    imageUrl: string;
    thumbnailUrl?: string;
    title?: string;
    description?: string;
    metadata?: Record<string, any>;
    timestamp: number;
    confidence?: number;
  };
}

// Heatmap configuration
export interface HeatmapConfig {
  enabled: boolean;
  intensity: number; // 0-1
  radius: number; // Pixel radius
  weight: 'uniform' | 'confidence' | 'object-count';
  gradient: string[]; // Color gradient
  opacity: number; // 0-1
  minZoom?: number;
  maxZoom?: number;
}

// Map drawing/annotation tools
export interface MapDrawing {
  id: string;
  type: 'point' | 'line' | 'polygon' | 'rectangle' | 'circle';
  coordinates: Coordinates | Coordinates[] | Coordinates[][];
  properties?: {
    name?: string;
    description?: string;
    color?: string;
    strokeWidth?: number;
    fillOpacity?: number;
  };
  createdAt: Date;
  updatedAt?: Date;
}

// Geographic search result
export interface GeoSearchResult {
  id: string;
  name: string;
  displayName: string;
  coordinates: Coordinates;
  boundingBox?: BoundingBox;
  type: 'city' | 'country' | 'region' | 'poi' | 'address';
  relevance: number; // 0-1
}

// Route/path information
export interface Route {
  id: string;
  name?: string;
  coordinates: Coordinates[];
  distance: number; // Total distance in meters
  duration?: number; // Estimated duration in seconds
  imageIds: string[]; // Images along this route
  createdAt: Date;
}

// Map interaction events
export interface MapEvent {
  type: 'click' | 'hover' | 'drag' | 'zoom' | 'move';
  coordinates?: Coordinates;
  imageId?: string;
  clusterId?: string;
  timestamp: number;
}

// Map configuration/settings
export interface MapSettings {
  style: MapStyle;
  showClusters: boolean;
  clusterRadius: number; // Pixel radius for clustering
  showHeatmap: boolean;
  heatmapConfig: HeatmapConfig;
  showControls: boolean;
  allowDrawing: boolean;
  defaultZoom: number;
  maxZoom: number;
  minZoom: number;
  center: Coordinates;
  fitBoundsOptions: {
    padding: number;
    maxZoom: number;
    duration: number;
  };
}

// Location-based filtering
export interface LocationFilter {
  center?: Coordinates;
  radius?: number; // Radius in meters
  boundingBox?: BoundingBox;
  country?: string;
  city?: string;
  region?: string;
}

// Geographic statistics
export interface GeoStats {
  totalLocations: number;
  uniqueLocations: number;
  boundingBox: BoundingBox;
  center: Coordinates;
  averageAccuracy: number;
  locationDensity: number; // Locations per km²
  countryCounts: Record<string, number>;
  cityCounts: Record<string, number>;
}

// Distance calculation result
export interface DistanceResult {
  distance: number; // Distance in meters
  unit: 'meters' | 'kilometers' | 'miles';
  bearing: number; // Compass bearing (0-360)
  duration?: number; // Estimated travel time in seconds
}

// Geocoding result (address from coordinates)
export interface GeocodeResult {
  coordinates: Coordinates;
  address: {
    street?: string;
    city?: string;
    region?: string;
    country?: string;
    postalCode?: string;
    formattedAddress: string;
  };
  accuracy: number; // 0-1
  source: 'mapbox' | 'nominatim' | 'google';
}

// Reverse geocoding request
export interface ReverseGeocodeRequest {
  coordinates: Coordinates;
  language?: string;
  types?: string[]; // Address component types to include
}

// Map tile information
export interface TileInfo {
  x: number;
  y: number;
  z: number; // Zoom level
  url: string;
  loaded: boolean;
  error?: string;
}

// Spatial query options
export interface SpatialQuery {
  type: 'within' | 'intersects' | 'nearby' | 'route';
  geometry: BoundingBox | Coordinates | Coordinates[];
  radius?: number; // For 'nearby' queries
  limit?: number;
  offset?: number;
}

// Location sharing/permissions
export interface LocationPermission {
  granted: boolean;
  accuracy: 'exact' | 'approximate' | 'denied';
  timestamp: Date;
  error?: string;
}

// Track/trajectory data
export interface LocationTrack {
  id: string;
  name?: string;
  points: LocationData[];
  totalDistance: number;
  totalDuration: number;
  averageSpeed: number;
  imageIds: string[]; // Associated images
  createdAt: Date;
  updatedAt: Date;
}

// Utility functions types
export type DistanceUnit = 'meters' | 'kilometers' | 'miles' | 'nautical-miles';

export interface DistanceCalculationOptions {
  unit: DistanceUnit;
  precision?: number; // Decimal places
  algorithm?: 'haversine' | 'vincenty'; // Calculation method
}

// Type guards
export const isValidCoordinates = (coords: any): coords is Coordinates => {
  return coords &&
    typeof coords.latitude === 'number' &&
    typeof coords.longitude === 'number' &&
    coords.latitude >= -90 && coords.latitude <= 90 &&
    coords.longitude >= -180 && coords.longitude <= 180;
};

export const isWithinBounds = (coords: Coordinates, bounds: BoundingBox): boolean => {
  return coords.latitude >= bounds.south &&
         coords.latitude <= bounds.north &&
         coords.longitude >= bounds.west &&
         coords.longitude <= bounds.east;
};

export const hasSufficientAccuracy = (location: LocationData, requiredAccuracy: number): boolean => {
  return location.accuracy !== undefined && location.accuracy <= requiredAccuracy;
};

// Default values
export const createDefaultMapSettings = (): MapSettings => ({
  style: 'streets-v11',
  showClusters: true,
  clusterRadius: 50,
  showHeatmap: false,
  heatmapConfig: {
    enabled: false,
    intensity: 0.5,
    radius: 20,
    weight: 'uniform',
    gradient: ['#00f', '#0ff', '#0f0', '#ff0', '#f00'],
    opacity: 0.6
  },
  showControls: true,
  allowDrawing: false,
  defaultZoom: 10,
  maxZoom: 20,
  minZoom: 1,
  center: { latitude: 48.8566, longitude: 2.3522 }, // Paris default
  fitBoundsOptions: {
    padding: 50,
    maxZoom: 15,
    duration: 1000
  }
});

export const createDefaultLocationFilter = (): LocationFilter => ({
  center: undefined,
  radius: undefined,
  boundingBox: undefined,
  country: undefined,
  city: undefined,
  region: undefined
});

// Geographic utility functions for location-based operations

import { ImageLocation } from '../types/location';
import { ImageData } from '../types/image';

/**
 * Calculate distance between two coordinates using Haversine formula
 * Returns distance in meters
 */
export const calculateDistance = (
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
): number => {
  const R = 6371000; // Earth's radius in meters
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaPhi = ((lat2 - lat1) * Math.PI) / 180;
  const deltaLambda = ((lng2 - lng1) * Math.PI) / 180;

  const a =
    Math.sin(deltaPhi / 2) * Math.sin(deltaPhi / 2) +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(deltaLambda / 2) * Math.sin(deltaLambda / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
};

/**
 * Calculate bearing between two coordinates
 * Returns bearing in degrees (0-360)
 */
export const calculateBearing = (
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
): number => {
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaLambda = ((lng2 - lng1) * Math.PI) / 180;

  const x = Math.sin(deltaLambda) * Math.cos(phi2);
  const y = Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);

  const theta = Math.atan2(x, y);

  return ((theta * 180) / Math.PI + 360) % 360;
};

/**
 * Calculate center point of multiple coordinates
 */
export const calculateCenter = (coordinates: Array<{ lat: number; lng: number }>): { lat: number; lng: number } => {
  if (coordinates.length === 0) {
    return { lat: 0, lng: 0 };
  }

  if (coordinates.length === 1) {
    return coordinates[0];
  }

  let x = 0;
  let y = 0;
  let z = 0;

  coordinates.forEach(coord => {
    const lat = (coord.lat * Math.PI) / 180;
    const lng = (coord.lng * Math.PI) / 180;

    x += Math.cos(lat) * Math.cos(lng);
    y += Math.cos(lat) * Math.sin(lng);
    z += Math.sin(lat);
  });

  const total = coordinates.length;
  x /= total;
  y /= total;
  z /= total;

  const centralLng = Math.atan2(y, x);
  const hyp = Math.sqrt(x * x + y * y);
  const centralLat = Math.atan2(z, hyp);

  return {
    lat: (centralLat * 180) / Math.PI,
    lng: (centralLng * 180) / Math.PI
  };
};

/**
 * Calculate bounding box for a set of coordinates
 */
export const calculateBounds = (coordinates: Array<{ lat: number; lng: number }>) => {
  if (coordinates.length === 0) {
    return null;
  }

  let minLat = coordinates[0].lat;
  let maxLat = coordinates[0].lat;
  let minLng = coordinates[0].lng;
  let maxLng = coordinates[0].lng;

  coordinates.forEach(coord => {
    minLat = Math.min(minLat, coord.lat);
    maxLat = Math.max(maxLat, coord.lat);
    minLng = Math.min(minLng, coord.lng);
    maxLng = Math.max(maxLng, coord.lng);
  });

  return {
    southwest: { lat: minLat, lng: minLng },
    northeast: { lat: maxLat, lng: maxLng },
    center: calculateCenter(coordinates),
    span: {
      lat: maxLat - minLat,
      lng: maxLng - minLng
    }
  };
};

/**
 * Check if coordinates are valid
 */
export const isValidCoordinates = (lat: number, lng: number): boolean => {
  return (
    typeof lat === 'number' &&
    typeof lng === 'number' &&
    !isNaN(lat) &&
    !isNaN(lng) &&
    lat >= -90 &&
    lat <= 90 &&
    lng >= -180 &&
    lng <= 180
  );
};

/**
 * Normalize longitude to -180 to 180 range
 */
export const normalizeLongitude = (lng: number): number => {
  return ((lng + 180) % 360) - 180;
};

/**
 * Convert degrees to radians
 */
export const toRadians = (degrees: number): number => {
  return (degrees * Math.PI) / 180;
};

/**
 * Convert radians to degrees
 */
export const toDegrees = (radians: number): number => {
  return (radians * 180) / Math.PI;
};

/**
 * Calculate zoom level based on bounding box and container size
 */
export const calculateZoomLevel = (
  bounds: { southwest: { lat: number; lng: number }; northeast: { lat: number; lng: number } },
  containerWidth: number,
  containerHeight: number,
  padding: number = 50
): number => {
  const WORLD_DIM = { height: 256, width: 256 };
  const ZOOM_MAX = 21;

  function latRad(lat: number): number {
    const sin = Math.sin((lat * Math.PI) / 180);
    const radX2 = Math.log((1 + sin) / (1 - sin)) / 2;
    return Math.max(Math.min(radX2, Math.PI), -Math.PI) / 2;
  }

  function zoom(mapPx: number, worldPx: number, fraction: number): number {
    return Math.floor(Math.log(mapPx / worldPx / fraction) / Math.LN2);
  }

  const latFraction = (latRad(bounds.northeast.lat) - latRad(bounds.southwest.lat)) / Math.PI;
  const lngDiff = bounds.northeast.lng - bounds.southwest.lng;
  const lngFraction = ((lngDiff < 0) ? (lngDiff + 360) : lngDiff) / 360;

  const latZoom = zoom(containerHeight - padding * 2, WORLD_DIM.height, latFraction);
  const lngZoom = zoom(containerWidth - padding * 2, WORLD_DIM.width, lngFraction);

  return Math.min(latZoom, lngZoom, ZOOM_MAX);
};

/**
 * Group nearby coordinates for clustering
 */
export const clusterCoordinates = (
  coordinates: Array<{ lat: number; lng: number; data?: any }>,
  threshold: number = 100 // meters
): Array<{ lat: number; lng: number; count: number; items: any[] }> => {
  const clusters: Array<{ lat: number; lng: number; count: number; items: any[] }> = [];
  const processed: boolean[] = new Array(coordinates.length).fill(false);

  coordinates.forEach((coord, index) => {
    if (processed[index]) return;

    const cluster = {
      lat: coord.lat,
      lng: coord.lng,
      count: 1,
      items: [coord.data || coord]
    };

    processed[index] = true;

    // Find nearby coordinates
    coordinates.forEach((otherCoord, otherIndex) => {
      if (processed[otherIndex] || index === otherIndex) return;

      const distance = calculateDistance(
        coord.lat,
        coord.lng,
        otherCoord.lat,
        otherCoord.lng
      );

      if (distance <= threshold) {
        cluster.items.push(otherCoord.data || otherCoord);
        cluster.count++;
        processed[otherIndex] = true;

        // Update cluster center (weighted average)
        cluster.lat = (cluster.lat * (cluster.count - 1) + otherCoord.lat) / cluster.count;
        cluster.lng = (cluster.lng * (cluster.count - 1) + otherCoord.lng) / cluster.count;
      }
    });

    clusters.push(cluster);
  });

  return clusters;
};

/**
 * Get images within a specific radius of a point
 */
export const getImagesInRadius = (
  images: ImageData[],
  centerLat: number,
  centerLng: number,
  radiusMeters: number
): ImageData[] => {
  return images.filter(image => {
    if (!image.location) return false;

    const distance = calculateDistance(
      centerLat,
      centerLng,
      image.location.latitude,
      image.location.longitude
    );

    return distance <= radiusMeters;
  });
};

/**
 * Get images within a bounding box
 */
export const getImagesInBounds = (
  images: ImageData[],
  bounds: { southwest: { lat: number; lng: number }; northeast: { lat: number; lng: number } }
): ImageData[] => {
  return images.filter(image => {
    if (!image.location) return false;

    const { latitude, longitude } = image.location;
    
    return (
      latitude >= bounds.southwest.lat &&
      latitude <= bounds.northeast.lat &&
      longitude >= bounds.southwest.lng &&
      longitude <= bounds.northeast.lng
    );
  });
};

/**
 * Calculate optimal map view for a set of images
 */
export const calculateOptimalView = (
  images: ImageData[],
  containerWidth: number = 800,
  containerHeight: number = 600,
  minZoom: number = 1,
  maxZoom: number = 18
) => {
  const imagesWithLocation = images.filter(img => img.location);
  
  if (imagesWithLocation.length === 0) {
    return {
      center: { lat: 0, lng: 0 },
      zoom: 2
    };
  }

  if (imagesWithLocation.length === 1) {
    return {
      center: {
        lat: imagesWithLocation[0].location!.latitude,
        lng: imagesWithLocation[0].location!.longitude
      },
      zoom: 15
    };
  }

  const coordinates = imagesWithLocation.map(img => ({
    lat: img.location!.latitude,
    lng: img.location!.longitude
  }));

  const bounds = calculateBounds(coordinates);
  if (!bounds) {
    return {
      center: { lat: 0, lng: 0 },
      zoom: 2
    };
  }

  const zoom = Math.max(
    minZoom,
    Math.min(
      maxZoom,
      calculateZoomLevel(bounds, containerWidth, containerHeight)
    )
  );

  return {
    center: bounds.center,
    zoom,
    bounds
  };
};

/**
 * Convert location object to coordinate array [lng, lat] for mapping libraries
 */
export const locationToCoords = (location: ImageLocation): [number, number] => {
  return [location.longitude, location.latitude];
};

/**
 * Convert coordinate array to location object
 */
export const coordsToLocation = (coords: [number, number], extra?: Partial<ImageLocation>): ImageLocation => {
  return {
    longitude: coords[0],
    latitude: coords[1],
    ...extra
  };
};

/**
 * Format coordinates for URL parameters
 */
export const formatCoordsForUrl = (lat: number, lng: number, precision: number = 6): string => {
  return `${lat.toFixed(precision)},${lng.toFixed(precision)}`;
};

/**
 * Parse coordinates from URL parameters
 */
export const parseCoordsFromUrl = (coordString: string): { lat: number; lng: number } | null => {
  const parts = coordString.split(',');
  if (parts.length !== 2) return null;

  const lat = parseFloat(parts[0]);
  const lng = parseFloat(parts[1]);

  if (isNaN(lat) || isNaN(lng) || !isValidCoordinates(lat, lng)) {
    return null;
  }

  return { lat, lng };
};

/**
 * Get cardinal direction from bearing
 */
export const getCardinalDirection = (bearing: number): string => {
  const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  const index = Math.round(bearing / 22.5) % 16;
  return directions[index];
};

/**
 * Check if a point is inside a polygon
 */
export const isPointInPolygon = (
  point: { lat: number; lng: number },
  polygon: Array<{ lat: number; lng: number }>
): boolean => {
  const x = point.lng;
  const y = point.lat;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].lng;
    const yi = polygon[i].lat;
    const xj = polygon[j].lng;
    const yj = polygon[j].lat;

    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
};

/**
 * Simplify polygon coordinates using Douglas-Peucker algorithm
 */
export const simplifyPolygon = (
  points: Array<{ lat: number; lng: number }>,
  tolerance: number = 0.001
): Array<{ lat: number; lng: number }> => {
  if (points.length <= 2) return points;

  // Find the point with the maximum distance from line between start and end
  let maxDistance = 0;
  let index = 0;
  const end = points.length - 1;

  for (let i = 1; i < end; i++) {
    const distance = perpendicularDistance(points[i], points[0], points[end]);
    if (distance > maxDistance) {
      index = i;
      maxDistance = distance;
    }
  }

  // If max distance is greater than tolerance, recursively simplify
  if (maxDistance > tolerance) {
    const recResults1 = simplifyPolygon(points.slice(0, index + 1), tolerance);
    const recResults2 = simplifyPolygon(points.slice(index), tolerance);

    return [...recResults1.slice(0, -1), ...recResults2];
  } else {
    return [points[0], points[end]];
  }
};

/**
 * Calculate perpendicular distance from point to line
 */
function perpendicularDistance(
  point: { lat: number; lng: number },
  lineStart: { lat: number; lng: number },
  lineEnd: { lat: number; lng: number }
): number {
  const dx = lineEnd.lng - lineStart.lng;
  const dy = lineEnd.lat - lineStart.lat;

  if (dx !== 0 || dy !== 0) {
    const t = ((point.lng - lineStart.lng) * dx + (point.lat - lineStart.lat) * dy) / (dx * dx + dy * dy);

    if (t > 1) {
      return calculateDistance(point.lat, point.lng, lineEnd.lat, lineEnd.lng);
    } else if (t > 0) {
      const projection = {
        lat: lineStart.lat + dy * t,
        lng: lineStart.lng + dx * t
      };
      return calculateDistance(point.lat, point.lng, projection.lat, projection.lng);
    }
  }

  return calculateDistance(point.lat, point.lng, lineStart.lat, lineStart.lng);
}

// // Geographic utility functions for location-based operations

// import { ImageLocation } from '../types/location';
// import { ImageData } from '../types/image';

// /**
//  * Calculate distance between two coordinates using Haversine formula
//  * Returns distance in meters
//  */
// export const calculateDistance = (
//   lat1: number,
//   lng1: number,
//   lat2: number,
//   lng2: number
// ): number => {
//   const R = 6371000; // Earth's radius in meters
//   const �1 = (lat1 * Math.PI) / 180;
//   const �2 = (lat2 * Math.PI) / 180;
//   const �� = ((lat2 - lat1) * Math.PI) / 180;
//   const �� = ((lng2 - lng1) * Math.PI) / 180;

//   const a =
//     Math.sin(�� / 2) * Math.sin(�� / 2) +
//     Math.cos(�1) * Math.cos(�2) * Math.sin(�� / 2) * Math.sin(�� / 2);
//   const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

//   return R * c;
// };

// /**
//  * Calculate bearing between two coordinates
//  * Returns bearing in degrees (0-360)
//  */
// export const calculateBearing = (
//   lat1: number,
//   lng1: number,
//   lat2: number,
//   lng2: number
// ): number => {
//   const �1 = (lat1 * Math.PI) / 180;
//   const �2 = (lat2 * Math.PI) / 180;
//   const �� = ((lng2 - lng1) * Math.PI) / 180;

//   const x = Math.sin(��) * Math.cos(�2);
//   const y = Math.cos(�1) * Math.sin(�2) - Math.sin(�1) * Math.cos(�2) * Math.cos(��);

//   const � = Math.atan2(x, y);

//   return ((� * 180) / Math.PI + 360) % 360;
// };

// /**
//  * Calculate center point of multiple coordinates
//  */
// export const calculateCenter = (coordinates: Array<{ lat: number; lng: number }>): { lat: number; lng: number } => {
//   if (coordinates.length === 0) {
//     return { lat: 0, lng: 0 };
//   }

//   if (coordinates.length === 1) {
//     return coordinates[0];
//   }

//   let x = 0;
//   let y = 0;
//   let z = 0;

//   coordinates.forEach(coord => {
//     const lat = (coord.lat * Math.PI) / 180;
//     const lng = (coord.lng * Math.PI) / 180;

//     x += Math.cos(lat) * Math.cos(lng);
//     y += Math.cos(lat) * Math.sin(lng);
//     z += Math.sin(lat);
//   });

//   const total = coordinates.length;
//   x /= total;
//   y /= total;
//   z /= total;

//   const centralLng = Math.atan2(y, x);
//   const hyp = Math.sqrt(x * x + y * y);
//   const centralLat = Math.atan2(z, hyp);

//   return {
//     lat: (centralLat * 180) / Math.PI,
//     lng: (centralLng * 180) / Math.PI
//   };
// };

// /**
//  * Calculate bounding box for a set of coordinates
//  */
// export const calculateBounds = (coordinates: Array<{ lat: number; lng: number }>) => {
//   if (coordinates.length === 0) {
//     return null;
//   }

//   let minLat = coordinates[0].lat;
//   let maxLat = coordinates[0].lat;
//   let minLng = coordinates[0].lng;
//   let maxLng = coordinates[0].lng;

//   coordinates.forEach(coord => {
//     minLat = Math.min(minLat, coord.lat);
//     maxLat = Math.max(maxLat, coord.lat);
//     minLng = Math.min(minLng, coord.lng);
//     maxLng = Math.max(maxLng, coord.lng);
//   });

//   return {
//     southwest: { lat: minLat, lng: minLng },
//     northeast: { lat: maxLat, lng: maxLng },
//     center: calculateCenter(coordinates),
//     span: {
//       lat: maxLat - minLat,
//       lng: maxLng - minLng
//     }
//   };
// };

// /**
//  * Check if coordinates are valid
//  */
// export const isValidCoordinates = (lat: number, lng: number): boolean => {
//   return (
//     typeof lat === 'number' &&
//     typeof lng === 'number' &&
//     !isNaN(lat) &&
//     !isNaN(lng) &&
//     lat >= -90 &&
//     lat <= 90 &&
//     lng >= -180 &&
//     lng <= 180
//   );
// };

// /**
//  * Normalize longitude to -180 to 180 range
//  */
// export const normalizeLongitude = (lng: number): number => {
//   return ((lng + 180) % 360) - 180;
// };

// /**
//  * Convert degrees to radians
//  */
// export const toRadians = (degrees: number): number => {
//   return (degrees * Math.PI) / 180;
// };

// /**
//  * Convert radians to degrees
//  */
// export const toDegrees = (radians: number): number => {
//   return (radians * 180) / Math.PI;
// };

// /**
//  * Calculate zoom level based on bounding box and container size
//  */
// export const calculateZoomLevel = (
//   bounds: { southwest: { lat: number; lng: number }; northeast: { lat: number; lng: number } },
//   containerWidth: number,
//   containerHeight: number,
//   padding: number = 50
// ): number => {
//   const WORLD_DIM = { height: 256, width: 256 };
//   const ZOOM_MAX = 21;

//   function latRad(lat: number): number {
//     const sin = Math.sin((lat * Math.PI) / 180);
//     const radX2 = Math.log((1 + sin) / (1 - sin)) / 2;
//     return Math.max(Math.min(radX2, Math.PI), -Math.PI) / 2;
//   }

//   function zoom(mapPx: number, worldPx: number, fraction: number): number {
//     return Math.floor(Math.log(mapPx / worldPx / fraction) / Math.LN2);
//   }

//   const latFraction = (latRad(bounds.northeast.lat) - latRad(bounds.southwest.lat)) / Math.PI;
//   const lngDiff = bounds.northeast.lng - bounds.southwest.lng;
//   const lngFraction = ((lngDiff < 0) ? (lngDiff + 360) : lngDiff) / 360;

//   const latZoom = zoom(containerHeight - padding * 2, WORLD_DIM.height, latFraction);
//   const lngZoom = zoom(containerWidth - padding * 2, WORLD_DIM.width, lngFraction);

//   return Math.min(latZoom, lngZoom, ZOOM_MAX);
// };

// /**
//  * Group nearby coordinates for clustering
//  */
// export const clusterCoordinates = (
//   coordinates: Array<{ lat: number; lng: number; data?: any }>,
//   threshold: number = 100 // meters
// ): Array<{ lat: number; lng: number; count: number; items: any[] }> => {
//   const clusters: Array<{ lat: number; lng: number; count: number; items: any[] }> = [];
//   const processed: boolean[] = new Array(coordinates.length).fill(false);

//   coordinates.forEach((coord, index) => {
//     if (processed[index]) return;

//     const cluster = {
//       lat: coord.lat,
//       lng: coord.lng,
//       count: 1,
//       items: [coord.data || coord]
//     };

//     processed[index] = true;

//     // Find nearby coordinates
//     coordinates.forEach((otherCoord, otherIndex) => {
//       if (processed[otherIndex] || index === otherIndex) return;

//       const distance = calculateDistance(
//         coord.lat,
//         coord.lng,
//         otherCoord.lat,
//         otherCoord.lng
//       );

//       if (distance <= threshold) {
//         cluster.items.push(otherCoord.data || otherCoord);
//         cluster.count++;
//         processed[otherIndex] = true;

//         // Update cluster center (weighted average)
//         cluster.lat = (cluster.lat * (cluster.count - 1) + otherCoord.lat) / cluster.count;
//         cluster.lng = (cluster.lng * (cluster.count - 1) + otherCoord.lng) / cluster.count;
//       }
//     });

//     clusters.push(cluster);
//   });

//   return clusters;
// };

// /**
//  * Get images within a specific radius of a point
//  */
// export const getImagesInRadius = (
//   images: ImageData[],
//   centerLat: number,
//   centerLng: number,
//   radiusMeters: number
// ): ImageData[] => {
//   return images.filter(image => {
//     if (!image.location) return false;

//     const distance = calculateDistance(
//       centerLat,
//       centerLng,
//       image.location.latitude,
//       image.location.longitude
//     );

//     return distance <= radiusMeters;
//   });
// };

// /**
//  * Get images within a bounding box
//  */
// export const getImagesInBounds = (
//   images: ImageData[],
//   bounds: { southwest: { lat: number; lng: number }; northeast: { lat: number; lng: number } }
// ): ImageData[] => {
//   return images.filter(image => {
//     if (!image.location) return false;

//     const { latitude, longitude } = image.location;
    
//     return (
//       latitude >= bounds.southwest.lat &&
//       latitude <= bounds.northeast.lat &&
//       longitude >= bounds.southwest.lng &&
//       longitude <= bounds.northeast.lng
//     );
//   });
// };

// /**
//  * Calculate optimal map view for a set of images
//  */
// export const calculateOptimalView = (
//   images: ImageData[],
//   containerWidth: number = 800,
//   containerHeight: number = 600,
//   minZoom: number = 1,
//   maxZoom: number = 18
// ) => {
//   const imagesWithLocation = images.filter(img => img.location);
  
//   if (imagesWithLocation.length === 0) {
//     return {
//       center: { lat: 0, lng: 0 },
//       zoom: 2
//     };
//   }

//   if (imagesWithLocation.length === 1) {
//     return {
//       center: {
//         lat: imagesWithLocation[0].location!.latitude,
//         lng: imagesWithLocation[0].location!.longitude
//       },
//       zoom: 15
//     };
//   }

//   const coordinates = imagesWithLocation.map(img => ({
//     lat: img.location!.latitude,
//     lng: img.location!.longitude
//   }));

//   const bounds = calculateBounds(coordinates);
//   if (!bounds) {
//     return {
//       center: { lat: 0, lng: 0 },
//       zoom: 2
//     };
//   }

//   const zoom = Math.max(
//     minZoom,
//     Math.min(
//       maxZoom,
//       calculateZoomLevel(bounds, containerWidth, containerHeight)
//     )
//   );

//   return {
//     center: bounds.center,
//     zoom,
//     bounds
//   };
// };

// /**
//  * Convert location object to coordinate array [lng, lat] for mapping libraries
//  */
// export const locationToCoords = (location: ImageLocation): [number, number] => {
//   return [location.longitude, location.latitude];
// };

// /**
//  * Convert coordinate array to location object
//  */
// export const coordsToLocation = (coords: [number, number], extra?: Partial<ImageLocation>): ImageLocation => {
//   return {
//     longitude: coords[0],
//     latitude: coords[1],
//     ...extra
//   };
// };

// /**
//  * Format coordinates for URL parameters
//  */
// export const formatCoordsForUrl = (lat: number, lng: number, precision: number = 6): string => {
//   return `${lat.toFixed(precision)},${lng.toFixed(precision)}`;
// };

// /**
//  * Parse coordinates from URL parameters
//  */
// export const parseCoordsFromUrl = (coordString: string): { lat: number; lng: number } | null => {
//   const parts = coordString.split(',');
//   if (parts.length !== 2) return null;

//   const lat = parseFloat(parts[0]);
//   const lng = parseFloat(parts[1]);

//   if (isNaN(lat) || isNaN(lng) || !isValidCoordinates(lat, lng)) {
//     return null;
//   }

//   return { lat, lng };
// };

// /**
//  * Get cardinal direction from bearing
//  */
// export const getCardinalDirection = (bearing: number): string => {
//   const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
//   const index = Math.round(bearing / 22.5) % 16;
//   return directions[index];
// };

// /**
//  * Check if a point is inside a polygon
//  */
// export const isPointInPolygon = (
//   point: { lat: number; lng: number },
//   polygon: Array<{ lat: number; lng: number }>
// ): boolean => {
//   const x = point.lng;
//   const y = point.lat;
//   let inside = false;

//   for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
//     const xi = polygon[i].lng;
//     const yi = polygon[i].lat;
//     const xj = polygon[j].lng;
//     const yj = polygon[j].lat;

//     if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
//       inside = !inside;
//     }
//   }

//   return inside;
// };

// /**
//  * Simplify polygon coordinates using Douglas-Peucker algorithm
//  */
// export const simplifyPolygon = (
//   points: Array<{ lat: number; lng: number }>,
//   tolerance: number = 0.001
// ): Array<{ lat: number; lng: number }> => {
//   if (points.length <= 2) return points;

//   // Find the point with the maximum distance from line between start and end
//   let maxDistance = 0;
//   let index = 0;
//   const end = points.length - 1;

//   for (let i = 1; i < end; i++) {
//     const distance = perpendicularDistance(points[i], points[0], points[end]);
//     if (distance > maxDistance) {
//       index = i;
//       maxDistance = distance;
//     }
//   }

//   // If max distance is greater than tolerance, recursively simplify
//   if (maxDistance > tolerance) {
//     const recResults1 = simplifyPolygon(points.slice(0, index + 1), tolerance);
//     const recResults2 = simplifyPolygon(points.slice(index), tolerance);

//     return [...recResults1.slice(0, -1), ...recResults2];
//   } else {
//     return [points[0], points[end]];
//   }
// };

// /**
//  * Calculate perpendicular distance from point to line
//  */
// function perpendicularDistance(
//   point: { lat: number; lng: number },
//   lineStart: { lat: number; lng: number },
//   lineEnd: { lat: number; lng: number }
// ): number {
//   const dx = lineEnd.lng - lineStart.lng;
//   const dy = lineEnd.lat - lineStart.lat;

//   if (dx !== 0 || dy !== 0) {
//     const t = ((point.lng - lineStart.lng) * dx + (point.lat - lineStart.lat) * dy) / (dx * dx + dy * dy);

//     if (t > 1) {
//       return calculateDistance(point.lat, point.lng, lineEnd.lat, lineEnd.lng);
//     } else if (t > 0) {
//       const projection = {
//         lat: lineStart.lat + dy * t,
//         lng: lineStart.lng + dx * t
//       };
//       return calculateDistance(point.lat, point.lng, projection.lat, projection.lng);
//     }
//   }

//   return calculateDistance(point.lat, point.lng, lineStart.lat, lineStart.lng);
// }
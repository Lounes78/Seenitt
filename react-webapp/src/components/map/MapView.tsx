import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ImageData } from '../../types/image';
import { ImageMarker } from './ImageMarker';
import { ImageCluster } from './ImageCluster';
import { MapPopup } from './MapPopup';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { 
  calculateOptimalView, 
  clusterCoordinates, 
  locationToCoords,
  getImagesInBounds 
} from '../../utils/geoUtils';
import { MAPBOX_TOKEN, MAP_DEFAULTS, MAP_SETTINGS, FEATURES } from '../../utils/constants';

interface MapViewProps {
  images: ImageData[];
  isLoading: boolean;
  onImageSelect: (image: ImageData) => void;
  onViewChange: () => void;
  selectedImage?: ImageData | null;
  className?: string;
}

interface MapState {
  center: { lat: number; lng: number };
  zoom: number;
  bounds?: any;
}

interface ClusterData {
  lat: number;
  lng: number;
  count: number;
  items: ImageData[];
}

interface MapInstance {
  on: (event: string, callback: Function) => void;
  off: (event: string, callback?: Function) => void;
  remove: () => void;
  setCenter: (center: [number, number]) => void;
  setZoom: (zoom: number) => void;
  flyTo: (options: { center?: [number, number]; zoom?: number }) => void;
  fitBounds: (bounds: [number, number, number, number], options?: { padding?: number }) => void;
  getBounds: () => {
    getNorthEast: () => { lat: number; lng: number };
    getSouthWest: () => { lat: number; lng: number };
  };
  project: (lngLat: [number, number]) => { x: number; y: number };
  unproject: (point: { x: number; y: number }) => [number, number];
}

export const MapView: React.FC<MapViewProps> = ({
  images,
  isLoading,
  onImageSelect,
  onViewChange,
  selectedImage,
  className = ''
}) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<MapInstance | null>(null);
  const markersRef = useRef<Map<string, any>>(new Map());
  const clustersRef = useRef<Map<string, any>>(new Map());

  const [mapLoaded, setMapLoaded] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapState, setMapState] = useState<MapState>({
    center: MAP_DEFAULTS.center,
    zoom: MAP_DEFAULTS.zoom
  });
  const [showClusters, setShowClusters] = useState(FEATURES.enableMapClustering);
  const [selectedCluster, setSelectedCluster] = useState<ClusterData | null>(null);
  const [expandedCluster, setExpandedCluster] = useState<string | null>(null);
  const [popupImage, setPopupImage] = useState<ImageData | null>(null);
  const [popupPosition, setPopupPosition] = useState<{ x: number; y: number } | null>(null);

  // Filter images with location
  const imagesWithLocation = images.filter(img => img.location);

  // Calculate clusters if clustering is enabled
  const clusters = showClusters ? 
    clusterCoordinates(
      imagesWithLocation.map(img => ({
        lat: img.location!.latitude,
        lng: img.location!.longitude,
        data: img
      })),
      MAP_SETTINGS.clusterRadius
    ).map(cluster => ({
      ...cluster,
      items: cluster.items as ImageData[]
    })) : [];

  // Individual markers (non-clustered or when clustering is disabled)
  const individualMarkers = showClusters ? 
    clusters.filter(cluster => cluster.count === 1) :
    imagesWithLocation.map(img => ({
      lat: img.location!.latitude,
      lng: img.location!.longitude,
      count: 1,
      items: [img]
    }));

  // Initialize map
  useEffect(() => {
    if (!mapRef.current) {
      return;
    }
    
    if (!MAPBOX_TOKEN) {
      setMapError('Mapbox token not configured');
      return;
    }

    // Load Mapbox GL JS
    const loadMapbox = async () => {
      try {
        console.log('Loading Mapbox with token:', MAPBOX_TOKEN.startsWith('pk.') ? MAPBOX_TOKEN.slice(0, 20) + '...' : MAPBOX_TOKEN);
        
        // Check if we have a real Mapbox token (starts with pk.) vs demo token
        if (MAPBOX_TOKEN.startsWith('pk.') && MAPBOX_TOKEN !== 'pk.demo_token') {
          // Try to load real Mapbox GL JS
          console.log('🗺️ Loading real Mapbox GL JS...');
          
          // Import Mapbox GL JS dynamically
          const mapboxgl = await import('mapbox-gl');
          mapboxgl.accessToken = MAPBOX_TOKEN;
          
          const map = new mapboxgl.Map({
            container: mapRef.current,
            style: MAP_DEFAULTS.style,
            center: [mapState.center.lng, mapState.center.lat],
            zoom: mapState.zoom
          });
          
          // Wait for map to load
          await new Promise((resolve) => {
            map.on('load', resolve);
          });
          
          mapInstanceRef.current = map;
          setMapLoaded(true);
          setMapError(null);
          console.log('✅ Real Mapbox loaded successfully');
          
        } else {
          // Fallback to mock implementation for demo token or invalid tokens
          console.log('📍 Using mock map implementation (demo/invalid token)');
        
          // Simulate loading delay
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          // Create mock map instance
        const mockMap: MapInstance = {
          on: (event: string, callback: Function) => {
            if (event === 'load') {
              setTimeout(callback, 100);
            }
          },
          off: () => {},
          remove: () => {},
          setCenter: (center: [number, number]) => {
            setMapState(prev => ({ 
              ...prev, 
              center: { lat: center[1], lng: center[0] } 
            }));
          },
          setZoom: (zoom: number) => {
            setMapState(prev => ({ ...prev, zoom }));
          },
          flyTo: (options: { center?: [number, number]; zoom?: number }) => {
            setMapState(prev => ({
              ...prev,
              center: options.center ? 
                { lat: options.center[1], lng: options.center[0] } : 
                prev.center,
              zoom: options.zoom || prev.zoom
            }));
          },
          fitBounds: (bounds: [number, number, number, number], options?: { padding?: number }) => {
            // Calculate center from bounds
            const center = {
              lat: (bounds[1] + bounds[3]) / 2,
              lng: (bounds[0] + bounds[2]) / 2
            };
            setMapState(prev => ({ ...prev, center, bounds }));
          },
          getBounds: () => ({
            getNorthEast: () => ({ lat: 90, lng: 180 }),
            getSouthWest: () => ({ lat: -90, lng: -180 })
          }),
          project: (lngLat: [number, number]) => ({ x: 100, y: 100 }),
          unproject: (point: { x: number; y: number }) => [0, 0]
        };

          mapInstanceRef.current = mockMap;
          setMapLoaded(true);
          setMapError(null);
        }

      } catch (error) {
        console.error('Failed to load Mapbox:', error);
        setMapError('Failed to load map');
      }
    };

    loadMapbox();

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
      }
    };
  }, []);

  // Update map when images change
  useEffect(() => {
    if (!mapLoaded || !mapInstanceRef.current || imagesWithLocation.length === 0) {
      return;
    }

    // Calculate optimal view for all images
    const optimalView = calculateOptimalView(
      imagesWithLocation,
      mapRef.current?.clientWidth || 800,
      mapRef.current?.clientHeight || 600
    );

    if (optimalView.bounds) {
      // Fit bounds to show all images
      const bounds: [number, number, number, number] = [
        optimalView.bounds.southwest.lng,
        optimalView.bounds.southwest.lat,
        optimalView.bounds.northeast.lng,
        optimalView.bounds.northeast.lat
      ];
      mapInstanceRef.current.fitBounds(bounds, { padding: 50 });
    } else {
      // Single image or fallback
      mapInstanceRef.current.flyTo({
        center: [optimalView.center.lng, optimalView.center.lat],
        zoom: optimalView.zoom
      });
    }
  }, [imagesWithLocation.length, mapLoaded]);

  // Handle marker click
  const handleMarkerClick = useCallback((image: ImageData, event?: React.MouseEvent) => {
    onImageSelect(image);
    
    if (event) {
      setPopupImage(image);
      setPopupPosition({ x: event.clientX, y: event.clientY });
    }
  }, [onImageSelect]);

  // Handle cluster click
  const handleClusterClick = useCallback((clusterImages: ImageData[]) => {
    if (clusterImages.length === 1) {
      handleMarkerClick(clusterImages[0]);
    } else {
      setSelectedCluster({
        lat: clusterImages[0].location!.latitude,
        lng: clusterImages[0].location!.longitude,
        count: clusterImages.length,
        items: clusterImages
      });
      
      // Zoom to cluster
      if (mapInstanceRef.current) {
        const coords = clusterImages.map(img => ({
          lat: img.location!.latitude,
          lng: img.location!.longitude
        }));
        
        const bounds = coords.reduce((acc, coord) => {
          return {
            minLat: Math.min(acc.minLat, coord.lat),
            maxLat: Math.max(acc.maxLat, coord.lat),
            minLng: Math.min(acc.minLng, coord.lng),
            maxLng: Math.max(acc.maxLng, coord.lng)
          };
        }, {
          minLat: coords[0].lat,
          maxLat: coords[0].lat,
          minLng: coords[0].lng,
          maxLng: coords[0].lng
        });

        const boundsArray: [number, number, number, number] = [
          bounds.minLng, bounds.minLat,
          bounds.maxLng, bounds.maxLat
        ];
        mapInstanceRef.current.fitBounds(boundsArray, { padding: 100 });
      }
    }
  }, [handleMarkerClick]);

  // Handle map controls
  const handleToggleClusters = useCallback(() => {
    setShowClusters(!showClusters);
    setSelectedCluster(null);
    setExpandedCluster(null);
  }, [showClusters]);

  const handleFitToImages = useCallback(() => {
    if (!mapInstanceRef.current || imagesWithLocation.length === 0) return;

    const optimalView = calculateOptimalView(
      imagesWithLocation,
      mapRef.current?.clientWidth || 800,
      mapRef.current?.clientHeight || 600
    );

    if (optimalView.bounds) {
      const bounds: [number, number, number, number] = [
        optimalView.bounds.southwest.lng,
        optimalView.bounds.southwest.lat,
        optimalView.bounds.northeast.lng,
        optimalView.bounds.northeast.lat
      ];
      mapInstanceRef.current.fitBounds(bounds, { padding: 50 });
    }
  }, [imagesWithLocation]);

  const handleClosePopup = useCallback(() => {
    setPopupImage(null);
    setPopupPosition(null);
  }, []);

  if (!MAPBOX_TOKEN) {
    return (
      <div className={`map-view ${className}`}>
        <div className="map-error">
          <h3>Map Configuration Error</h3>
          <p>Mapbox token is not configured. Please check your environment variables.</p>
          <code>REACT_APP_MAPBOX_TOKEN</code>
        </div>
      </div>
    );
  }

  if (mapError) {
    return (
      <div className={`map-view ${className}`}>
        <div className="map-error">
          <h3>Map Error</h3>
          <p>{mapError}</p>
          <button onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`map-view ${className}`}>
      {/* Map controls */}
      <div className="map-controls">
        <div className="controls-group">
          <button
            className={`control-btn ${showClusters ? 'control-btn--active' : ''}`}
            onClick={handleToggleClusters}
            title="Toggle clustering"
          >
            ⚏ Cluster
          </button>
          
          <button
            className="control-btn"
            onClick={handleFitToImages}
            disabled={imagesWithLocation.length === 0}
            title="Fit to all images"
          >
            ⧉ Fit All
          </button>
          
          <button
            className="control-btn"
            onClick={onViewChange}
            title="Switch to gallery view"
          >
            ☰ Gallery
          </button>
        </div>

        <div className="map-info">
          <span className="info-text">
            {imagesWithLocation.length} of {images.length} images with location
          </span>
          {showClusters && clusters.length > 0 && (
            <span className="info-text">
              • {clusters.filter(c => c.count > 1).length} clusters
            </span>
          )}
        </div>
      </div>

      {/* Map container */}
      <div className="map-container">
        {!mapLoaded && (
          <div className="map-loading">
            <LoadingSpinner size="large" />
            <p>Loading map...</p>
          </div>
        )}

        <div 
          ref={mapRef}
          className="map-canvas"
          style={{ opacity: mapLoaded ? 1 : 0 }}
        >
          {/* This would be where Mapbox GL JS renders */}
          <div className="map-placeholder">
            <div className="placeholder-content">
              <h3>Interactive Map</h3>
              <p>Mapbox GL JS would render here</p>
              <div className="placeholder-stats">
                <div>Center: {mapState.center.lat.toFixed(4)}, {mapState.center.lng.toFixed(4)}</div>
                <div>Zoom: {mapState.zoom.toFixed(1)}</div>
                <div>Images: {imagesWithLocation.length}</div>
                {showClusters && <div>Clusters: {clusters.length}</div>}
              </div>
            </div>
          </div>
        </div>

        {/* Render markers and clusters (in real implementation, these would be Mapbox layers) */}
        {mapLoaded && (
          <div className="map-overlays">
            {/* Individual markers */}
            {individualMarkers.map((marker, index) => (
              <ImageMarker
                key={`marker-${marker.items[0].id}`}
                image={marker.items[0]}
                selected={selectedImage?.id === marker.items[0].id}
                onClick={(img) => handleMarkerClick(img)}
                onDoubleClick={(img) => handleMarkerClick(img)}
                className="map-marker"
              />
            ))}

            {/* Clusters */}
            {showClusters && clusters.filter(cluster => cluster.count > 1).map((cluster, index) => (
              <ImageCluster
                key={`cluster-${index}`}
                images={cluster.items}
                center={{ lat: cluster.lat, lng: cluster.lng }}
                onClick={(images) => handleClusterClick(images)}
                onImageClick={(img) => handleMarkerClick(img)}
                expanded={expandedCluster === `cluster-${index}`}
                className="map-cluster"
              />
            ))}
          </div>
        )}

        {/* Loading overlay */}
        {isLoading && (
          <div className="map-loading-overlay">
            <LoadingSpinner size="medium" />
            <span>Updating map...</span>
          </div>
        )}
      </div>

      {/* Popup */}
      {popupImage && (
        <MapPopup
          image={popupImage}
          onClose={handleClosePopup}
          onViewImage={(img) => {
            handleClosePopup();
            onImageSelect(img);
          }}
          onViewGallery={() => {
            handleClosePopup();
            onViewChange();
          }}
          position={popupPosition || undefined}
        />
      )}

      <style>{`
        .map-view {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: #f8f9fa;
          position: relative;
        }

        .map-controls {
          background: white;
          padding: 12px 20px;
          border-bottom: 1px solid #e9ecef;
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 12px;
        }

        .controls-group {
          display: flex;
          gap: 8px;
        }

        .control-btn {
          padding: 8px 16px;
          border: 1px solid #dee2e6;
          border-radius: 6px;
          background: white;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .control-btn:hover:not(:disabled) {
          border-color: #007bff;
          color: #007bff;
        }

        .control-btn--active {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }

        .control-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .map-info {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #6c757d;
        }

        .info-text {
          white-space: nowrap;
        }

        .map-container {
          flex: 1;
          position: relative;
          overflow: hidden;
        }

        .map-loading {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          text-align: center;
          z-index: 1000;
        }

        .map-canvas {
          width: 100%;
          height: 100%;
          transition: opacity 0.3s ease;
        }

        .map-placeholder {
          width: 100%;
          height: 100%;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          text-align: center;
        }

        .placeholder-content h3 {
          margin: 0 0 12px 0;
          font-size: 24px;
        }

        .placeholder-content p {
          margin: 0 0 20px 0;
          opacity: 0.8;
        }

        .placeholder-stats {
          background: rgba(255, 255, 255, 0.1);
          padding: 16px;
          border-radius: 8px;
          text-align: left;
        }

        .placeholder-stats div {
          margin-bottom: 4px;
          font-size: 14px;
        }

        .map-overlays {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
        }

        .map-overlays > * {
          pointer-events: auto;
        }

        .map-loading-overlay {
          position: absolute;
          top: 20px;
          right: 20px;
          background: rgba(255, 255, 255, 0.95);
          padding: 12px 16px;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #2d3748;
        }

        .map-error {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          text-align: center;
          padding: 40px;
          color: #6c757d;
        }

        .map-error h3 {
          font-size: 20px;
          margin: 0 0 12px 0;
          color: #dc3545;
        }

        .map-error p {
          margin: 0 0 16px 0;
          max-width: 400px;
        }

        .map-error code {
          background: #f8f9fa;
          padding: 4px 8px;
          border-radius: 4px;
          font-family: 'Monaco', 'Courier New', monospace;
          margin: 8px 0;
          display: inline-block;
        }

        .map-error button {
          padding: 10px 20px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
        }

        .map-error button:hover {
          background: #0056b3;
        }

        /* Responsive design */
        @media (max-width: 768px) {
          .map-controls {
            padding: 12px 16px;
            flex-direction: column;
            align-items: stretch;
          }

          .controls-group {
            justify-content: center;
            flex-wrap: wrap;
          }

          .map-info {
            justify-content: center;
            text-align: center;
          }

          .info-text {
            font-size: 13px;
          }

          .map-loading-overlay {
            top: 10px;
            right: 10px;
            left: 10px;
            width: auto;
          }
        }
      `}</style>
    </div>
  );
};
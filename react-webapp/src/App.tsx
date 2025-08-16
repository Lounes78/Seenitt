import React, { useState, useEffect, useCallback } from 'react';

// Components
import { Header } from './components/common/Header';
import { LoadingSpinner } from './components/common/LoadingSpinner';
import { StatusIndicator } from './components/common/StatusIndicator';
import { Dashboard } from './components/layout/Dashboard';
import { Navigation } from './components/layout/Navigation';
import { ViewToggle } from './components/layout/ViewToggle';
import { MapView } from './components/map/MapView';
import { GalleryView } from './components/gallery/GalleryView';

// Hooks
import { useSSE } from './hooks/useSSE';
import { useImageData } from './hooks/useImageData';
import { useLocalStorage } from './hooks/useLocalStorage';

// Types
import { ProcessingResult, ConnectionStatus, ViewMode } from './types/api';
import { ImageData } from './types/image';

// Services
import { apiService } from './services/api';

// Utils
import { MAPBOX_TOKEN } from './utils/constants';
 
const App: React.FC = () => {
    // View state management
    const [currentView, setCurrentView] = useLocalStorage<ViewMode>('currentView', 'map');
    const [isInitialized, setIsInitialized] = useState(false);
    const [initError, setInitError] = useState<string | null>(null);

    // Connection and session state
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
    const [streamUrl, setStreamUrl] = useState<string | null>(null);


    const {
        images, 
        addImage, 
        updateImage, 
        clearImages,
        getImagesBySession,
        totalImages 
    } = useImageData();

    // SSE connection for real-time updates
    const {
        isConnected: sseConnected,
        connectionError: sseError,
        connect: connectSSE,
        disconnect: disconnectSSE
    } = useSSE({
        onMessage: handleSSEMessage,
        onError: handleSSEError,
        onConnect: handleSSEConnect,
        onDisconnect: handleSSEDisconnect
    });


    // init the app
    useEffect(() => {
        initializeApp();
        return () => {
            disconnectSSE();
        };
    }, []);

    // Auto-reconnect SSE if disconnected (only if we had a successful connection before)
    useEffect(() => {
    if (isInitialized && !sseConnected && connectionStatus === 'connected') {
        const reconnectTimer = setTimeout(() => {
            console.log('Attempting to reconnect SSE...');
            connectSSE();
        }, 3000);

        return () => clearTimeout(reconnectTimer); // we clean up to avoid memory leaks
    }
    }, [isInitialized, sseConnected, connectionStatus, connectSSE]); // Only run this effect when these values change


    // init the app
    const initializeApp = async () => {
        try {
            setConnectionStatus('connecting');
            
            // Check if Mapbox token is configured
            if (!MAPBOX_TOKEN || MAPBOX_TOKEN === 'pk.demo_token') {
                console.warn('⚠️ Using demo Mapbox token. Map functionality may be limited.');
            }

            // Try backend health check (optional - app can work without backend)
            try {
                const healthStatus = await apiService.getHealth();
                console.log('✅ Backend health check passed:', healthStatus);
                
                // Only connect to SSE if backend is available
                await connectSSE();
                setConnectionStatus('connected');
            } catch (backendError) {
                console.warn('⚠️ Backend not available, running in standalone mode:', backendError);
                setConnectionStatus('disconnected');
                
                // Load mock data in development mode when backend is not available
                if (process.env.NODE_ENV === 'development') {
                    loadMockData();
                }
            }

            setIsInitialized(true);
            setInitError(null);

        } catch(error) {
            console.error('App init failed', error);
            setInitError(error instanceof Error ? error.message : 'Unknown initialization error');
            setConnectionStatus('error');
        }
    };

    // Load mock data for development/demo purposes
    const loadMockData = () => {
        console.log('📊 Loading mock data for standalone demo...');
        
        const mockImages: ImageData[] = [
            {
                id: 'mock-1',
                sessionId: 'demo-session',
                imageUrl: 'https://picsum.photos/800/600?random=1',
                thumbnailUrl: 'https://picsum.photos/200/150?random=1',
                location: {
                    latitude: 37.7749,
                    longitude: -122.4194,
                    accuracy: 10
                },
                metadata: {
                    timestamp: Date.now() - 60000,
                    confidence: 0.85,
                    objects: ['tree', 'building'],
                    tags: ['nature', 'urban'],
                    processingTime: 120
                },
                status: 'completed',
                createdAt: new Date(Date.now() - 60000)
            },
            {
                id: 'mock-2',
                sessionId: 'demo-session',
                imageUrl: 'https://picsum.photos/800/600?random=2',
                thumbnailUrl: 'https://picsum.photos/200/150?random=2',
                location: {
                    latitude: 37.7849,
                    longitude: -122.4094,
                    accuracy: 15
                },
                metadata: {
                    timestamp: Date.now() - 120000,
                    confidence: 0.92,
                    objects: ['car', 'street'],
                    tags: ['transport', 'urban'],
                    processingTime: 95
                },
                status: 'completed',
                createdAt: new Date(Date.now() - 120000)
            },
            {
                id: 'mock-3',
                sessionId: 'demo-session',
                imageUrl: 'https://picsum.photos/800/600?random=3',
                thumbnailUrl: 'https://picsum.photos/200/150?random=3',
                location: {
                    latitude: 37.7649,
                    longitude: -122.4294,
                    accuracy: 8
                },
                metadata: {
                    timestamp: Date.now() - 180000,
                    confidence: 0.78,
                    objects: ['person', 'bicycle'],
                    tags: ['people', 'transport'],
                    processingTime: 150
                },
                status: 'completed',
                createdAt: new Date(Date.now() - 180000)
            }
        ];

        // Add mock images to the store
        mockImages.forEach(image => addImage(image));
        
        // Set a mock session
        setCurrentSessionId('demo-session');
        
        console.log(`✅ Loaded ${mockImages.length} mock images`);
    };


    // handle incominng sse message from backgound
    function handleSSEMessage(data: any): void {
        console.log('SSE message received:', data);

        switch(data.type){
            case 'connected':
                console.log('SSE Connected for user:', data.userId);
                break;

            case 'session_started':
                setCurrentSessionId(data.sessionId);
                setStreamUrl(data.streamUrl);
                console.log('New session started:', data.sessionId);
                break;

            case 'processing_result':
                handleProcessingResult(data);
                break;

            case 'processing_error':
                console.error('Processing error:', data.message);
                break;

            case 'session_ended':
                console.log('⏹️ Session ended:', data.sessionId);
                if (data.sessionId === currentSessionId) {
                    setCurrentSessionId(null);
                    setStreamUrl(null);
                }
                break;
            
            default:
                console.log('🔷 Unknown message type:', data.type);
        }
    }



    // handle the processing result
    const handleProcessingResult  = useCallback((result: ProcessingResult) => {
        try {
            const imageData: ImageData = {
                id: `${result.sessionId}-${result.timestamp}`,
                sessionId: result.sessionId,
                imageUrl: result.results.imageUrl || result.results.image_path,
                thumbnailUrl: result.results.thumbnailUrl || result.results.thumbnail_path,
            
                location: result.results.location ? {
                latitude: result.results.location.lat || result.results.location.latitude,
                longitude: result.results.location.lng || result.results.location.longitude,
                accuracy: result.results.location.accuracy
                } : undefined,
            
                metadata: {
                    timestamp: result.timestamp,
                    confidence: result.results.confidence,
                    objects: result.results.objects || [],
                    tags: result.results.tags || [],
                    processingTime: result.results.processing_time,
                    ...result.results.metadata
                },
        
                status: result.status,
                createdAt: new Date(result.timestamp)
            };

            addImage(imageData);
            console.log('New image processed:', imageData);
        
        } catch(error) {
            console.error('Error processing result:', error, result);
        }

    }, [addImage]); // aka only recreates this func if addImage change

    function handleSSEError(error: Error): void {
        console.error('SSE connection error', error);
        setConnectionStatus('error');
    }

    function handleSSEConnect(): void {
        console.log('SSE Connected');
        setConnectionStatus('connected');
    }

    function handleSSEDisconnect(): void {
        console.log('SSE Disconnected');
        setConnectionStatus('disconnected');
    }

    const handleViewChange = useCallback((view: ViewMode) => {
        setCurrentView(view);
    }, [setCurrentView]);


    // manual refresh handler
    const handleRefresh = useCallback(async () => {
        // Only make API calls if we're connected to backend
        if (connectionStatus === 'connected' && currentSessionId) {
            try {
                const results = await apiService.getResults(currentSessionId);
                console.log('Manual refresh - loaded results:', results);

                // process any missing results
                results.results.forEach((result: ProcessingResult) => {
                    handleProcessingResult(result);
                });

            } catch(error) {
                console.error('Failed to refresh data', error);
            }
        } else {
            console.log('🔄 Refresh skipped - running in standalone mode (no backend connected)');
        }
    }, [connectionStatus, currentSessionId, handleProcessingResult]);


    const handleClearData = useCallback(() => {
        clearImages();
        console.log('All image data cleared')
    }, [clearImages]);


    // Show loading spinner during init
    if (!isInitialized && !initError) {
        return (
            <div className="app-loading">
                <LoadingSpinner size="large" />
                <p>Initializing SeenittApp...</p>
                <p className="loading-subtitle">Connecting to backend and initializing map...</p>
            </div>
        );
    }

    // Show initialization error
    if (initError) {
        return (
            <div className="app-error">
                <h2>Initialization Error</h2>
                <p>{initError}</p>
                <button onClick={() => window.location.reload()} className="retry-button">
                Retry
            </button>
            </div>
        );
    }

    // Get current session images
    const currentSessionImages = currentSessionId ? getImagesBySession(currentSessionId) : images;


    // Map view gets full-screen treatment
    if (currentView === 'map') {
        return (
            <MapView 
                images={currentSessionImages}
                isLoading={connectionStatus === 'connecting'}
                onImageSelect={(image) => console.log('Image selected:', image)}
                onViewChange={() => setCurrentView('gallery')}
            />
        );
    }

    // Other views get centered layout
    return (
        <div className="app">
            {/* Header with status and controls */}
            <Header>
                <StatusIndicator
                    status={connectionStatus}
                    sessionId={currentSessionId}
                    streamUrl={streamUrl}
                    totalImages={totalImages}
                />
                <ViewToggle
                    currentView={currentView}
                    onViewChange={handleViewChange}
                />
            </Header>
        
            {/* Main navigation */}
            <Navigation 
                currentView={currentView}
                onViewChange={handleViewChange}
                onRefresh={handleRefresh}
                onClear={handleClearData}
                imageCount={currentSessionImages.length}
            />

            {/* Main content area */}
            <main className="app-main"> 
                {currentView === 'dashboard' && (
                    <Dashboard 
                        images={images}
                        connectionStatus={connectionStatus}
                        currentSessionId={currentSessionId}
                        streamUrl={streamUrl}
                        onViewChange={handleViewChange}
                    />
                )}
          
                {currentView === 'gallery' && (
                    <GalleryView 
                    images={currentSessionImages}
                    isLoading={connectionStatus === 'connecting'}
                    onImageSelect={(image) => console.log('Image selected:', image)}
                    onViewChange={() => setCurrentView('map')}
                    />
                )}
            </main>


            {/* Development info - only in dev mode */}
            {process.env.NODE_ENV === 'development' && (
                <div className="dev-info">
                <details>
                    <summary>Dev Info</summary>
                    <pre>{JSON.stringify({
                    connectionStatus,
                    sseConnected,
                    currentSessionId,
                    totalImages,
                    currentView,
                    hasStreamUrl: !!streamUrl
                    }, null, 2)}</pre>
                </details>
                </div>
            )}
        </div>
    );
};


export default App;
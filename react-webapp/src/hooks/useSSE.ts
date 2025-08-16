import { useState, useEffect, useCallback, useRef } from 'react';

// Types
interface SSEOptions {
  onMessage?: (data: any) => void;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface SSEState {
  isConnected: boolean;
  connectionError: string | null;
  reconnectAttempts: number;
  lastMessageTime: number | null;
}


export const useSSE = (options: SSEOptions = {}) => {
	// unpack options
  const {
    onMessage,
    onError,
    onConnect,
    onDisconnect,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10
  } = options;

  // State management
  const [state, setState] = useState<SSEState>({
    isConnected: false,
    connectionError: null,
    reconnectAttempts: 0,
    lastMessageTime: null
  });

  // Refs for cleanup and persistence
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  /**
   * Connect to SSE stream
   */
  const connect = useCallback(async () => {
    if (!mountedRef.current) return;

    // Close existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      console.log('Attempting SSE connection...');
      
      // Create new EventSource connection
      const eventSource = new EventSource('/api/stream', {
        withCredentials: true // Important for your authenticated endpoints
      });

      eventSourceRef.current = eventSource;

      // Handle successful connection
      eventSource.onopen = () => {
        if (!mountedRef.current) return;
        
        console.log('SSE Connection established');
        setState(prev => ({
          ...prev,
          isConnected: true,
          connectionError: null,
          reconnectAttempts: 0
        }));
        
        onConnect?.();
      };

      // Handle incoming messages
      eventSource.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const data = JSON.parse(event.data);
          console.log('SSE Message:', data);
          
          setState(prev => ({
            ...prev,
            lastMessageTime: Date.now()
          }));

          onMessage?.(data);
        } catch (error) {
          console.error('Error parsing SSE message:', error, event.data);
          onError?.(new Error('Failed to parse SSE message'));
        }
      };

      // Handle connection errors
      eventSource.onerror = (event) => {
        if (!mountedRef.current) return;

        console.error('SSE Connection error:', event);
        
        const errorMessage = `SSE connection failed (readyState: ${eventSource.readyState})`;
        
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionError: errorMessage
        }));

        onError?.(new Error(errorMessage));
        onDisconnect?.();

        // Auto-reconnect logic
        if (state.reconnectAttempts < maxReconnectAttempts) {
          scheduleReconnect();
        } else {
          console.error('Max reconnection attempts reached');
          setState(prev => ({
            ...prev,
            connectionError: 'Max reconnection attempts reached'
          }));
        }
      };

    } catch (error) {
      console.error('Failed to create SSE connection:', error);
      setState(prev => ({
        ...prev,
        isConnected: false,
        connectionError: error instanceof Error ? error.message : 'Unknown connection error'
      }));
      onError?.(error instanceof Error ? error : new Error('Unknown connection error'));
    }
  }, [onConnect, onMessage, onError, onDisconnect, maxReconnectAttempts, state.reconnectAttempts]);

  /**
   * Disconnect from SSE stream
   */
  const disconnect = useCallback(() => {
    console.log('Disconnecting SSE...');

    // Clear reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close EventSource
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    if (mountedRef.current) {
      setState(prev => ({
        ...prev,
        isConnected: false,
        connectionError: null,
        reconnectAttempts: 0
      }));
      onDisconnect?.();
    }
  }, [onDisconnect]);

  /**
   * Schedule automatic reconnection
   */
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    setState(prev => ({
      ...prev,
      reconnectAttempts: prev.reconnectAttempts + 1
    }));

    const delay = reconnectInterval * Math.pow(1.5, state.reconnectAttempts); // Exponential backoff
    console.log(`Scheduling reconnect in ${delay}ms (attempt ${state.reconnectAttempts + 1}/${maxReconnectAttempts})`);

    reconnectTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, delay);
  }, [connect, reconnectInterval, state.reconnectAttempts, maxReconnectAttempts]);

  /**
   * Manual reconnect function
   */
  const reconnect = useCallback(() => {
    setState(prev => ({ ...prev, reconnectAttempts: 0 }));
    disconnect();
    setTimeout(() => connect(), 100);
  }, [connect, disconnect]);

  /**
   * Check connection health
   */
  const isHealthy = useCallback(() => {
    if (!state.isConnected) return false;
    if (!state.lastMessageTime) return true; // No messages yet, but connected
    
    const timeSinceLastMessage = Date.now() - state.lastMessageTime;
    return timeSinceLastMessage < 60000; // Consider unhealthy if no message for 1 minute
  }, [state.isConnected, state.lastMessageTime]);

  /**
   * Setup and cleanup
   */
  useEffect(() => {
    mountedRef.current = true;
    
    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [disconnect]);

  /**
   * Health check interval
   */
  useEffect(() => {
    if (!state.isConnected) return;

    const healthCheckInterval = setInterval(() => {
      if (!isHealthy()) {
        console.warn('SSE connection appears unhealthy, attempting reconnect...');
        reconnect();
      }
    }, 30000); // Check every 30 seconds

    return () => clearInterval(healthCheckInterval);
  }, [state.isConnected, isHealthy, reconnect]);

  // Return hook interface
  return {
    // Connection state
    isConnected: state.isConnected,
    connectionError: state.connectionError,
    reconnectAttempts: state.reconnectAttempts,
    lastMessageTime: state.lastMessageTime,
    
    // Connection controls
    connect,
    disconnect,
    reconnect,
    
    // Utility functions
    isHealthy: isHealthy()
  };
};

export default useSSE;
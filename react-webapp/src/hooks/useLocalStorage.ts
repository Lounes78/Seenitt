import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for persisting state in localStorage
 * Provides automatic serialization/deserialization and error handling
 */
export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void, () => void] {
  
  // State to store our value
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return initialValue;
    }

    try {
      // Get from local storage by key
      const item = window.localStorage.getItem(key);
      
      // Parse stored json or if none return initialValue
      if (item === null) {
        return initialValue;
      }
      
      return JSON.parse(item);
    } catch (error) {
      // If error also return initialValue
      console.warn(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  // Return a wrapped version of useState's setter function that persists the new value to localStorage
  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      // Allow value to be a function so we have the same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      
      // Save state
      setStoredValue(valueToStore);
      
      // Save to local storage
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      // A more advanced implementation would handle the error case
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  // Function to remove the item from localStorage
  const removeValue = useCallback(() => {
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.removeItem(key);
      }
      setStoredValue(initialValue);
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  }, [key, initialValue]);

  // Listen for changes to the localStorage key from other tabs/windows
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try {
          const newValue = JSON.parse(e.newValue);
          setStoredValue(newValue);
        } catch (error) {
          console.warn(`Error parsing localStorage change for key "${key}":`, error);
        }
      }
    };

    // Listen for storage events
    window.addEventListener('storage', handleStorageChange);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [key]);

  return [storedValue, setValue, removeValue];
}

/**
 * Hook for managing multiple localStorage keys with a prefix
 * Useful for managing user preferences, settings, etc.
 */
export function useLocalStorageState<T extends Record<string, any>>(
  prefix: string,
  initialState: T
) {
  const [state, setState] = useState<T>(() => {
    if (typeof window === 'undefined') {
      return initialState;
    }

    try {
      const stored: Partial<T> = {};
      
      // Load all keys with the prefix
      for (const [key, defaultValue] of Object.entries(initialState)) {
        const storageKey = `${prefix}_${key}`;
        const item = window.localStorage.getItem(storageKey);
        
        if (item !== null) {
          stored[key as keyof T] = JSON.parse(item);
        } else {
          stored[key as keyof T] = defaultValue;
        }
      }
      
      return { ...initialState, ...stored };
    } catch (error) {
      console.warn(`Error loading localStorage state with prefix "${prefix}":`, error);
      return initialState;
    }
  });

  const updateState = useCallback((updates: Partial<T> | ((prevState: T) => Partial<T>)) => {
    setState(prevState => {
      const updatesObj = updates instanceof Function ? updates(prevState) : updates;
      const newState = { ...prevState, ...updatesObj };

      // Save updated keys to localStorage
      if (typeof window !== 'undefined') {
        for (const [key, value] of Object.entries(updatesObj)) {
          const storageKey = `${prefix}_${key}`;
          try {
            window.localStorage.setItem(storageKey, JSON.stringify(value));
          } catch (error) {
            console.error(`Error saving localStorage key "${storageKey}":`, error);
          }
        }
      }

      return newState;
    });
  }, [prefix]);

  const clearState = useCallback(() => {
    if (typeof window !== 'undefined') {
      // Remove all keys with the prefix
      for (const key of Object.keys(initialState)) {
        const storageKey = `${prefix}_${key}`;
        try {
          window.localStorage.removeItem(storageKey);
        } catch (error) {
          console.error(`Error removing localStorage key "${storageKey}":`, error);
        }
      }
    }
    setState(initialState);
  }, [prefix, initialState]);

  return [state, updateState, clearState] as const;
}

/**
 * Hook for managing user preferences with localStorage persistence
 * Perfect for your app's view preferences, map settings, etc.
 */
export function useAppPreferences() {
  const [preferences, updatePreferences, clearPreferences] = useLocalStorageState('seenitt_prefs', {
    // View preferences
    defaultView: 'map' as 'map' | 'gallery' | 'dashboard',
    mapStyle: 'streets-v11' as string,
    galleryGridSize: 'medium' as 'small' | 'medium' | 'large',
    
    // Display preferences
    showImageInfo: true,
    showConfidenceScores: true,
    showCoordinates: false,
    groupBySession: false,
    
    // Map preferences
    mapZoom: 10,
    mapCenter: { lat: 48.8566, lng: 2.3522 }, // Paris default
    clusterImages: true,
    showHeatmap: false,
    
    // Gallery preferences
    sortBy: 'newest' as 'newest' | 'oldest' | 'confidence' | 'location',
    filterTags: [] as string[],
    showThumbnails: true,
    autoRefresh: true,
    
    // Notification preferences
    notifyNewImages: true,
    playSounds: false,
    
    // Performance preferences
    maxImagesInMemory: 1000,
    imageQuality: 'medium' as 'low' | 'medium' | 'high',
    preloadImages: true
  });

  return {
    preferences,
    updatePreferences,
    clearPreferences,
    
    // Convenience getters
    get defaultView() { return preferences.defaultView; },
    get mapSettings() {
      return {
        style: preferences.mapStyle,
        zoom: preferences.mapZoom,
        center: preferences.mapCenter,
        clusterImages: preferences.clusterImages,
        showHeatmap: preferences.showHeatmap
      };
    },
    get gallerySettings() {
      return {
        gridSize: preferences.galleryGridSize,
        sortBy: preferences.sortBy,
        showThumbnails: preferences.showThumbnails,
        filterTags: preferences.filterTags
      };
    },
    get displaySettings() {
      return {
        showImageInfo: preferences.showImageInfo,
        showConfidenceScores: preferences.showConfidenceScores,
        showCoordinates: preferences.showCoordinates,
        groupBySession: preferences.groupBySession
      };
    }
  };
}

export default useLocalStorage;

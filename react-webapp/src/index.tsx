import React from 'react';
import { createRoot} from 'react-dom/client';
import App from './App';

import './styles/global.css';

import { ErrorBoundary } from './components/common/ErrorBoundary';

const isDevelopment = process.env.NODE_ENV === 'development';

const logError = (error: Error, errorInfo: any) => {
  console.error('Application Error:', error);
  console.error('Error Info:', errorInfo);
  
  // In production, you might want to send this to an error reporting service
  if (!isDevelopment) {
    // Example: sendToErrorReporting(error, errorInfo);
  }
};


// getting the root element in the index.html
const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element not found. Make sure there is a div with id="root" in your HTML.');
}

const root = createRoot(rootElement);

// app init function
const initializeApp = async () => {
    try {
        // we wheck if backedn is available
        const healthCheck = await fetch('/api/health').catch(() => null);
        if (!healthCheck?.ok && process.env.NODE_ENV === 'production') {
            console.warn('Backend health check failed. App may have limited functionality.');
        }

        // render the react app
        root.render(
            <React.StrictMode>
                <ErrorBoundary onError={logError}>
                    <App />
                </ErrorBoundary>
            </React.StrictMode>
        );

        // Hide loading screen after React has rendered
        // Small delay to ensure components are mounted
        setTimeout(() => {
        if (window.hideLoadingScreen) {
            window.hideLoadingScreen();
        }
        }, 300);


        console.log('SeenittApp initialized successfully');

    } catch (error) {
        console.error('Failed to initialize SeenittApp:', error);
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
        loadingScreen.innerHTML = `
            <div style="color: #ef4444; text-align: center; padding: 20px;">
            <h2>Application Error</h2>
            <p>Failed to initialize the application.</p>
            <p style="font-size: 14px; opacity: 0.8;">Please refresh the page or contact support.</p>
            <button 
                onclick="window.location.reload()" 
                style="margin-top: 20px; padding: 10px 20px; background: #ef4444; color: white; border: none; border-radius: 5px; cursor: pointer;"
            >
                Refresh Page
            </button>
            </div>
        `;
        }
    }
};


// Handle hot module replacement in development
if (isDevelopment && import.meta.hot) {
  import.meta.hot.accept('./App', () => {
    console.log('Hot reloading App component');
  });
}


// Global error handler for unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  logError(new Error(event.reason), { type: 'unhandledrejection' });
});

// Global error handler for uncaught errors
window.addEventListener('error', (event) => {
  console.error('Uncaught error:', event.error);
  logError(event.error, { type: 'uncaughtError', filename: event.filename, lineno: event.lineno });
});


// and we finally init the app
initializeApp();

// Export for testing purposes
export { initializeApp };
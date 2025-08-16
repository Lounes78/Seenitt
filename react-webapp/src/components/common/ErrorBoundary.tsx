import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo
    });

    // Call the onError prop if provided
    this.props.onError?.(error, errorInfo);
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  private handleReload = () => {
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div className="error-boundary">
          <div className="error-boundary__content">
            <div className="error-boundary__icon">
              ⚠️
            </div>
            
            <h2 className="error-boundary__title">
              Something went wrong
            </h2>
            
            <p className="error-boundary__message">
              An unexpected error occurred while rendering this component.
            </p>

            <div className="error-boundary__actions">
              <button 
                className="error-boundary__button error-boundary__button--primary"
                onClick={this.handleRetry}
              >
                Try Again
              </button>
              
              <button 
                className="error-boundary__button error-boundary__button--secondary"
                onClick={this.handleReload}
              >
                Reload Page
              </button>
            </div>

            {/* Show error details in development */}
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="error-boundary__details">
                <summary>Error Details (Development Only)</summary>
                <div className="error-boundary__error-info">
                  <h4>Error:</h4>
                  <pre>{this.state.error.message}</pre>
                  
                  <h4>Stack Trace:</h4>
                  <pre>{this.state.error.stack}</pre>

                  {this.state.errorInfo && (
                    <>
                      <h4>Component Stack:</h4>
                      <pre>{this.state.errorInfo.componentStack}</pre>
                    </>
                  )}
                </div>
              </details>
            )}

            <div className="error-boundary__help">
              <p>
                If this error persists, please try:
              </p>
              <ul>
                <li>Refreshing the page</li>
                <li>Clearing your browser cache</li>
                <li>Checking your internet connection</li>
                <li>Contacting support if the issue continues</li>
              </ul>
            </div>
          </div>

          <style>{`
            .error-boundary {
              min-height: 400px;
              display: flex;
              align-items: center;
              justify-content: center;
              padding: 20px;
              background-color: #f8f9fa;
            }

            .error-boundary__content {
              max-width: 500px;
              text-align: center;
              background: white;
              padding: 40px;
              border-radius: 12px;
              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .error-boundary__icon {
              font-size: 48px;
              margin-bottom: 20px;
            }

            .error-boundary__title {
              color: #dc3545;
              font-size: 24px;
              font-weight: 600;
              margin-bottom: 16px;
              margin-top: 0;
            }

            .error-boundary__message {
              color: #6c757d;
              font-size: 16px;
              line-height: 1.5;
              margin-bottom: 24px;
            }

            .error-boundary__actions {
              display: flex;
              gap: 12px;
              justify-content: center;
              margin-bottom: 24px;
            }

            .error-boundary__button {
              padding: 12px 24px;
              border-radius: 8px;
              font-size: 14px;
              font-weight: 500;
              cursor: pointer;
              transition: all 0.2s;
              border: none;
            }

            .error-boundary__button--primary {
              background-color: #007bff;
              color: white;
            }

            .error-boundary__button--primary:hover {
              background-color: #0056b3;
            }

            .error-boundary__button--secondary {
              background-color: #6c757d;
              color: white;
            }

            .error-boundary__button--secondary:hover {
              background-color: #545b62;
            }

            .error-boundary__details {
              text-align: left;
              background-color: #f8f9fa;
              padding: 16px;
              border-radius: 8px;
              margin-bottom: 20px;
            }

            .error-boundary__details summary {
              cursor: pointer;
              font-weight: 500;
              margin-bottom: 12px;
            }

            .error-boundary__error-info h4 {
              margin: 16px 0 8px 0;
              color: #dc3545;
            }

            .error-boundary__error-info pre {
              background-color: #e9ecef;
              padding: 12px;
              border-radius: 4px;
              font-size: 12px;
              overflow-x: auto;
              white-space: pre-wrap;
            }

            .error-boundary__help {
              background-color: #f8f9fa;
              padding: 16px;
              border-radius: 8px;
              text-align: left;
            }

            .error-boundary__help p {
              margin: 0 0 12px 0;
              font-weight: 500;
            }

            .error-boundary__help ul {
              margin: 0;
              padding-left: 20px;
            }

            .error-boundary__help li {
              margin-bottom: 4px;
            }
          `}</style>
        </div>
      );
    }

    return this.props.children;
  }
}

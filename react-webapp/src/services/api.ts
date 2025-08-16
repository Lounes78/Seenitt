import {
  ApiResponse,
  ApiError,
  ApiErrorCode,
  HealthCheckResponse,
  GetResultsResponse,
  ProcessingRequest,
  ExportRequest,
  ExportResponse,
  StatsRequest,
  StatsResponse,
  SearchParams,
  HttpMethod,
  RequestConfig,
  DEFAULT_API_CONFIG,
  DEFAULT_RETRY_CONFIG
} from '../types/api';

/**
 * API Service for communicating with your Express backend
 * Handles all REST endpoints and error management
 */
class ApiService {
  private baseUrl: string;
  private timeout: number;
  private retries: number;
  private defaultHeaders: Record<string, string>;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || '/api';
    this.timeout = 30000;
    this.retries = 3;
    this.defaultHeaders = {
      'Content-Type': 'application/json'
    };
  }

  /**
   * Generic HTTP request method with retry logic and error handling
   */
  private async request<T>(config: RequestConfig): Promise<ApiResponse<T>> {
    const {
      method,
      url,
      data,
      params,
      headers = {},
      timeout = this.timeout,
      retryConfig = DEFAULT_RETRY_CONFIG
    } = config;

    const fullUrl = url.startsWith('/') ? `${this.baseUrl}${url}` : url;
    const queryString = params ? this.buildQueryString(params) : '';
    const requestUrl = queryString ? `${fullUrl}?${queryString}` : fullUrl;

    const requestHeaders = {
      ...this.defaultHeaders,
      ...headers
    };

    let lastError: Error;
    let attempt = 0;

    while (attempt <= retryConfig.maxAttempts) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const response = await fetch(requestUrl, {
          method,
          headers: requestHeaders,
          body: data ? JSON.stringify(data) : undefined,
          credentials: 'include',
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        
        return {
          success: true,
          data: result,
          timestamp: Date.now()
        };

      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        attempt++;

        // Don't retry on client errors (4xx) except 408, 429
        if (error instanceof Error && error.message.includes('HTTP 4')) {
          const status = parseInt(error.message.match(/HTTP (\d+)/)?.[1] || '400');
          if (!retryConfig.retryOn.includes(status)) {
            break;
          }
        }

        if (attempt <= retryConfig.maxAttempts) {
          const delay = Math.min(
            retryConfig.baseDelay * Math.pow(retryConfig.backoffFactor, attempt - 1),
            retryConfig.maxDelay
          );
          console.log(`API request failed, retrying in ${delay}ms (attempt ${attempt}/${retryConfig.maxAttempts})`);
          await this.sleep(delay);
        }
      }
    }

    // All retries failed
    console.error(`API request failed after ${attempt} attempts:`, lastError);
    return {
      success: false,
      error: this.formatError(lastError),
      timestamp: Date.now()
    };
  }

  /**
   * Health check - GET /api/health
   */
  async getHealth(): Promise<HealthCheckResponse> {
    const response = await this.request<HealthCheckResponse>({
      method: 'GET',
      url: '/health'
    });

    if (!response.success) {
      throw new Error(response.error || 'Health check failed');
    }

    return response.data!;
  }

  /**
   * Get processing results - GET /api/results/:sessionId
   */
  async getResults(sessionId: string, params?: SearchParams): Promise<GetResultsResponse> {
    const response = await this.request<GetResultsResponse>({
      method: 'GET',
      url: `/results/${sessionId}`,
      params
    });

    if (!response.success) {
      throw new Error(response.error || 'Failed to fetch results');
    }

    return response.data!;
  }

  /**
   * Trigger manual processing - POST /api/process
   */
  async startProcessing(request: ProcessingRequest): Promise<{ status: string; sessionId: string }> {
    const response = await this.request<{ status: string; sessionId: string }>({
      method: 'POST',
      url: '/process',
      data: request
    });

    if (!response.success) {
      throw new Error(response.error || 'Failed to start processing');
    }

    return response.data!;
  }

  /**
   * Get session statistics - GET /api/stats
   */
  async getStats(request?: StatsRequest): Promise<StatsResponse> {
    const response = await this.request<StatsResponse>({
      method: 'GET',
      url: '/stats',
      params: request
    });

    if (!response.success) {
      throw new Error(response.error || 'Failed to fetch statistics');
    }

    return response.data!;
  }

  /**
   * Export data - POST /api/export
   */
  async exportData(request: ExportRequest): Promise<ExportResponse> {
    const response = await this.request<ExportResponse>({
      method: 'POST',
      url: '/export',
      data: request
    });

    if (!response.success) {
      throw new Error(response.error || 'Export failed');
    }

    return response.data!;
  }

  /**
   * Search images - GET /api/search
   */
  async searchImages(params: SearchParams): Promise<GetResultsResponse> {
    const response = await this.request<GetResultsResponse>({
      method: 'GET',
      url: '/search',
      params
    });

    if (!response.success) {
      throw new Error(response.error || 'Search failed');
    }

    return response.data!;
  }

  /**
   * Delete session - DELETE /api/sessions/:sessionId
   */
  async deleteSession(sessionId: string): Promise<{ success: boolean }> {
    const response = await this.request<{ success: boolean }>({
      method: 'DELETE',
      url: `/sessions/${sessionId}`
    });

    if (!response.success) {
      throw new Error(response.error || 'Failed to delete session');
    }

    return response.data!;
  }

  /**
   * Upload image for processing - POST /api/upload
   */
  async uploadImage(
    file: File, 
    sessionId: string,
    onProgress?: (progress: number) => void
  ): Promise<{ imageId: string; status: string }> {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('sessionId', sessionId);

      const xhr = new XMLHttpRequest();

      // Progress tracking
      if (onProgress) {
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            onProgress(percentComplete);
          }
        });
      }

      // Success handler
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            resolve(result);
          } catch (error) {
            reject(new Error('Invalid response format'));
          }
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`));
        }
      });

      // Error handler
      xhr.addEventListener('error', () => {
        reject(new Error('Upload failed due to network error'));
      });

      // Timeout handler
      xhr.timeout = this.timeout;
      xhr.addEventListener('timeout', () => {
        reject(new Error('Upload timed out'));
      });

      // Send request
      xhr.open('POST', `${this.baseUrl}/upload`);
      xhr.withCredentials = true;
      xhr.send(formData);
    });
  }

  /**
   * Download file (images, exports, etc.)
   */
  async downloadFile(url: string, filename?: string): Promise<Blob> {
    const response = await fetch(url, {
      method: 'GET',
      credentials: 'include'
    });

    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }

    const blob = await response.blob();

    // Trigger download if filename provided
    if (filename) {
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    }

    return blob;
  }

  /**
   * Utility methods
   */
  private buildQueryString(params: Record<string, any>): string {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        if (Array.isArray(value)) {
          value.forEach(item => searchParams.append(key, String(item)));
        } else if (value instanceof Date) {
          searchParams.append(key, value.toISOString());
        } else {
          searchParams.append(key, String(value));
        }
      }
    });

    return searchParams.toString();
  }

  private formatError(error: Error): string {
    if (error.name === 'AbortError') {
      return 'Request timeout';
    }
    
    if (error.message.includes('Failed to fetch')) {
      return 'Network error - please check your connection';
    }

    if (error.message.includes('HTTP 401')) {
      return 'Authentication required';
    }

    if (error.message.includes('HTTP 403')) {
      return 'Access denied';
    }

    if (error.message.includes('HTTP 404')) {
      return 'Resource not found';
    }

    if (error.message.includes('HTTP 500')) {
      return 'Internal server error';
    }

    return error.message;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Connection test
   */
  async testConnection(): Promise<boolean> {
    try {
      await this.getHealth();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get API status
   */
  getStatus(): {
    baseUrl: string;
    timeout: number;
    retries: number;
    isOnline: boolean;
  } {
    return {
      baseUrl: this.baseUrl,
      timeout: this.timeout,
      retries: this.retries,
      isOnline: navigator.onLine
    };
  }
}

// Create singleton instance
export const apiService = new ApiService();
export default apiService;

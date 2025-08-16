import React from 'react';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
  text?: string;
  className?: string;
  inline?: boolean;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  color = '#007bff',
  text,
  className = '',
  inline = false
}) => {
  const sizeClasses = {
    small: 'loading-spinner--small',
    medium: 'loading-spinner--medium',
    large: 'loading-spinner--large'
  };

  const containerClass = inline 
    ? 'loading-spinner-container loading-spinner-container--inline'
    : 'loading-spinner-container';

  return (
    <div className={`${containerClass} ${className}`}>
      <div 
        className={`loading-spinner ${sizeClasses[size]}`}
        style={{ borderTopColor: color }}
      >
        <div className="loading-spinner__inner" />
      </div>
      
      {text && (
        <div className="loading-spinner__text">
          {text}
        </div>
      )}

      <style>{`
        .loading-spinner-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 20px;
        }

        .loading-spinner-container--inline {
          display: inline-flex;
          padding: 0;
          margin-right: 8px;
        }

        .loading-spinner {
          border: 3px solid rgba(0, 123, 255, 0.1);
          border-top: 3px solid ${color};
          border-radius: 50%;
          animation: loading-spinner-spin 1s linear infinite;
          position: relative;
        }

        .loading-spinner--small {
          width: 20px;
          height: 20px;
          border-width: 2px;
        }

        .loading-spinner--medium {
          width: 40px;
          height: 40px;
          border-width: 3px;
        }

        .loading-spinner--large {
          width: 60px;
          height: 60px;
          border-width: 4px;
        }

        .loading-spinner__inner {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 60%;
          height: 60%;
          border: 2px solid transparent;
          border-top: 2px solid ${color};
          border-radius: 50%;
          animation: loading-spinner-spin 0.8s linear infinite reverse;
          opacity: 0.6;
        }

        .loading-spinner--small .loading-spinner__inner {
          border-width: 1px;
        }

        .loading-spinner--large .loading-spinner__inner {
          border-width: 3px;
        }

        .loading-spinner__text {
          margin-top: 12px;
          font-size: 14px;
          color: #6c757d;
          text-align: center;
          font-weight: 500;
        }

        .loading-spinner-container--inline .loading-spinner__text {
          margin-top: 0;
          margin-left: 8px;
        }

        @keyframes loading-spinner-spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

// Skeleton loader component
interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
  className?: string;
  count?: number;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = '20px',
  borderRadius = '4px',
  className = '',
  count = 1
}) => {
  const skeletons = Array.from({ length: count }, (_, index) => (
    <div key={index} className={`skeleton ${className}`}>
      <style>{`
        .skeleton {
          width: ${typeof width === 'number' ? `${width}px` : width};
          height: ${typeof height === 'number' ? `${height}px` : height};
          border-radius: ${typeof borderRadius === 'number' ? `${borderRadius}px` : borderRadius};
          background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
          background-size: 200% 100%;
          animation: skeleton-loading 1.5s infinite;
          margin-bottom: 8px;
        }

        .skeleton:last-child {
          margin-bottom: 0;
        }

        @keyframes skeleton-loading {
          0% {
            background-position: 200% 0;
          }
          100% {
            background-position: -200% 0;
          }
        }
      `}</style>
    </div>
  ));

  return count === 1 ? skeletons[0] : <>{skeletons}</>;
};

// Pulsing dot loader
interface PulsingDotsProps {
  color?: string;
  size?: number;
  className?: string;
}

export const PulsingDots: React.FC<PulsingDotsProps> = ({
  color = '#007bff',
  size = 8,
  className = ''
}) => {
  return (
    <div className={`pulsing-dots ${className}`}>
      <div className="pulsing-dots__dot" />
      <div className="pulsing-dots__dot" />
      <div className="pulsing-dots__dot" />

      <style>{`
        .pulsing-dots {
          display: inline-flex;
          align-items: center;
          gap: 4px;
        }

        .pulsing-dots__dot {
          width: ${size}px;
          height: ${size}px;
          border-radius: 50%;
          background-color: ${color};
          animation: pulsing-dots-pulse 1.4s ease-in-out infinite both;
        }

        .pulsing-dots__dot:nth-child(1) {
          animation-delay: -0.32s;
        }

        .pulsing-dots__dot:nth-child(2) {
          animation-delay: -0.16s;
        }

        @keyframes pulsing-dots-pulse {
          0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
          }
          40% {
            transform: scale(1);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
};

// Progress bar component
interface ProgressBarProps {
  progress: number; // 0-100
  height?: number;
  color?: string;
  backgroundColor?: string;
  animated?: boolean;
  showPercentage?: boolean;
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  height = 8,
  color = '#007bff',
  backgroundColor = '#e9ecef',
  animated = false,
  showPercentage = false,
  className = ''
}) => {
  const clampedProgress = Math.max(0, Math.min(100, progress));

  return (
    <div className={`progress-bar ${className}`}>
      {showPercentage && (
        <div className="progress-bar__percentage">
          {Math.round(clampedProgress)}%
        </div>
      )}
      
      <div className="progress-bar__track">
        <div 
          className={`progress-bar__fill ${animated ? 'progress-bar__fill--animated' : ''}`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>

      <style>{`
        .progress-bar {
          width: 100%;
        }

        .progress-bar__percentage {
          font-size: 12px;
          font-weight: 500;
          color: #6c757d;
          margin-bottom: 4px;
          text-align: right;
        }

        .progress-bar__track {
          width: 100%;
          height: ${height}px;
          background-color: ${backgroundColor};
          border-radius: ${height / 2}px;
          overflow: hidden;
          position: relative;
        }

        .progress-bar__fill {
          height: 100%;
          background-color: ${color};
          border-radius: ${height / 2}px;
          transition: width 0.3s ease;
          position: relative;
        }

        .progress-bar__fill--animated::after {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          bottom: 0;
          right: 0;
          background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.4),
            transparent
          );
          animation: progress-bar-shimmer 2s infinite;
        }

        @keyframes progress-bar-shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
      `}</style>
    </div>
  );
};

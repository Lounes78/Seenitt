import React from 'react';
import { ConnectionStatus } from '../../types/api';

interface StatusIndicatorProps {
  status: ConnectionStatus;
  sessionId?: string | null;
  streamUrl?: string | null;
  totalImages: number;
  className?: string;
  compact?: boolean;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  sessionId,
  streamUrl,
  totalImages,
  className = '',
  compact = false
}) => {
  const getStatusConfig = (status: ConnectionStatus) => {
    switch (status) {
      case 'connected':
        return {
          color: '#28a745',
          bgColor: '#d4edda',
          icon: '✓',
          text: 'Connected',
          pulse: false
        };
      case 'connecting':
        return {
          color: '#ffc107',
          bgColor: '#fff3cd',
          icon: '○',
          text: 'Connecting...',
          pulse: true
        };
      case 'reconnecting':
        return {
          color: '#fd7e14',
          bgColor: '#fde2e4',
          icon: '↻',
          text: 'Reconnecting...',
          pulse: true
        };
      case 'disconnected':
        return {
          color: '#6c757d',
          bgColor: '#f8f9fa',
          icon: '○',
          text: 'Disconnected',
          pulse: false
        };
      case 'error':
        return {
          color: '#dc3545',
          bgColor: '#f8d7da',
          icon: '⚠',
          text: 'Connection Error',
          pulse: false
        };
      default:
        return {
          color: '#6c757d',
          bgColor: '#f8f9fa',
          icon: '?',
          text: 'Unknown',
          pulse: false
        };
    }
  };

  const statusConfig = getStatusConfig(status);

  if (compact) {
    return (
      <div className={`status-indicator-compact ${className}`}>
        <div 
          className={`status-dot ${statusConfig.pulse ? 'status-dot--pulse' : ''}`}
          style={{ backgroundColor: statusConfig.color }}
        />
        <span className="status-text">{statusConfig.text}</span>
        
        {totalImages > 0 && (
          <span className="image-count">
            {totalImages} images
          </span>
        )}

        <style>{`
          .status-indicator-compact {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
          }

          .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            position: relative;
          }

          .status-dot--pulse::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            border-radius: 50%;
            background-color: inherit;
            opacity: 0.3;
            animation: status-pulse 2s infinite;
          }

          .status-text {
            color: ${statusConfig.color};
            font-weight: 500;
          }

          .image-count {
            color: #6c757d;
            font-size: 12px;
          }

          @keyframes status-pulse {
            0% {
              transform: scale(1);
              opacity: 0.3;
            }
            50% {
              transform: scale(1.5);
              opacity: 0.1;
            }
            100% {
              transform: scale(1);
              opacity: 0.3;
            }
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className={`status-indicator ${className}`}>
      <div className="status-indicator__main">
        <div className="status-indicator__badge">
          <span 
            className={`status-icon ${statusConfig.pulse ? 'status-icon--pulse' : ''}`}
            style={{ color: statusConfig.color }}
          >
            {statusConfig.icon}
          </span>
          <span className="status-text" style={{ color: statusConfig.color }}>
            {statusConfig.text}
          </span>
        </div>

        <div className="status-indicator__info">
          {sessionId && (
            <div className="info-item">
              <span className="info-label">Session:</span>
              <span className="info-value" title={sessionId}>
                {sessionId.substring(0, 8)}...
              </span>
            </div>
          )}

          {streamUrl && (
            <div className="info-item">
              <span className="info-label">Stream:</span>
              <span className="info-value status-active">Active</span>
            </div>
          )}

          {totalImages > 0 && (
            <div className="info-item">
              <span className="info-label">Images:</span>
              <span className="info-value">{totalImages.toLocaleString()}</span>
            </div>
          )}
        </div>
      </div>

      {/* Connection quality indicator */}
      <div className="connection-quality">
        <div 
          className={`quality-bar ${status === 'connected' ? 'quality-bar--active' : ''}`}
        />
        <div 
          className={`quality-bar ${status === 'connected' ? 'quality-bar--active' : ''}`}
        />
        <div 
          className={`quality-bar ${status === 'connected' ? 'quality-bar--active' : ''}`}
        />
      </div>

      <style>{`
        .status-indicator {
          background: ${statusConfig.bgColor};
          border: 1px solid ${statusConfig.color}20;
          border-radius: 8px;
          padding: 12px 16px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          min-width: 250px;
        }

        .status-indicator__main {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .status-indicator__badge {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-icon {
          font-size: 16px;
          font-weight: bold;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 20px;
          height: 20px;
        }

        .status-icon--pulse {
          animation: status-icon-pulse 2s infinite;
        }

        .status-text {
          font-weight: 600;
          font-size: 14px;
        }

        .status-indicator__info {
          display: flex;
          gap: 16px;
          font-size: 12px;
        }

        .info-item {
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .info-label {
          color: #6c757d;
          font-weight: 500;
        }

        .info-value {
          color: #212529;
          font-weight: 600;
          font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }

        .status-active {
          color: #28a745 !important;
        }

        .connection-quality {
          display: flex;
          flex-direction: column;
          gap: 2px;
          align-items: flex-end;
        }

        .quality-bar {
          width: 12px;
          height: 3px;
          background-color: #e9ecef;
          border-radius: 1.5px;
          transition: background-color 0.3s;
        }

        .quality-bar:nth-child(1) {
          width: 8px;
        }

        .quality-bar:nth-child(2) {
          width: 10px;
        }

        .quality-bar--active {
          background-color: ${statusConfig.color};
        }

        @keyframes status-icon-pulse {
          0%, 100% {
            transform: scale(1);
            opacity: 1;
          }
          50% {
            transform: scale(1.1);
            opacity: 0.8;
          }
        }
      `}</style>
    </div>
  );
};

// Status badge component for simpler usage
interface StatusBadgeProps {
  status: ConnectionStatus;
  size?: 'small' | 'medium' | 'large';
  showText?: boolean;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = 'medium',
  showText = true,
  className = ''
}) => {
  const getStatusConfig = (status: ConnectionStatus) => {
    switch (status) {
      case 'connected':
        return { color: '#28a745', text: 'Online' };
      case 'connecting':
      case 'reconnecting':
        return { color: '#ffc107', text: 'Connecting' };
      case 'disconnected':
        return { color: '#6c757d', text: 'Offline' };
      case 'error':
        return { color: '#dc3545', text: 'Error' };
      default:
        return { color: '#6c757d', text: 'Unknown' };
    }
  };

  const statusConfig = getStatusConfig(status);
  const sizeConfig = {
    small: { dot: 6, font: 12, padding: '4px 8px' },
    medium: { dot: 8, font: 14, padding: '6px 12px' },
    large: { dot: 10, font: 16, padding: '8px 16px' }
  };

  const { dot, font, padding } = sizeConfig[size];

  return (
    <div className={`status-badge ${className}`}>
      <div 
        className="status-badge__dot"
        style={{ 
          backgroundColor: statusConfig.color,
          width: dot,
          height: dot
        }}
      />
      
      {showText && (
        <span className="status-badge__text">
          {statusConfig.text}
        </span>
      )}

      <style>{`
        .status-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: ${padding};
          background-color: ${statusConfig.color}15;
          border: 1px solid ${statusConfig.color}30;
          border-radius: 20px;
          font-size: ${font}px;
          font-weight: 500;
        }

        .status-badge__dot {
          border-radius: 50%;
          flex-shrink: 0;
        }

        .status-badge__text {
          color: ${statusConfig.color};
          white-space: nowrap;
        }
      `}</style>
    </div>
  );
};

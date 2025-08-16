import React, { useMemo } from 'react';
import { ImageData } from '../../types/image';
import { ConnectionStatus, ViewMode } from '../../types/api';
import { dataProcessor } from '../../services/dataProcessor';

interface DashboardProps {
  images: ImageData[];
  connectionStatus: ConnectionStatus;
  currentSessionId: string | null;
  streamUrl: string | null;
  onViewChange: (view: ViewMode) => void;
  className?: string;
}

export const Dashboard: React.FC<DashboardProps> = ({
  images,
  connectionStatus,
  currentSessionId,
  streamUrl,
  onViewChange,
  className = ''
}) => {
  // Calculate statistics
  const stats = useMemo(() => {
    return dataProcessor.calculateImageStats(images);
  }, [images]);

  // Recent activity (last 24 hours)
  const recentImages = useMemo(() => {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
    return images.filter(img => new Date(img.createdAt) > oneDayAgo);
  }, [images]);

  // Processing rate (images per minute in last hour)
  const processingRate = useMemo(() => {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
    const recentCount = images.filter(img => new Date(img.createdAt) > oneHourAgo).length;
    return Math.round(recentCount / 60 * 100) / 100; // Per minute
  }, [images]);

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    subtitle?: string;
    icon: string;
    color: string;
    onClick?: () => void;
  }> = ({ title, value, subtitle, icon, color, onClick }) => (
    <div 
      className={`stat-card ${onClick ? 'stat-card--clickable' : ''}`}
      onClick={onClick}
    >
      <div className="stat-card__icon" style={{ backgroundColor: color + '20', color }}>
        {icon}
      </div>
      <div className="stat-card__content">
        <div className="stat-card__value">{value}</div>
        <div className="stat-card__title">{title}</div>
        {subtitle && <div className="stat-card__subtitle">{subtitle}</div>}
      </div>
    </div>
  );

  return (
    <div className={`dashboard ${className}`}>
      <div className="dashboard__container">
        
        {/* Header Section */}
        <div className="dashboard__header">
          <h2 className="dashboard__title">Dashboard Overview</h2>
          <div className="dashboard__status">
            <div className="status-info">
              <span className="status-label">Status:</span>
              <span className={`status-value status-value--${connectionStatus}`}>
                {connectionStatus.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </span>
            </div>
            {currentSessionId && (
              <div className="session-info">
                <span className="session-label">Session:</span>
                <span className="session-id">{currentSessionId.substring(0, 8)}...</span>
              </div>
            )}
          </div>
        </div>

        {/* Stats Grid */}
        <div className="dashboard__stats">
          <StatCard
            title="Total Images"
            value={stats.total.toLocaleString()}
            subtitle={`${stats.recentCount} in last 24h`}
            icon="📷"
            color="#007bff"
            onClick={() => onViewChange('gallery')}
          />

          <StatCard
            title="With Location"
            value={`${stats.locationPercentage}%`}
            subtitle={`${stats.withLocation} images`}
            icon="📍"
            color="#28a745"
            onClick={() => onViewChange('map')}
          />

          <StatCard
            title="Objects Detected"
            value={stats.withObjects.toLocaleString()}
            subtitle={`${Math.round((stats.withObjects / Math.max(stats.total, 1)) * 100)}% of images`}
            icon="🎯"
            color="#ffc107"
          />

          <StatCard
            title="Avg Confidence"
            value={`${Math.round(stats.avgConfidence * 100)}%`}
            subtitle="Detection accuracy"
            icon="⭐"
            color="#17a2b8"
          />

          <StatCard
            title="Processing Rate"
            value={`${processingRate}/min`}
            subtitle="Current throughput"
            icon="⚡"
            color="#6f42c1"
          />

          <StatCard
            title="Active Sessions"
            value={stats.sessions}
            subtitle={currentSessionId ? 'Including current' : 'Historical data'}
            icon="📡"
            color="#fd7e14"
          />
        </div>

        {/* Charts Section */}
        <div className="dashboard__charts">
          
          {/* Status Breakdown */}
          <div className="chart-card">
            <h3 className="chart-title">Processing Status</h3>
            <div className="status-breakdown">
              {Object.entries(stats.byStatus).map(([status, count]) => (
                <div key={status} className="status-item">
                  <div className="status-bar">
                    <div 
                      className="status-fill"
                      style={{ 
                        width: `${(count / Math.max(stats.total, 1)) * 100}%`,
                        backgroundColor: getStatusColor(status)
                      }}
                    />
                  </div>
                  <div className="status-info">
                    <span className="status-name">{status}</span>
                    <span className="status-count">{count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="chart-card">
            <h3 className="chart-title">Recent Activity</h3>
            <div className="activity-list">
              {recentImages.slice(0, 5).map((image) => (
                <div key={image.id} className="activity-item">
                  <div className="activity-time">
                    {new Date(image.createdAt).toLocaleTimeString()}
                  </div>
                  <div className="activity-info">
                    <div className="activity-session">
                      Session: {image.sessionId.substring(0, 8)}...
                    </div>
                    {image.metadata.confidence && (
                      <div className="activity-confidence">
                        {Math.round(image.metadata.confidence * 100)}% confidence
                      </div>
                    )}
                  </div>
                  {image.location && (
                    <div className="activity-location">📍</div>
                  )}
                </div>
              ))}
              
              {recentImages.length === 0 && (
                <div className="no-activity">
                  No recent activity in the last 24 hours
                </div>
              )}
            </div>
          </div>

        </div>

        {/* Quick Actions */}
        <div className="dashboard__actions">
          <h3 className="actions-title">Quick Actions</h3>
          <div className="actions-grid">
            <button 
              className="action-button action-button--primary"
              onClick={() => onViewChange('map')}
              disabled={stats.withLocation === 0}
            >
              <span className="action-icon">🗺️</span>
              <span className="action-text">View on Map</span>
              <span className="action-count">{stats.withLocation}</span>
            </button>

            <button 
              className="action-button action-button--secondary"
              onClick={() => onViewChange('gallery')}
              disabled={stats.total === 0}
            >
              <span className="action-icon">🖼️</span>
              <span className="action-text">Browse Gallery</span>
              <span className="action-count">{stats.total}</span>
            </button>

            <button 
              className="action-button action-button--info"
              disabled={!streamUrl}
            >
              <span className="action-icon">📹</span>
              <span className="action-text">Live Stream</span>
              <span className="action-status">
                {streamUrl ? 'Active' : 'Inactive'}
              </span>
            </button>
          </div>
        </div>

      </div>

      <style>{`
        .dashboard {
          min-height: calc(100vh - 200px);
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          padding: 20px;
        }

        .dashboard__container {
          max-width: 1200px;
          margin: 0 auto;
          display: flex;
          flex-direction: column;
          gap: 24px;
        }

        .dashboard__header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: white;
          padding: 24px;
          border-radius: 12px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .dashboard__title {
          font-size: 28px;
          font-weight: 700;
          margin: 0;
          color: #2d3748;
        }

        .dashboard__status {
          display: flex;
          gap: 24px;
          font-size: 14px;
        }

        .status-info, .session-info {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-label, .session-label {
          color: #6c757d;
          font-weight: 500;
        }

        .status-value {
          font-weight: 600;
          text-transform: capitalize;
        }

        .status-value--connected { color: #28a745; }
        .status-value--connecting { color: #ffc107; }
        .status-value--reconnecting { color: #fd7e14; }
        .status-value--disconnected { color: #6c757d; }
        .status-value--error { color: #dc3545; }

        .session-id {
          font-family: 'Monaco', monospace;
          background: #f8f9fa;
          padding: 4px 8px;
          border-radius: 4px;
          color: #495057;
        }

        .dashboard__stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 20px;
        }

        .stat-card {
          background: white;
          padding: 24px;
          border-radius: 12px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          display: flex;
          align-items: center;
          gap: 16px;
          transition: all 0.2s;
        }

        .stat-card--clickable {
          cursor: pointer;
        }

        .stat-card--clickable:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }

        .stat-card__icon {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 20px;
          flex-shrink: 0;
        }

        .stat-card__content {
          flex: 1;
        }

        .stat-card__value {
          font-size: 24px;
          font-weight: 700;
          color: #2d3748;
          margin-bottom: 4px;
        }

        .stat-card__title {
          font-size: 14px;
          font-weight: 600;
          color: #4a5568;
          margin-bottom: 2px;
        }

        .stat-card__subtitle {
          font-size: 12px;
          color: #6c757d;
        }

        .dashboard__charts {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
        }

        .chart-card {
          background: white;
          padding: 24px;
          border-radius: 12px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .chart-title {
          font-size: 18px;
          font-weight: 600;
          margin: 0 0 20px 0;
          color: #2d3748;
        }

        .status-breakdown {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .status-item {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .status-bar {
          flex: 1;
          height: 8px;
          background: #f1f3f4;
          border-radius: 4px;
          overflow: hidden;
        }

        .status-fill {
          height: 100%;
          transition: width 0.3s;
        }

        .status-info {
          min-width: 100px;
          display: flex;
          justify-content: space-between;
          font-size: 14px;
        }

        .status-name {
          color: #4a5568;
          text-transform: capitalize;
        }

        .status-count {
          color: #2d3748;
          font-weight: 600;
        }

        .activity-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
          max-height: 200px;
          overflow-y: auto;
        }

        .activity-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px;
          background: #f8f9fa;
          border-radius: 8px;
          font-size: 13px;
        }

        .activity-time {
          font-family: 'Monaco', monospace;
          color: #6c757d;
          min-width: 70px;
        }

        .activity-info {
          flex: 1;
        }

        .activity-session {
          color: #495057;
          font-weight: 500;
        }

        .activity-confidence {
          color: #6c757d;
          font-size: 12px;
        }

        .activity-location {
          font-size: 16px;
        }

        .no-activity {
          text-align: center;
          color: #6c757d;
          font-style: italic;
          padding: 40px 20px;
        }

        .dashboard__actions {
          background: white;
          padding: 24px;
          border-radius: 12px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .actions-title {
          font-size: 18px;
          font-weight: 600;
          margin: 0 0 20px 0;
          color: #2d3748;
        }

        .actions-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
        }

        .action-button {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px 20px;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s;
          font-size: 14px;
          font-weight: 500;
          text-align: left;
        }

        .action-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .action-button--primary {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
        }

        .action-button--primary:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }

        .action-button--secondary {
          background: #6c757d;
          color: white;
        }

        .action-button--secondary:hover:not(:disabled) {
          background: #545b62;
          transform: translateY(-1px);
        }

        .action-button--info {
          background: #17a2b8;
          color: white;
        }

        .action-button--info:hover:not(:disabled) {
          background: #138496;
          transform: translateY(-1px);
        }

        .action-icon {
          font-size: 20px;
        }

        .action-text {
          flex: 1;
        }

        .action-count, .action-status {
          font-size: 12px;
          background: rgba(255, 255, 255, 0.2);
          padding: 2px 8px;
          border-radius: 12px;
        }

        @media (max-width: 768px) {
          .dashboard {
            padding: 16px;
          }

          .dashboard__header {
            flex-direction: column;
            gap: 16px;
            align-items: flex-start;
          }

          .dashboard__status {
            flex-direction: column;
            gap: 8px;
          }

          .dashboard__stats {
            grid-template-columns: 1fr;
          }

          .dashboard__charts {
            grid-template-columns: 1fr;
          }

          .stat-card {
            padding: 20px;
          }

          .chart-card {
            padding: 20px;
          }

          .actions-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

// Utility function to get status colors
function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    completed: '#28a745',
    processing: '#ffc107',
    error: '#dc3545',
    pending: '#6c757d',
    cancelled: '#fd7e14'
  };
  return colors[status] || '#6c757d';
}

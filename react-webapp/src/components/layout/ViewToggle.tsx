import React from 'react';
import { ViewMode } from '../../types/api';

interface ViewToggleProps {
  currentView: ViewMode;
  onViewChange: (view: ViewMode) => void;
  className?: string;
  size?: 'small' | 'medium' | 'large';
  variant?: 'buttons' | 'tabs' | 'pills';
}

export const ViewToggle: React.FC<ViewToggleProps> = ({
  currentView,
  onViewChange,
  className = '',
  size = 'medium',
  variant = 'buttons'
}) => {
  const views = [
    {
      key: 'dashboard' as ViewMode,
      label: 'Dashboard',
      icon: '📊',
      shortLabel: 'Stats'
    },
    {
      key: 'map' as ViewMode,
      label: 'Map',
      icon: '🗺️',
      shortLabel: 'Map'
    },
    {
      key: 'gallery' as ViewMode,
      label: 'Gallery',
      icon: '🖼️',
      shortLabel: 'Grid'
    }
  ];

  const sizeClasses = {
    small: 'view-toggle--small',
    medium: 'view-toggle--medium',
    large: 'view-toggle--large'
  };

  const variantClasses = {
    buttons: 'view-toggle--buttons',
    tabs: 'view-toggle--tabs',
    pills: 'view-toggle--pills'
  };

  return (
    <div className={`view-toggle ${sizeClasses[size]} ${variantClasses[variant]} ${className}`}>
      {views.map((view) => (
        <button
          key={view.key}
          className={`view-toggle__item ${currentView === view.key ? 'view-toggle__item--active' : ''}`}
          onClick={() => onViewChange(view.key)}
          title={view.label}
        >
          <span className="view-toggle__icon">{view.icon}</span>
          <span className="view-toggle__label">{view.shortLabel}</span>
        </button>
      ))}

      <style>{`
        .view-toggle {
          display: inline-flex;
          background-color: #f8f9fa;
          border-radius: 8px;
          padding: 4px;
          gap: 2px;
        }

        .view-toggle--buttons {
          border: 1px solid #dee2e6;
        }

        .view-toggle--tabs {
          background: none;
          border-bottom: 1px solid #dee2e6;
          border-radius: 0;
          padding: 0;
          gap: 0;
        }

        .view-toggle--pills {
          background: none;
          padding: 0;
          gap: 8px;
        }

        .view-toggle__item {
          display: flex;
          align-items: center;
          gap: 6px;
          background: none;
          border: none;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
          font-weight: 500;
        }

        /* Button variant styles */
        .view-toggle--buttons .view-toggle__item {
          padding: 8px 12px;
          border-radius: 6px;
          color: #6c757d;
        }

        .view-toggle--buttons .view-toggle__item:hover {
          background-color: #e9ecef;
          color: #495057;
        }

        .view-toggle--buttons .view-toggle__item--active {
          background-color: white;
          color: #007bff;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        /* Tabs variant styles */
        .view-toggle--tabs .view-toggle__item {
          padding: 12px 16px;
          color: #6c757d;
          border-bottom: 2px solid transparent;
        }

        .view-toggle--tabs .view-toggle__item:hover {
          color: #495057;
          border-bottom-color: #dee2e6;
        }

        .view-toggle--tabs .view-toggle__item--active {
          color: #007bff;
          border-bottom-color: #007bff;
        }

        /* Pills variant styles */
        .view-toggle--pills .view-toggle__item {
          padding: 6px 12px;
          border-radius: 20px;
          color: #6c757d;
          border: 1px solid transparent;
        }

        .view-toggle--pills .view-toggle__item:hover {
          background-color: #f8f9fa;
          color: #495057;
        }

        .view-toggle--pills .view-toggle__item--active {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-color: transparent;
        }

        /* Size variations */
        .view-toggle--small .view-toggle__item {
          font-size: 12px;
        }

        .view-toggle--small.view-toggle--buttons .view-toggle__item {
          padding: 6px 10px;
        }

        .view-toggle--small.view-toggle--tabs .view-toggle__item {
          padding: 10px 12px;
        }

        .view-toggle--small.view-toggle--pills .view-toggle__item {
          padding: 4px 10px;
        }

        .view-toggle--medium .view-toggle__item {
          font-size: 14px;
        }

        .view-toggle--large .view-toggle__item {
          font-size: 16px;
        }

        .view-toggle--large.view-toggle--buttons .view-toggle__item {
          padding: 10px 16px;
        }

        .view-toggle--large.view-toggle--tabs .view-toggle__item {
          padding: 16px 20px;
        }

        .view-toggle--large.view-toggle--pills .view-toggle__item {
          padding: 8px 16px;
        }

        .view-toggle__icon {
          font-size: 1.1em;
        }

        .view-toggle__label {
          font-weight: inherit;
        }

        @media (max-width: 480px) {
          .view-toggle__label {
            display: none;
          }

          .view-toggle__item {
            gap: 0;
          }

          .view-toggle__icon {
            font-size: 1.3em;
          }
        }
      `}</style>
    </div>
  );
};

// Simplified view switcher for mobile/compact layouts
interface ViewSwitcherProps {
  currentView: ViewMode;
  onViewChange: (view: ViewMode) => void;
  className?: string;
}

export const ViewSwitcher: React.FC<ViewSwitcherProps> = ({
  currentView,
  onViewChange,
  className = ''
}) => {
  const views = {
    dashboard: { label: 'Dashboard', icon: '📊' },
    map: { label: 'Map View', icon: '🗺️' },
    gallery: { label: 'Gallery', icon: '🖼️' }
  };

  const currentViewData = views[currentView];

  return (
    <div className={`view-switcher ${className}`}>
      <select 
        value={currentView}
        onChange={(e) => onViewChange(e.target.value as ViewMode)}
        className="view-switcher__select"
      >
        {Object.entries(views).map(([key, data]) => (
          <option key={key} value={key}>
            {data.icon} {data.label}
          </option>
        ))}
      </select>

      <div className="view-switcher__display">
        <span className="view-switcher__icon">{currentViewData.icon}</span>
        <span className="view-switcher__label">{currentViewData.label}</span>
        <span className="view-switcher__arrow">▼</span>
      </div>

      <style>{`
        .view-switcher {
          position: relative;
          display: inline-block;
        }

        .view-switcher__select {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          opacity: 0;
          cursor: pointer;
        }

        .view-switcher__display {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          background: white;
          border: 1px solid #dee2e6;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 500;
          color: #495057;
          cursor: pointer;
          transition: all 0.2s;
        }

        .view-switcher__display:hover {
          border-color: #007bff;
          box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.1);
        }

        .view-switcher__icon {
          font-size: 16px;
        }

        .view-switcher__arrow {
          margin-left: auto;
          font-size: 10px;
          color: #6c757d;
          transition: transform 0.2s;
        }

        .view-switcher:hover .view-switcher__arrow {
          transform: rotate(180deg);
        }
      `}</style>
    </div>
  );
};

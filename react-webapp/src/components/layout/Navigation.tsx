import React from 'react';
import { ViewMode } from '../../types/api';

interface NavigationProps {
  currentView: ViewMode;
  onViewChange: (view: ViewMode) => void;
  onRefresh: () => void;
  onClear: () => void;
  imageCount: number;
  className?: string;
}

export const Navigation: React.FC<NavigationProps> = ({
  currentView,
  onViewChange,
  onRefresh,
  onClear,
  imageCount,
  className = ''
}) => {
  const navItems = [
    {
      key: 'dashboard' as ViewMode,
      label: 'Dashboard',
      icon: '📊',
      description: 'Overview & Stats'
    },
    {
      key: 'map' as ViewMode,
      label: 'Map View',
      icon: '🗺️',
      description: 'Geographic View'
    },
    {
      key: 'gallery' as ViewMode,
      label: 'Gallery',
      icon: '🖼️',
      description: 'Image Gallery'
    }
  ];

  return (
    <nav className={`flex flex-col space-y-2 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Navigation</h2>
        <div className="flex space-x-2">
          <button
            onClick={onRefresh}
            className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
            title="Refresh data"
          >
            🔄 Refresh
          </button>
          <button
            onClick={onClear}
            className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
            title="Clear all data"
          >
            🗑️ Clear
          </button>
        </div>
      </div>

      <div className="mb-4 p-3 bg-gray-100 rounded-lg">
        <p className="text-sm text-gray-600">
          Images: <span className="font-semibold text-gray-800">{imageCount}</span>
        </p>
      </div>

      <div className="space-y-1">
        {navItems.map((item) => (
          <button
            key={item.key}
            onClick={() => onViewChange(item.key)}
            className={`w-full text-left p-3 rounded-lg transition-colors ${
              currentView === item.key
                ? 'bg-blue-500 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
            }`}
          >
            <div className="flex items-center space-x-3">
              <span className="text-lg">{item.icon}</span>
              <div>
                <div className="font-medium">{item.label}</div>
                <div className={`text-xs ${
                  currentView === item.key ? 'text-blue-100' : 'text-gray-500'
                }`}>
                  {item.description}
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </nav>
  );
};

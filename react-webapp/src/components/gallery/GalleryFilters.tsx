import React, { useState, useCallback } from 'react';
import { ImageFilter, SortOption, DateRange } from '../../types/image';
import { formatDate } from '../../utils/formatters';

interface GalleryFiltersProps {
  filter: ImageFilter;
  sortBy: SortOption;
  onFilterChange: (filter: ImageFilter) => void;
  onSortChange: (sort: SortOption) => void;
  availableTags: string[];
  availableSessions: string[];
  className?: string;
}

export const GalleryFilters: React.FC<GalleryFiltersProps> = ({
  filter,
  sortBy,
  onFilterChange,
  onSortChange,
  availableTags,
  availableSessions,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [dateRangeInput, setDateRangeInput] = useState({
    start: filter.dateRange?.start ? formatDate(filter.dateRange.start, 'short') : '',
    end: filter.dateRange?.end ? formatDate(filter.dateRange.end, 'short') : ''
  });

  // Handle filter changes
  const handleSessionChange = useCallback((sessionId: string) => {
    onFilterChange({
      ...filter,
      sessionId: sessionId === 'all' ? null : sessionId
    });
  }, [filter, onFilterChange]);

  const handleLocationFilterChange = useCallback((hasLocation: boolean | null) => {
    onFilterChange({
      ...filter,
      hasLocation
    });
  }, [filter, onFilterChange]);

  const handleConfidenceChange = useCallback((confidence: string) => {
    onFilterChange({
      ...filter,
      confidence: confidence === '' ? null : parseFloat(confidence)
    });
  }, [filter, onFilterChange]);

  const handleTagToggle = useCallback((tag: string) => {
    const newTags = filter.tags?.includes(tag)
      ? filter.tags.filter(t => t !== tag)
      : [...(filter.tags || []), tag];
    
    onFilterChange({
      ...filter,
      tags: newTags
    });
  }, [filter, onFilterChange]);

  const handleDateRangeChange = useCallback((field: 'start' | 'end', value: string) => {
    const newInput = { ...dateRangeInput, [field]: value };
    setDateRangeInput(newInput);

    // Parse dates and update filter
    let newDateRange: DateRange | null = null;
    
    if (newInput.start || newInput.end) {
      const startDate = newInput.start ? new Date(newInput.start) : new Date(0);
      const endDate = newInput.end ? new Date(newInput.end) : new Date();
      
      if (!isNaN(startDate.getTime()) && !isNaN(endDate.getTime())) {
        newDateRange = { start: startDate, end: endDate };
      }
    }

    onFilterChange({
      ...filter,
      dateRange: newDateRange
    });
  }, [filter, onFilterChange, dateRangeInput]);

  const handleClearFilters = useCallback(() => {
    onFilterChange({
      sessionId: null,
      dateRange: null,
      hasLocation: null,
      tags: [],
      confidence: null,
      categories: [],
      objects: []
    });
    setDateRangeInput({ start: '', end: '' });
  }, [onFilterChange]);

  const hasActiveFilters = !!(
    filter.sessionId ||
    filter.dateRange ||
    filter.hasLocation !== null ||
    (filter.tags && filter.tags.length > 0) ||
    filter.confidence !== null
  );

  return (
    <div className={`gallery-filters ${className}`}>
      <div className="filters-header">
        <div className="filters-main">
          {/* Sort dropdown */}
          <div className="filter-group">
            <label className="filter-label">Sort by:</label>
            <select
              className="filter-select"
              value={sortBy}
              onChange={(e) => onSortChange(e.target.value as SortOption)}
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="confidence">Highest confidence</option>
              <option value="location">With location</option>
              <option value="name">Name</option>
              <option value="size">File size</option>
              <option value="processing-time">Processing time</option>
            </select>
          </div>

          {/* Session filter */}
          {availableSessions.length > 1 && (
            <div className="filter-group">
              <label className="filter-label">Session:</label>
              <select
                className="filter-select"
                value={filter.sessionId || 'all'}
                onChange={(e) => handleSessionChange(e.target.value)}
              >
                <option value="all">All sessions</option>
                {availableSessions.map(sessionId => (
                  <option key={sessionId} value={sessionId}>
                    {sessionId.slice(-8)}...
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Location filter */}
          <div className="filter-group">
            <label className="filter-label">Location:</label>
            <select
              className="filter-select"
              value={filter.hasLocation === null ? 'all' : filter.hasLocation.toString()}
              onChange={(e) => {
                const value = e.target.value;
                handleLocationFilterChange(
                  value === 'all' ? null : value === 'true'
                );
              }}
            >
              <option value="all">All images</option>
              <option value="true">With location</option>
              <option value="false">Without location</option>
            </select>
          </div>

          {/* Advanced filters toggle */}
          <button
            className={`filter-toggle ${isExpanded ? 'filter-toggle--active' : ''}`}
            onClick={() => setIsExpanded(!isExpanded)}
            title="Show advanced filters"
          >
            = Filters
            {hasActiveFilters && <span className="active-indicator">�</span>}
          </button>
        </div>

        {/* Clear filters */}
        {hasActiveFilters && (
          <button
            className="filter-clear"
            onClick={handleClearFilters}
            title="Clear all filters"
          >
             Clear
          </button>
        )}
      </div>

      {/* Advanced filters */}
      {isExpanded && (
        <div className="filters-advanced">
          <div className="filters-row">
            {/* Confidence filter */}
            <div className="filter-group">
              <label className="filter-label">Min confidence:</label>
              <input
                type="range"
                className="filter-range"
                min="0"
                max="1"
                step="0.1"
                value={filter.confidence || 0}
                onChange={(e) => handleConfidenceChange(e.target.value)}
              />
              <span className="range-value">
                {filter.confidence ? Math.round(filter.confidence * 100) : 0}%
              </span>
            </div>

            {/* Date range filter */}
            <div className="filter-group">
              <label className="filter-label">Date range:</label>
              <div className="date-inputs">
                <input
                  type="date"
                  className="filter-input filter-input--date"
                  value={dateRangeInput.start}
                  onChange={(e) => handleDateRangeChange('start', e.target.value)}
                  placeholder="Start date"
                />
                <span className="date-separator">to</span>
                <input
                  type="date"
                  className="filter-input filter-input--date"
                  value={dateRangeInput.end}
                  onChange={(e) => handleDateRangeChange('end', e.target.value)}
                  placeholder="End date"
                />
              </div>
            </div>
          </div>

          {/* Tags filter */}
          {availableTags.length > 0 && (
            <div className="filter-group filter-group--full">
              <label className="filter-label">Tags:</label>
              <div className="tags-container">
                {availableTags.slice(0, 10).map(tag => (
                  <button
                    key={tag}
                    className={`tag-button ${filter.tags?.includes(tag) ? 'tag-button--active' : ''}`}
                    onClick={() => handleTagToggle(tag)}
                  >
                    {tag}
                  </button>
                ))}
                {availableTags.length > 10 && (
                  <span className="tags-more">
                    +{availableTags.length - 10} more
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Quick filters */}
          <div className="filter-group filter-group--full">
            <label className="filter-label">Quick filters:</label>
            <div className="quick-filters">
              <button 
                className={`quick-filter ${filter.hasLocation === true ? 'quick-filter--active' : ''}`}
                onClick={() => handleLocationFilterChange(filter.hasLocation === true ? null : true)}
              >
                =� With location
              </button>
              <button 
                className={`quick-filter ${(filter.confidence && filter.confidence >= 0.8) ? 'quick-filter--active' : ''}`}
                onClick={() => onFilterChange({
                  ...filter,
                  confidence: (filter.confidence && filter.confidence >= 0.8) ? null : 0.8
                })}
              >
                P High confidence
              </button>
              <button 
                className="quick-filter"
                onClick={() => {
                  const today = new Date();
                  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
                  onFilterChange({
                    ...filter,
                    dateRange: { start: yesterday, end: today }
                  });
                  setDateRangeInput({
                    start: formatDate(yesterday, 'short'),
                    end: formatDate(today, 'short')
                  });
                }}
              >
                =� Last 24h
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .gallery-filters {
          background: white;
          border-bottom: 1px solid #e9ecef;
          padding: 16px 20px;
        }

        .filters-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          flex-wrap: wrap;
        }

        .filters-main {
          display: flex;
          align-items: center;
          gap: 20px;
          flex-wrap: wrap;
        }

        .filter-group {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .filter-group--full {
          width: 100%;
          flex-direction: column;
          align-items: flex-start;
          gap: 8px;
        }

        .filter-label {
          font-size: 14px;
          font-weight: 500;
          color: #495057;
          white-space: nowrap;
        }

        .filter-select {
          padding: 6px 12px;
          border: 1px solid #dee2e6;
          border-radius: 6px;
          font-size: 14px;
          background: white;
          cursor: pointer;
          min-width: 120px;
        }

        .filter-select:focus {
          outline: none;
          border-color: #007bff;
          box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }

        .filter-toggle {
          padding: 8px 16px;
          border: 1px solid #dee2e6;
          border-radius: 20px;
          background: white;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
          position: relative;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .filter-toggle:hover {
          border-color: #007bff;
          color: #007bff;
        }

        .filter-toggle--active {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }

        .active-indicator {
          color: #28a745;
          font-size: 8px;
          position: absolute;
          top: 2px;
          right: 2px;
        }

        .filter-toggle--active .active-indicator {
          color: #90ff90;
        }

        .filter-clear {
          padding: 6px 12px;
          border: 1px solid #dc3545;
          border-radius: 6px;
          background: white;
          color: #dc3545;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .filter-clear:hover {
          background: #dc3545;
          color: white;
        }

        .filters-advanced {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid #e9ecef;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .filters-row {
          display: flex;
          gap: 24px;
          flex-wrap: wrap;
        }

        .filter-range {
          width: 120px;
          margin: 0 8px;
        }

        .range-value {
          font-size: 13px;
          color: #6c757d;
          min-width: 35px;
        }

        .date-inputs {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .filter-input {
          padding: 6px 10px;
          border: 1px solid #dee2e6;
          border-radius: 4px;
          font-size: 13px;
        }

        .filter-input--date {
          width: 140px;
        }

        .filter-input:focus {
          outline: none;
          border-color: #007bff;
          box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }

        .date-separator {
          font-size: 13px;
          color: #6c757d;
        }

        .tags-container {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          align-items: center;
        }

        .tag-button {
          padding: 4px 12px;
          border: 1px solid #dee2e6;
          border-radius: 16px;
          background: white;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .tag-button:hover {
          border-color: #007bff;
          color: #007bff;
        }

        .tag-button--active {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }

        .tags-more {
          font-size: 13px;
          color: #6c757d;
          font-style: italic;
        }

        .quick-filters {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .quick-filter {
          padding: 6px 12px;
          border: 1px solid #dee2e6;
          border-radius: 16px;
          background: white;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .quick-filter:hover {
          border-color: #007bff;
          color: #007bff;
        }

        .quick-filter--active {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }

        /* Responsive design */
        @media (max-width: 768px) {
          .gallery-filters {
            padding: 12px 16px;
          }

          .filters-header {
            flex-direction: column;
            align-items: stretch;
            gap: 12px;
          }

          .filters-main {
            flex-direction: column;
            gap: 12px;
          }

          .filter-group {
            justify-content: space-between;
            width: 100%;
          }

          .filter-select {
            min-width: 0;
            flex: 1;
          }

          .filters-row {
            flex-direction: column;
            gap: 12px;
          }

          .date-inputs {
            flex-direction: column;
            align-items: stretch;
            gap: 8px;
          }

          .filter-input--date {
            width: 100%;
          }

          .filter-range {
            width: 100%;
            margin: 0;
          }
        }
      `}</style>
    </div>
  );
};
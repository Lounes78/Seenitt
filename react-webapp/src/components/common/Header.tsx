import React, { ReactNode } from 'react';

interface HeaderProps {
  children?: ReactNode;
  title?: string;
  className?: string;
  sticky?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  children,
  title = 'SeenittApp',
  className = '',
  sticky = true
}) => {
  return (
    <header className={`header ${sticky ? 'header--sticky' : ''} ${className}`}>
      <div className="header__container">
        <div className="header__brand">
          <div className="header__logo">
            <div className="logo-icon">
              📷
            </div>
            <h1 className="logo-text">{title}</h1>
          </div>
          
          <div className="header__subtitle">
            Real-time Image Processing & Mapping
          </div>
        </div>

        <div className="header__content">
          {children}
        </div>
      </div>

      <style>{`
        .header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          z-index: 1000;
        }

        .header--sticky {
          position: sticky;
          top: 0;
        }

        .header__container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 16px 20px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          min-height: 70px;
        }

        .header__brand {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .header__logo {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .logo-icon {
          font-size: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          width: 44px;
          height: 44px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          backdrop-filter: blur(10px);
        }

        .logo-text {
          font-size: 24px;
          font-weight: 700;
          margin: 0;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }

        .header__subtitle {
          font-size: 13px;
          opacity: 0.9;
          font-weight: 400;
          margin-left: 56px;
        }

        .header__content {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        @media (max-width: 768px) {
          .header__container {
            flex-direction: column;
            align-items: flex-start;
            gap: 12px;
            padding: 16px;
          }

          .header__content {
            width: 100%;
            justify-content: space-between;
          }

          .header__subtitle {
            margin-left: 0;
            font-size: 12px;
          }

          .logo-text {
            font-size: 20px;
          }

          .logo-icon {
            font-size: 24px;
            width: 36px;
            height: 36px;
          }
        }
      `}</style>
    </header>
  );
};

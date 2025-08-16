import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    hmr: { overlay: false }
  },
  define: {
    'process.env': {
      NODE_ENV: JSON.stringify(process.env.NODE_ENV || 'development'),
      REACT_APP_MAPBOX_TOKEN: JSON.stringify(process.env.REACT_APP_MAPBOX_TOKEN || 'pk.demo_token'),
      REACT_APP_API_BASE_URL: JSON.stringify(process.env.REACT_APP_API_BASE_URL || '/api'),
      REACT_APP_BACKEND_URL: JSON.stringify(process.env.REACT_APP_BACKEND_URL || '')
    }
  }
})
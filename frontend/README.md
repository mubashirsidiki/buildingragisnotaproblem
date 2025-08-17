# Frontend - BuildingRAGisNotAProblem

A modern Next.js 14 frontend application for the RAG system, featuring a beautiful UI for PDF uploads, vector search, and AI chat.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
# or
yarn install
# or
pnpm install
```

### 2. Environment Configuration

**The project uses a single `.env` file for all environment variables:**

```env
# Backend API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# Feature Flags
NEXT_PUBLIC_ENABLE_RAG=true
NEXT_PUBLIC_ENABLE_CHAT=true
NEXT_PUBLIC_ENABLE_UPLOAD=true

# Development Settings
NEXT_PUBLIC_DEBUG_MODE=true
NEXT_PUBLIC_LOG_LEVEL=debug
```

**To customize for your environment, edit the `.env` file directly.**

### 3. Start Development Server
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🌍 Environment Variables

### Required Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_BASE_URL` | Backend API base URL | `http://localhost:8000/api` |
| `NEXT_PUBLIC_BACKEND_URL` | Backend server URL | `http://localhost:8000` |

### Feature Flags

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_ENABLE_RAG` | Enable RAG functionality | `true` |
| `NEXT_PUBLIC_ENABLE_CHAT` | Enable chat functionality | `true` |
| `NEXT_PUBLIC_ENABLE_UPLOAD` | Enable PDF upload functionality | `true` |

### Development Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_DEBUG_MODE` | Enable debug logging | `false` |
| `NEXT_PUBLIC_LOG_LEVEL` | Log level (debug, info, warn, error) | `info` |

## 🏗️ Architecture

### Configuration Management
- **`lib/config.ts`**: Centralized configuration with type safety
- **Environment-based settings**: Different configs for dev/staging/prod
- **Feature flags**: Enable/disable functionality per environment

### API Layer
- **`lib/api.ts`**: Type-safe API service layer
- **Centralized error handling**: Consistent error handling across all API calls
- **Request/response logging**: Debug logging for development
- **Type definitions**: Full TypeScript interfaces for all API responses

### Component Structure
- **App Router**: Next.js 14 App Router for modern routing
- **UI Components**: Radix UI components with Tailwind CSS styling
- **Responsive Design**: Mobile-first responsive design
- **Theme Support**: Dark/light mode with system preference detection

## 📱 Pages & Features

### Landing Page (`/`)
- Hero section with clear value proposition
- Feature showcase
- Call-to-action buttons

### PDF Upload (`/upload`)
- Drag & drop file upload
- Advanced chunking configuration
- Processing progress tracking
- Results display with statistics

### Dashboard (`/dashboard`)
- **Vector Search Tab**: Semantic search with configurable parameters
- **AI Chat Tab**: RAG-enhanced chat with source citations
- **Configuration Panels**: Search and chat settings
- **Real-time Results**: Live search and chat responses

## 🔧 Development

### Code Quality
- **TypeScript**: Full type safety
- **ESLint**: Code linting (configured to ignore during builds)
- **Prettier**: Code formatting

### Build & Deployment
```bash
# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

### Environment-Specific Builds
The application automatically adapts to different environments:

- **Development**: Uses local backend URLs, debug logging enabled
- **Production**: Uses production backend URLs, minimal logging
- **Staging**: Configurable via environment variables

## 🔌 API Integration

### Backend Communication
- **RESTful API**: Standard HTTP methods
- **Error Handling**: Graceful error handling with user feedback
- **Loading States**: Loading indicators for all async operations
- **Retry Logic**: Automatic retry for failed requests

### API Endpoints Used
- `POST /v1/pdf-rag/upload` - PDF processing
- `POST /v1/pdf-rag/search` - Vector search
- `POST /v1/chat/simple` - Simple chat
- `POST /v1/chat/rag` - RAG-enhanced chat

## 🎨 Styling & UI

### Design System
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible component primitives
- **Custom Components**: Reusable UI components
- **Responsive Grid**: Mobile-first responsive layout

### Theme Support
- **Dark/Light Mode**: Automatic theme switching
- **System Preference**: Respects user's system theme
- **Custom Colors**: Emerald accent color scheme
- **Consistent Spacing**: Standardized spacing scale

## 🚀 Performance

### Optimization Features
- **Next.js 14**: Latest performance optimizations
- **Image Optimization**: Automatic image optimization
- **Code Splitting**: Automatic route-based code splitting
- **Static Generation**: Static generation where possible

### Monitoring
- **Debug Logging**: Development-time API logging
- **Error Tracking**: Comprehensive error handling
- **Performance Metrics**: Built-in Next.js performance monitoring

## 🔒 Security

### Security Headers
- **X-Frame-Options**: Prevents clickjacking
- **X-Content-Type-Options**: Prevents MIME type sniffing
- **Referrer-Policy**: Controls referrer information

### API Security
- **CORS Configuration**: Proper CORS handling
- **Input Validation**: Client-side validation
- **Error Sanitization**: Safe error messages

## 📚 Dependencies

### Core Dependencies
- **Next.js 14**: React framework
- **React 18**: UI library
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling

### UI Components
- **Radix UI**: Accessible components
- **Lucide React**: Icons
- **React Hook Form**: Form handling
- **React Dropzone**: File uploads

### Development Tools
- **ESLint**: Code linting
- **PostCSS**: CSS processing
- **Autoprefixer**: CSS compatibility

## 🐛 Troubleshooting

### Common Issues

**API Connection Errors:**
- Verify backend is running on the configured URL
- Check CORS configuration in backend
- Ensure environment variables are set correctly

**Build Errors:**
- Clear `.next` folder and node_modules
- Verify TypeScript configuration
- Check for missing dependencies

**Environment Issues:**
- Ensure `.env` file exists and is configured
- Restart development server after environment changes
- Check browser console for configuration errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

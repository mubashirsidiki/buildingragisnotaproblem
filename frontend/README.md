# Frontend - Next.js RAG Interface

Modern web interface for the RAG system built with Next.js 14, TypeScript, and Tailwind CSS.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- pnpm (recommended) or npm

### Installation
```bash
cd frontend
pnpm install
```

### Development
```bash
pnpm dev
```
Access at: http://localhost:3000

### Build & Production
```bash
pnpm build
pnpm start
```

## 🏗️ Project Structure

```
frontend/
├── app/                    # Next.js 14 app directory
│   ├── upload/            # PDF upload interface
│   ├── workspace/         # Search and chat workspace
│   ├── analytics/         # Usage analytics dashboard
│   └── docs/              # Documentation pages
├── components/            # Reusable UI components
│   ├── ui/               # Shadcn/ui components
│   └── navigation.tsx    # Main navigation
├── lib/                   # Utilities and API client
└── styles/                # Global CSS and Tailwind
```

## 🔧 Key Features

- **PDF Upload**: Drag & drop interface with chunking options
- **Search Interface**: Vector search with filters and reranking
- **Chat Interface**: RAG-powered conversations
- **Responsive Design**: Mobile-first approach
- **Dark/Light Mode**: Theme switching support

## 📱 Pages

- **`/`** - Landing page
- **`/upload`** - PDF upload and processing
- **`/workspace`** - Search and chat interface
- **`/analytics`** - Usage statistics and monitoring

## 🎨 UI Components

Built with:
- **Shadcn/ui** - Modern component library
- **Tailwind CSS** - Utility-first styling
- **Radix UI** - Accessible primitives
- **Lucide React** - Icon library

## 🔌 API Integration

The frontend communicates with the AI backend through:
- **REST APIs** for PDF operations
- **WebSocket** for real-time chat
- **GraphQL** for analytics (planned)

## 🌐 Environment Variables

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_ENABLE_RAG=true
NEXT_PUBLIC_ENABLE_CHAT=true
NEXT_PUBLIC_ENABLE_UPLOAD=true
```

## 🐳 Docker

```bash
# Build image
docker build -t rag-frontend .

# Run container
docker run -p 3000:3000 rag-frontend
```

## 📊 Development Tools

- **TypeScript** - Type safety
- **ESLint** - Code quality
- **Prettier** - Code formatting
- **Tailwind CSS** - Utility-first CSS

## 🚀 Deployment

The frontend is optimized for:
- **Vercel** (recommended)
- **Netlify**
- **Docker containers**
- **Static hosting**

## 🔍 Troubleshooting

**Common issues:**
- Port 3000 already in use → Change port in `next.config.mjs`
- API connection errors → Check backend URL in environment
- Build failures → Clear `.next` folder and reinstall dependencies

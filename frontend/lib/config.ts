/**
 * Frontend Configuration Utility
 * Centralizes all environment variables and configuration settings
 */

export const config = {
  // API Configuration
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api',
    backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
    endpoints: {
      pdfRag: {
        upload: '/v1/pdf-rag/upload',
        search: '/v1/pdf-rag/search',
        clear: '/v1/pdf-rag/clear',
        health: '/v1/pdf-rag/health',
      },
      chat: {
        simple: '/v1/chat/simple',
        rag: '/v1/chat/rag',
        full: '/v1/chat/',
      },
    },
  },

  // Feature Flags
  features: {
    rag: process.env.NEXT_PUBLIC_ENABLE_RAG === 'true',
    chat: process.env.NEXT_PUBLIC_ENABLE_CHAT === 'true',
    upload: process.env.NEXT_PUBLIC_ENABLE_UPLOAD === 'true',
  },

  // Development Settings
  debug: {
    enabled: process.env.NEXT_PUBLIC_DEBUG_MODE === 'true',
    logLevel: process.env.NEXT_PUBLIC_LOG_LEVEL || 'info',
  },

  // App Configuration
  app: {
    name: process.env.NEXT_PUBLIC_APP_NAME || 'BuildingRAGisNotAProblem',
    version: process.env.NEXT_PUBLIC_APP_VERSION || '1.0.0',
    description: 'Effortlessly enhance your applications with intelligent data retrieval and generation',
  },
} as const

/**
 * Helper function to build full API URLs
 */
export const buildApiUrl = (endpoint: string): string => {
  return `${config.api.baseUrl}${endpoint}`
}

/**
 * Helper function to build backend URLs
 */
export const buildBackendUrl = (endpoint: string): string => {
  return `${config.api.backendUrl}${endpoint}`
}

/**
 * Type-safe config access
 */
export type Config = typeof config
export type ApiEndpoints = typeof config.api.endpoints

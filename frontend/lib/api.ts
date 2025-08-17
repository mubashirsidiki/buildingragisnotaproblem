/**
 * API Service Layer
 * Provides type-safe API calls using the configuration
 */

import { config, buildApiUrl } from './config'

// Types for API responses
export interface ApiResponse<T = any> {
  data?: T
  error?: string
  message?: string
}

export interface UploadResponse {
  message: string
  filename: string
  chunks_created: number
  chunking_mode: string
  pdf_stats: {
    total_pages: number
    total_characters: number
    total_words: number
    total_sentences: number
    total_paragraphs: number
  }
}

export interface SearchResult {
  rank: number
  cosine_similarity_score: number
  cross_encoder_score: number
  id: string
  filename: string
  chunk_text: string
  chunk_index: string
  page_number: number
  created_at: string
  start_pos: number
  end_pos: number
  start_line: number
  end_line: number
  sentence_count: number
}

export interface SearchResponse {
  chunks: SearchResult[]
  total_results: number
  distance_metric: string
}

export interface ChatRequest {
  message: string
  user_id: string
  conversation_history?: Array<{
    role: 'user' | 'assistant'
    content: string
  }>
}

export interface ChatResponse {
  response: string
  user_id: string
}

export interface RagChatResponse extends ChatResponse {
  used_rag: boolean
  rag_sources?: SearchResult[]
}

// API service class
export class ApiService {
  private baseUrl: string

  constructor() {
    this.baseUrl = config.api.baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = buildApiUrl(endpoint)
    
    if (config.debug.enabled) {
      console.log(`🌐 API Request: ${options.method || 'GET'} ${url}`)
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      if (config.debug.enabled) {
        console.log(`✅ API Response:`, data)
      }

      return { data }
    } catch (error) {
      if (config.debug.enabled) {
        console.error(`❌ API Error:`, error)
      }
      
      return {
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      }
    }
  }

  // PDF RAG API methods
  async uploadPdf(formData: FormData): Promise<ApiResponse<UploadResponse>> {
    const url = buildApiUrl(config.api.endpoints.pdfRag.upload)
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return { data }
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Upload failed',
      }
    }
  }

  async searchChunks(query: string, params: {
    limit?: number
    min_cosine_similarity?: number
    min_cross_score?: number
    expand_query?: boolean
    rerank?: boolean
  } = {}): Promise<ApiResponse<SearchResponse>> {
    // The backend expects a POST request with query parameters in the URL
    const queryParams = new URLSearchParams({
      query,
      limit: (params.limit || 5).toString(),
      min_cosine_similarity: (params.min_cosine_similarity || 0.5).toString(),
      min_cross_score: (params.min_cross_score || 0.0).toString(),
      expand_query: (params.expand_query !== undefined ? params.expand_query : true).toString(),
      rerank: (params.rerank !== undefined ? params.rerank : true).toString(),
    })
    
    const url = `${buildApiUrl(config.api.endpoints.pdfRag.search)}?${queryParams}`
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // No body needed since parameters are in URL
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return { data }
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Search failed',
      }
    }
  }

  async clearDatabase(): Promise<ApiResponse<{ message: string }>> {
    return apiService.request<{ message: string }>(config.api.endpoints.pdfRag.clear, {
      method: 'DELETE',
    })
  }

  async healthCheck(): Promise<ApiResponse<any>> {
    return apiService.request(config.api.endpoints.pdfRag.health, {
      method: 'GET',
    })
  }

  // Chat API methods
  async simpleChat(request: ChatRequest): Promise<ApiResponse<ChatResponse>> {
    return apiService.request<ChatResponse>(config.api.endpoints.chat.simple, {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async ragChat(request: ChatRequest, params: {
    limit?: number
    min_cosine_similarity?: number
    min_cross_score?: number
    expand_query?: boolean
    rerank?: boolean
  } = {}): Promise<ApiResponse<RagChatResponse>> {
    return apiService.request<RagChatResponse>(config.api.endpoints.chat.rag, {
      method: 'POST',
      body: JSON.stringify({ ...request, ...params }),
    })
  }

  async fullChat(request: ChatRequest): Promise<ApiResponse<ChatResponse>> {
    return apiService.request<ChatResponse>(config.api.endpoints.chat.full, {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  // Analytics API methods
  async getUserAnalytics(userId: string): Promise<ApiResponse<any>> {
    return apiService.request(`/v1/analytics/usage/${userId}`, {
      method: 'GET',
    })
  }

  async getPricingConfig(): Promise<ApiResponse<any>> {
    return apiService.request('/v1/analytics/pricing', {
      method: 'GET',
    })
  }
}

// Export singleton instance
export const apiService = new ApiService()

// Export individual methods for convenience
export const {
  uploadPdf,
  searchChunks,
  clearDatabase,
  healthCheck,
  simpleChat,
  ragChat,
  fullChat,
  getUserAnalytics,
  getPricingConfig,
} = apiService

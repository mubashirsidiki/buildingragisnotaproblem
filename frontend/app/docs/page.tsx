"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Copy,
  Upload,
  Search,
  Trash2,
  MessageSquare,
  Bot,
  Code,
  Play,
} from "lucide-react"
import Link from "next/link"

interface ApiEndpoint {
  id: string
  title: string
  method: "GET" | "POST" | "DELETE"
  endpoint: string
  description: string
  requestSchema: any
  responseSchema: any
  example: {
    request: any
    response: any
  }
  icon: React.ReactNode
}

const apiEndpoints: ApiEndpoint[] = [
  {
    id: "pdf-upload",
    title: "PDF Upload & Processing",
    method: "POST",
    endpoint: "/v1/pdf-rag/upload",
    description:
      "Upload and process PDFs into semantic chunks, generate embeddings, and store them in a vector database.",
    icon: <Upload className="h-5 w-5" />,
    requestSchema: {
      type: "multipart/form-data",
      properties: {
        file: { type: "file", description: "PDF file to upload" },
        chunking_mode: { type: "string", enum: ["semantic", "fixed", "sentence"], default: "semantic" },
        max_chunk_size: { type: "integer", default: 1000 },
        breakpoint_threshold_type: { type: "string", enum: ["percentile", "standard_deviation", "interquartile"] },
        breakpoint_threshold_amount: { type: "number", default: 0.95 },
        chunk_overlap: { type: "integer", default: 200 },
        sentence_group_size: { type: "integer", default: 3 },
      },
    },
    responseSchema: {
      type: "object",
      properties: {
        message: { type: "string" },
        filename: { type: "string" },
        chunks_created: { type: "integer" },
        processing_time: { type: "number" },
        pdf_stats: {
          type: "object",
          properties: {
            total_pages: { type: "integer" },
            total_characters: { type: "integer" },
            total_words: { type: "integer" },
            total_sentences: { type: "integer" },
            total_paragraphs: { type: "integer" },
          },
        },
      },
    },
    example: {
      request: {
        file: "document.pdf",
        chunking_mode: "semantic",
        max_chunk_size: 1000,
        breakpoint_threshold_type: "percentile",
        breakpoint_threshold_amount: 0.95,
        chunk_overlap: 200,
        sentence_group_size: 3,
      },
      response: {
        message: "PDF processed successfully",
        filename: "document.pdf",
        chunks_created: 45,
        processing_time: 12.34,
        pdf_stats: {
          total_pages: 10,
          total_characters: 25000,
          total_words: 4500,
          total_sentences: 200,
          total_paragraphs: 50,
        },
      },
    },
  },
  {
    id: "vector-search",
    title: "Vector Search",
    method: "POST",
    endpoint: "/v1/pdf-rag/search",
    description: "Perform vector similarity search with optional query expansion and reranking.",
    icon: <Search className="h-5 w-5" />,
    requestSchema: {
      type: "application/json",
      properties: {
        query: { type: "string", description: "Search query text" },
        limit: { type: "integer", default: 10, description: "Maximum number of results" },
        min_cosine_similarity: { type: "number", default: 0.5, description: "Minimum similarity threshold" },
        min_cross_score: { type: "number", default: 0.0, description: "Minimum cross-encoder score" },
        expand_query: { type: "boolean", default: true, description: "Enable query expansion" },
        rerank: { type: "boolean", default: true, description: "Enable result reranking" },
      },
    },
    responseSchema: {
      type: "object",
      properties: {
        distance_metric: { type: "string" },
        total_results: { type: "integer" },
        chunks: {
          type: "array",
          items: {
            type: "object",
            properties: {
              rank: { type: "integer" },
              cosine_similarity_score: { type: "number" },
              cross_encoder_score: { type: "number" },
              id: { type: "string" },
              filename: { type: "string" },
              chunk_text: { type: "string" },
              chunk_index: { type: "string" },
              page_number: { type: "integer" },
              created_at: { type: "string", format: "ISO 8601" },
              start_pos: { type: "integer" },
              end_pos: { type: "integer" },
              start_line: { type: "integer" },
              end_line: { type: "integer" },
              sentence_count: { type: "integer" },
            },
          },
        },
      },
    },
    example: {
      request: {
        query: "machine learning algorithms",
        limit: 5,
        min_cosine_similarity: 0.5,
        min_cross_score: 0.0,
        expand_query: true,
        rerank: true,
      },
      response: {
        distance_metric: "cosine",
        total_results: 5,
        chunks: [
          {
            rank: 1,
            cosine_similarity_score: 0.89,
            cross_encoder_score: 0.92,
            id: "chunk_123",
            filename: "ml_guide.pdf",
            chunk_text: "Machine learning algorithms are computational methods that...",
            chunk_index: "0",
            page_number: 5,
            created_at: "2024-01-15T10:30:00Z",
            start_pos: 1250,
            end_pos: 2100,
            start_line: 45,
            end_line: 52,
            sentence_count: 8,
          },
        ],
      },
    },
  },
  {
    id: "clear-database",
    title: "Clear Database",
    method: "DELETE",
    endpoint: "/v1/pdf-rag/clear",
    description: "Remove all chunks and embeddings from the database.",
    icon: <Trash2 className="h-5 w-5" />,
    requestSchema: {
      type: "none",
      properties: {},
    },
    responseSchema: {
      type: "object",
      properties: {
        message: { type: "string" },
        chunks_removed: { type: "integer" },
      },
    },
    example: {
      request: {},
      response: {
        message: "Database cleared successfully",
        chunks_removed: 150,
      },
    },
  },
  {
    id: "simple-chat",
    title: "Simple Chat",
    method: "POST",
    endpoint: "/v1/chat/simple",
    description: "Basic GPT-powered chat without RAG.",
    icon: <MessageSquare className="h-5 w-5" />,
    requestSchema: {
      type: "application/json",
      properties: {
        message: { type: "string", description: "User message" },
        user_id: { type: "string", format: "UUID", description: "Unique user identifier" },
        conversation_history: {
          type: "array",
          items: {
            type: "object",
            properties: {
              role: { type: "string", enum: ["user", "assistant"] },
              content: { type: "string" },
            },
          },
        },
      },
    },
    responseSchema: {
      type: "object",
      properties: {
        response: { type: "string" },
        user_id: { type: "string", format: "UUID" },
      },
    },
    example: {
      request: {
        message: "What is artificial intelligence?",
        user_id: "550e8400-e29b-41d4-a716-446655440000",
        conversation_history: [],
      },
      response: {
        response:
          "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines...",
        user_id: "550e8400-e29b-41d4-a716-446655440000",
      },
    },
  },
  {
    id: "rag-chat",
    title: "RAG Chat",
    method: "POST",
    endpoint: "/v1/chat/rag",
    description: "GPT-powered chat with automatic knowledge base lookup and source citations.",
    icon: <Bot className="h-5 w-5" />,
    requestSchema: {
      type: "application/json",
      properties: {
        message: { type: "string", description: "User message" },
        user_id: { type: "string", format: "UUID", description: "Unique user identifier" },
        conversation_history: {
          type: "array",
          items: {
            type: "object",
            properties: {
              role: { type: "string", enum: ["user", "assistant"] },
              content: { type: "string" },
            },
          },
        },
        limit: { type: "integer", default: 10 },
        min_cosine_similarity: { type: "number", default: 0.5 },
        min_cross_score: { type: "number", default: 0.0 },
        expand_query: { type: "boolean", default: true },
        rerank: { type: "boolean", default: true },
      },
    },
    responseSchema: {
      type: "object",
      properties: {
        response: { type: "string" },
        user_id: { type: "string", format: "UUID" },
        used_rag: { type: "boolean" },
        rag_sources: {
          type: "array",
          items: {
            type: "object",
            properties: {
              rank: { type: "integer" },
              cosine_similarity_score: { type: "number" },
              cross_encoder_score: { type: "number" },
              id: { type: "string" },
              filename: { type: "string" },
              chunk_text: { type: "string" },
              chunk_index: { type: "string" },
              page_number: { type: "integer" },
              created_at: { type: "string", format: "ISO 8601" },
              start_pos: { type: "integer" },
              end_pos: { type: "integer" },
              start_line: { type: "integer" },
              end_line: { type: "integer" },
              sentence_count: { type: "integer" },
            },
          },
        },
      },
    },
    example: {
      request: {
        message: "How do neural networks work?",
        user_id: "550e8400-e29b-41d4-a716-446655440000",
        conversation_history: [],
        limit: 5,
        min_cosine_similarity: 0.5,
        min_cross_score: 0.0,
        expand_query: true,
        rerank: true,
      },
      response: {
        response: "Based on the uploaded documents, neural networks work by...",
        user_id: "550e8400-e29b-41d4-a716-446655440000",
        used_rag: true,
        rag_sources: [
          {
            rank: 1,
            cosine_similarity_score: 0.91,
            cross_encoder_score: 0.88,
            id: "chunk_456",
            filename: "neural_networks.pdf",
            chunk_text: "Neural networks are computing systems inspired by biological neural networks...",
            chunk_index: "2",
            page_number: 12,
            created_at: "2024-01-15T10:30:00Z",
            start_pos: 2500,
            end_pos: 3200,
            start_line: 85,
            end_line: 95,
            sentence_count: 12,
          },
        ],
      },
    },
  },
]

export default function DocsPage() {
  const [selectedEndpoint, setSelectedEndpoint] = useState<string>(apiEndpoints[0].id)
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    request: true,
    response: true,
    example: false,
  })

  const currentEndpoint = apiEndpoints.find((ep) => ep.id === selectedEndpoint) || apiEndpoints[0]

  const toggleSection = (section: string) => {
    setOpenSections((prev) => ({ ...prev, [section]: !prev[section] }))
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const getMethodColor = (method: string) => {
    switch (method) {
      case "GET":
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200"
      case "POST":
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200"
      case "DELETE":
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200"
      default:
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200"
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {/* Page Header */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-serif font-bold text-foreground dark:text-foreground mb-3">
            API Documentation
          </h1>
          <p className="text-lg text-muted-foreground dark:text-muted-foreground/80 max-w-2xl mx-auto">
            Comprehensive API reference for BuildingRAGisNotAProblem. Explore endpoints, request/response schemas, and examples for seamless integration.
          </p>
        </div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <Card className="sticky top-8 border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
              <CardHeader className="border-b border-emerald-500/20 dark:border-emerald-400/30">
                <CardTitle className="text-foreground dark:text-foreground">API Endpoints</CardTitle>
                <CardDescription className="text-muted-foreground dark:text-muted-foreground/80">Select an endpoint to view its documentation</CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <nav className="space-y-1">
                  {apiEndpoints.map((endpoint) => (
                    <button
                      key={endpoint.id}
                      onClick={() => setSelectedEndpoint(endpoint.id)}
                      className={`w-full flex items-center gap-3 px-4 py-3 text-left text-sm transition-all duration-200 hover:bg-emerald-50 dark:hover:bg-emerald-950/20 ${
                        selectedEndpoint === endpoint.id 
                          ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-800 dark:text-emerald-200 border-r-2 border-emerald-500" 
                          : "text-muted-foreground dark:text-muted-foreground/80"
                      }`}
                    >
                      <div className={`${selectedEndpoint === endpoint.id ? 'text-emerald-600 dark:text-emerald-400' : 'text-emerald-500 dark:text-emerald-400'}`}>
                        {endpoint.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate">{endpoint.title}</div>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline" className={`text-xs ${getMethodColor(endpoint.method)} border-emerald-200 dark:border-emerald-700`}>
                            {endpoint.method}
                          </Badge>
                        </div>
                      </div>
                    </button>
                  ))}
                </nav>
              </CardContent>
            </Card>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Endpoint Header */}
            <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
              <CardHeader className="border-b border-emerald-500/20 dark:border-emerald-400/30">
                <div className="flex items-center gap-3">
                  <div className="text-emerald-600 dark:text-emerald-400">{currentEndpoint.icon}</div>
                  <div>
                    <CardTitle className="text-2xl text-foreground dark:text-foreground">{currentEndpoint.title}</CardTitle>
                    <div className="flex items-center gap-3 mt-2">
                      <Badge className={`${getMethodColor(currentEndpoint.method)} border-emerald-200 dark:border-emerald-700`}>
                        {currentEndpoint.method}
                      </Badge>
                      <code className="px-3 py-2 bg-muted/80 dark:bg-muted/60 rounded-md text-sm font-mono border border-emerald-200/50 dark:border-emerald-700/50 text-emerald-700 dark:text-emerald-300">
                        {currentEndpoint.endpoint}
                      </code>
                    </div>
                  </div>
                </div>
                <CardDescription className="text-base mt-4 text-muted-foreground dark:text-muted-foreground/80">
                  {currentEndpoint.description}
                </CardDescription>
              </CardHeader>
            </Card>

            {/* Request Schema */}
            <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
              <Collapsible open={openSections.request} onOpenChange={() => toggleSection("request")}>
                <CollapsibleTrigger asChild>
                  <CardHeader className="cursor-pointer hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20 transition-colors border-b border-emerald-500/20 dark:border-emerald-400/30">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-2 text-foreground dark:text-foreground">
                        Request Schema
                        <Badge variant="secondary" className="bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-700">
                          {currentEndpoint.requestSchema.type}
                        </Badge>
                      </CardTitle>
                      {openSections.request ? (
                        <ChevronDown className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      )}
                    </div>
                  </CardHeader>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <CardContent>
                    {Object.keys(currentEndpoint.requestSchema.properties).length > 0 ? (
                      <div className="space-y-4">
                        {Object.entries(currentEndpoint.requestSchema.properties).map(([key, value]: [string, any]) => (
                          <div key={key} className="border-2 border-emerald-200/50 dark:border-emerald-700/50 rounded-lg p-4 bg-background/80 dark:bg-background/60 hover:bg-emerald-50/30 dark:hover:bg-emerald-950/20 transition-colors">
                            <div className="flex items-center justify-between mb-2">
                              <code className="font-mono text-sm font-medium text-emerald-600 dark:text-emerald-400">{key}</code>
                              <Badge variant="outline" className="text-xs bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-700">
                                {value.type}
                              </Badge>
                            </div>
                            {value.description && (
                              <p className="text-sm text-muted-foreground dark:text-muted-foreground/80 mb-2">{value.description}</p>
                            )}
                            {value.enum && (
                              <div className="text-xs text-muted-foreground dark:text-muted-foreground/70">
                                Options: {value.enum.map((option: string) => `"${option}"`).join(", ")}
                              </div>
                            )}
                            {value.default !== undefined && (
                              <div className="text-xs text-muted-foreground dark:text-muted-foreground/70">
                                Default: {JSON.stringify(value.default)}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-muted-foreground">No request body required.</p>
                    )}
                  </CardContent>
                </CollapsibleContent>
              </Collapsible>
            </Card>

            {/* Response Schema */}
            <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
              <Collapsible open={openSections.response} onOpenChange={() => toggleSection("response")}>
                <CollapsibleTrigger asChild>
                  <CardHeader className="cursor-pointer hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20 transition-colors border-b border-emerald-500/20 dark:border-emerald-400/30">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-foreground dark:text-foreground">Response Schema</CardTitle>
                      {openSections.response ? (
                        <ChevronDown className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      )}
                    </div>
                  </CardHeader>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <CardContent>
                    <ScrollArea className="h-[400px]">
                      <pre className="text-sm bg-muted p-4 rounded-lg overflow-x-auto">
                        <code>{JSON.stringify(currentEndpoint.responseSchema, null, 2)}</code>
                      </pre>
                    </ScrollArea>
                  </CardContent>
                </CollapsibleContent>
              </Collapsible>
            </Card>

            {/* Example */}
            <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
              <Collapsible open={openSections.example} onOpenChange={() => toggleSection("example")}>
                <CollapsibleTrigger asChild>
                  <CardHeader className="cursor-pointer hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20 transition-colors border-b border-emerald-500/20 dark:border-emerald-400/30">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-foreground dark:text-foreground">Example</CardTitle>
                      {openSections.example ? (
                        <ChevronDown className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      )}
                    </div>
                  </CardHeader>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <CardContent>
                    <Tabs defaultValue="request" className="space-y-4">
                      <TabsList className="bg-background/80 dark:bg-background/60 border-2 border-emerald-500/30 dark:border-emerald-400/40">
                        <TabsTrigger value="request" className="data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md">Request</TabsTrigger>
                        <TabsTrigger value="response" className="data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md">Response</TabsTrigger>
                        <TabsTrigger value="curl" className="data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md">cURL</TabsTrigger>
                      </TabsList>
                      <TabsContent value="request">
                        <div className="relative">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="absolute top-2 right-2 z-10"
                            onClick={() => copyToClipboard(JSON.stringify(currentEndpoint.example.request, null, 2))}
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                          <ScrollArea className="h-[300px]">
                            <pre className="text-sm bg-muted p-4 rounded-lg overflow-x-auto">
                              <code>{JSON.stringify(currentEndpoint.example.request, null, 2)}</code>
                            </pre>
                          </ScrollArea>
                        </div>
                      </TabsContent>
                      <TabsContent value="response">
                        <div className="relative">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="absolute top-2 right-2 z-10"
                            onClick={() => copyToClipboard(JSON.stringify(currentEndpoint.example.response, null, 2))}
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                          <ScrollArea className="h-[300px]">
                            <pre className="text-sm bg-muted p-4 rounded-lg overflow-x-auto">
                              <code>{JSON.stringify(currentEndpoint.example.response, null, 2)}</code>
                            </pre>
                          </ScrollArea>
                        </div>
                      </TabsContent>
                      <TabsContent value="curl">
                        <div className="relative">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="absolute top-2 right-2 z-10"
                            onClick={() =>
                              copyToClipboard(`curl -X ${currentEndpoint.method} \\
  https://api.buildingraginotaproblem.com${currentEndpoint.endpoint} \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(currentEndpoint.example.request)}'`)
                            }
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                          <ScrollArea className="h-[300px]">
                            <pre className="text-sm bg-muted p-4 rounded-lg overflow-x-auto">
                              <code>{`curl -X ${currentEndpoint.method} \\
  https://api.buildingraginotaproblem.com${currentEndpoint.endpoint} \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(currentEndpoint.example.request, null, 2)}'`}</code>
                            </pre>
                          </ScrollArea>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </CollapsibleContent>
              </Collapsible>
            </Card>


          </div>
        </div>
      </div>
    </div>
  )
}

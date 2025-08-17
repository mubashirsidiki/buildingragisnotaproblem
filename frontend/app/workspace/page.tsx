"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Search, MessageSquare, Send, Bot, User, FileText, Trash2, Copy, ChevronDown, ChevronUp } from "lucide-react"
import { searchChunks, simpleChat, ragChat } from "@/lib/api"

interface SearchResult {
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

interface ChatMessage {
  role: "user" | "assistant"
  content: string
  sources?: SearchResult[]
  timestamp: Date
  isTyping?: boolean
}

interface SearchConfig {
  limit: number
  min_cosine_similarity: number
  min_cross_score: number
  expand_query: boolean
  rerank: boolean
}

export default function WorkspacePage() {
  const [activeTab, setActiveTab] = useState("search")
  const [searchQuery, setSearchQuery] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const [searchConfig, setSearchConfig] = useState<SearchConfig>({
    limit: 10,
    min_cosine_similarity: 0.5,
    min_cross_score: 0.0,
    expand_query: true,
    rerank: true,
  })

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('workspace-chat-messages')
      return saved ? JSON.parse(saved) : []
    }
    return []
  })
  const [chatInput, setChatInput] = useState("")
  const [isChatting, setIsChatting] = useState(false)
  const [chatMode, setChatMode] = useState<"simple" | "rag">("rag")
  const [userId] = useState(() => "demo_user_123")
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  const [searchResults, setSearchResults] = useState<SearchResult[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('workspace-search-results')
      return saved ? JSON.parse(saved) : []
    }
    return []
  })

  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [chatMessages])

  // Save chat messages to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined' && chatMessages.length > 0) {
      localStorage.setItem('workspace-chat-messages', JSON.stringify(chatMessages))
    }
  }, [chatMessages])

  // Save search results to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined' && searchResults.length > 0) {
      localStorage.setItem('workspace-search-results', JSON.stringify(searchResults))
    }
  }, [searchResults])

  // Typewriter effect for AI messages
  const typewriterEffect = (fullMessage: string, messageIndex: number) => {
    const words = fullMessage.split(' ')
    let currentWordIndex = 0
    
    const interval = setInterval(() => {
      if (currentWordIndex < words.length) {
        const currentText = words.slice(0, currentWordIndex + 1).join(' ')
        
        setChatMessages((prev: ChatMessage[]) => {
          const newMessages = [...prev]
          if (newMessages[messageIndex]) {
            newMessages[messageIndex] = { ...newMessages[messageIndex], content: currentText, isTyping: true }
          }
          return newMessages
        })
        
        currentWordIndex++
      } else {
        // Finish typing effect
        setChatMessages((prev: ChatMessage[]) => {
          const newMessages = [...prev]
          if (newMessages[messageIndex]) {
            newMessages[messageIndex] = { ...newMessages[messageIndex], content: fullMessage, isTyping: false }
          }
          return newMessages
        })
        clearInterval(interval)
      }
    }, 80) // Faster typing for smoother effect
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    try {
      const result = await searchChunks(searchQuery, searchConfig)
      
      if (result.error) {
        console.error("Search failed:", result.error)
        return
      }
      
      if (result.data) {
        setSearchResults(result.data.chunks || [])
      }
    } catch (error) {
      console.error("Search failed:", error)
    } finally {
      setIsSearching(false)
    }
  }

  const handleChat = async () => {
    if (!chatInput.trim()) return

    const userMessage: ChatMessage = {
      role: "user",
      content: chatInput,
      timestamp: new Date(),
    }

    setChatMessages((prev: ChatMessage[]) => [...prev, userMessage])
    setChatInput("")
    setIsChatting(true)

    try {
      const requestBody = {
        message: chatInput,
        user_id: userId,
        conversation_history: chatMessages.map((msg) => ({
          role: msg.role,
          content: msg.content,
        })),
      }

      let result
      if (chatMode === "simple") {
        result = await simpleChat(requestBody)
      } else {
        result = await ragChat(requestBody, searchConfig)
      }

      if (result.error) {
        console.error("Chat failed:", result.error)
        return
      }

      if (result.data?.response) {
        const assistantMessage: ChatMessage = {
          role: "assistant",
          content: "",
          sources: (result.data as any).rag_sources || [],
          timestamp: new Date(),
          isTyping: true,
        }
        
        // Calculate the correct message index before adding the message
        const messageIndex = chatMessages.length + 1 // +1 because we already added the user message
        
        setChatMessages((prev) => [...prev, assistantMessage])
        
        // Start typewriter effect with a short delay
        setTimeout(() => typewriterEffect(result.data!.response, messageIndex), 500)
      }
    } catch (error) {
      console.error("Chat failed:", error)
    } finally {
      setIsChatting(false)
    }
  }

  const clearChat = () => {
    setChatMessages([])
    setExpandedSources(new Set())
    if (typeof window !== 'undefined') {
      localStorage.removeItem('workspace-chat-messages')
      localStorage.removeItem('workspace-search-results')
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const toggleSources = (messageIndex: number) => {
    setExpandedSources((prev: Set<number>) => {
      const newSet = new Set(prev)
      if (newSet.has(messageIndex)) {
        newSet.delete(messageIndex)
      } else {
        newSet.add(messageIndex)
      }
      return newSet
    })
  }

  // Rotating thinking messages
  const ThinkingMessage = () => {
    const [messageIndex, setMessageIndex] = useState(0)
    const thinkingMessages = [
      "Analyzing your question...",
      "Processing information...",
      "Searching through documents...",
      "Crafting a response...",
      "Almost there...",
      "Finalizing answer..."
    ]

    useEffect(() => {
      const interval = setInterval(() => {
        setMessageIndex(prev => (prev + 1) % thinkingMessages.length)
      }, 2000)
      return () => clearInterval(interval)
    }, [])

    return (
      <p className="text-sm text-muted-foreground">
        {thinkingMessages[messageIndex]}
      </p>
    )
  }



  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="mb-6">
          <h1 className="text-2xl font-serif font-bold text-foreground bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            Workspace
          </h1>
          <p className="text-muted-foreground mt-1">Search documents and chat with AI</p>
        </div>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-2 bg-background/80 dark:bg-background/60 backdrop-blur-sm border-2 border-emerald-500/30 dark:border-emerald-400/40 shadow-md rounded-lg p-1">
            <TabsTrigger 
              value="search" 
              className="flex items-center gap-2 data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md rounded-md transition-all duration-200"
            >
              <Search className="h-4 w-4" />
              Vector Search
            </TabsTrigger>
            <TabsTrigger 
              value="chat" 
              className="flex items-center gap-2 data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md rounded-md transition-all duration-200"
            >
              <MessageSquare className="h-4 w-4" />
              AI Chat
            </TabsTrigger>
          </TabsList>

          {/* Search Tab */}
          <TabsContent value="search" className="space-y-4">
            <div className="w-full space-y-4">
              <Card className="backdrop-blur-sm border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg hover:shadow-xl transition-all duration-300">
                <CardHeader className="border-b border-emerald-500/20 dark:border-emerald-400/30">
                  <CardTitle className="flex items-center gap-2 text-foreground dark:text-foreground">
                    <Search className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                    Vector Search
                  </CardTitle>
                  <CardDescription className="text-muted-foreground dark:text-muted-foreground/80">
                    Search through your uploaded documents using semantic similarity
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex gap-2">
                    <Input
                      placeholder="Enter your search query..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                      className="flex-1"
                      disabled={isSearching}
                    />
                    <Button 
                      onClick={handleSearch} 
                      disabled={!searchQuery.trim() || isSearching}
                      className="min-w-[100px] relative"
                    >
                      {isSearching ? (
                        <>
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                          </div>
                          <span className="opacity-0">Searching...</span>
                        </>
                      ) : (
                        <>
                          <Search className="h-4 w-4 mr-2" />
                          Search
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Search Results */}
              {isSearching ? (
                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <div className="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                    <span>Searching for relevant chunks...</span>
                  </div>
                  
                  {/* Skeleton loader for results */}
                  <div className="space-y-3">
                    {[...Array(3)].map((_, i) => (
                      <div key={i} className="p-4 border border-border/50 rounded-lg space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="h-4 bg-muted rounded w-32 animate-pulse"></div>
                          <div className="h-4 bg-muted rounded w-16 animate-pulse"></div>
                        </div>
                        <div className="space-y-2">
                          <div className="h-3 bg-muted rounded w-full animate-pulse"></div>
                          <div className="h-3 bg-muted rounded w-3/4 animate-pulse"></div>
                          <div className="h-3 bg-muted rounded w-1/2 animate-pulse"></div>
                        </div>
                        <div className="flex gap-2">
                          <div className="h-3 bg-muted rounded w-20 animate-pulse"></div>
                          <div className="h-3 bg-muted rounded w-16 animate-pulse"></div>
                          <div className="h-3 bg-muted rounded w-24 animate-pulse"></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : searchResults.length > 0 ? (
                <Card className="backdrop-blur-sm border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg hover:shadow-xl transition-all duration-300">
                  <CardHeader className="border-b border-emerald-500/20 dark:border-emerald-400/30">
                    <CardTitle className="flex items-center gap-2 text-foreground dark:text-foreground">
                      <Search className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                      Search Results ({searchResults.length})
                    </CardTitle>
                    <CardDescription className="text-muted-foreground dark:text-muted-foreground/80">
                      Found {searchResults.length} relevant document chunks
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-[400px] pr-4">
                      <div className="space-y-4">
                        {searchResults.map((result, index) => (
                          <div key={index} className="p-4 border-2 border-emerald-500/30 dark:border-emerald-400/40 rounded-lg hover:bg-muted/30 dark:hover:bg-muted/20 transition-all duration-200 bg-background/80 dark:bg-background/60 shadow-sm hover:shadow-md">
                            <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                              <div className="flex items-center gap-2 min-w-0">
                                <FileText className="h-4 w-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                                <span className="font-medium text-sm truncate text-foreground dark:text-foreground">{result.filename}</span>
                                <Badge variant="outline" className="text-xs flex-shrink-0 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-700">
                                  Page {result.page_number}
                                </Badge>
                              </div>
                              <div className="flex items-center gap-2 text-xs text-muted-foreground dark:text-muted-foreground/80 flex-shrink-0">
                                <span className="bg-emerald-100 dark:bg-emerald-900/20 px-2 py-1 rounded-md">Rank: {result.rank}</span>
                                <span className="bg-emerald-100 dark:bg-emerald-900/20 px-2 py-1 rounded-md">Score: {result.cosine_similarity_score.toFixed(3)}</span>
                              </div>
                            </div>
                            <p className="text-sm text-foreground dark:text-foreground mb-3 leading-relaxed break-words whitespace-pre-wrap">{result.chunk_text}</p>
                            <div className="flex items-center gap-4 text-xs text-muted-foreground dark:text-muted-foreground/80 flex-wrap pt-2 border-t border-emerald-200/30 dark:border-emerald-700/30">
                              <span>Chunk {result.chunk_index}</span>
                              <span>{result.sentence_count} sentences</span>
                              <span>Lines {result.start_line}-{result.end_line}</span>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => copyToClipboard(result.chunk_text)}
                                className="h-6 px-2 text-xs hover:bg-emerald-50 dark:hover:bg-emerald-950/20 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-700"
                              >
                                <Copy className="h-3 w-3 mr-1" />
                                Copy
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              ) : (
                <div className="text-center text-muted-foreground py-8">
                  <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Enter a search query to find relevant document chunks</p>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Chat Tab */}
          <TabsContent value="chat" className="space-y-4">
                          <Card className="h-[600px] flex flex-col border-2 border-emerald-500/40 dark:border-emerald-400/50 bg-card/95 dark:bg-card/90 backdrop-blur-sm shadow-lg">
                <CardHeader className="flex-shrink-0 border-b border-emerald-500/30 dark:border-emerald-400/40 py-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2 text-foreground dark:text-foreground text-lg">
                        <MessageSquare className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                        AI Chat
                      </CardTitle>
                      <CardDescription className="text-muted-foreground dark:text-muted-foreground/80 text-sm">
                        Chat with AI using {chatMode === "rag" ? "RAG-enhanced" : "simple"} mode
                      </CardDescription>
                    </div>
                    <Button variant="outline" size="sm" onClick={clearChat} className="border-emerald-500/40 dark:border-emerald-400/50 hover:bg-emerald-50 dark:hover:bg-emerald-950/20 text-emerald-700 dark:text-emerald-300">
                      <Trash2 className="h-4 w-4 mr-2" />
                      Clear
                    </Button>
                  </div>
                </CardHeader>

              <div className="flex-1 flex flex-col min-h-0">
                <ScrollArea className="flex-1 px-6 py-4">
                  <div className="space-y-6">
                    {chatMessages.length === 0 && (
                      <div className="text-center text-muted-foreground py-8">
                        <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p>Start a conversation with the AI assistant</p>
                      </div>
                    )}
                    
                    {chatMessages.map((message, index) => (
                      <div key={index} className="space-y-4">
                        <div className={`flex gap-3 ${message.role === "user" ? "justify-end" : ""}`}>
                          <div className={`flex gap-3 max-w-[80%] ${message.role === "user" ? "flex-row-reverse" : ""}`}>
                            <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0">
                              {message.role === "user" ? (
                                <div className="w-full h-full bg-emerald-600 rounded-full flex items-center justify-center">
                                  <User className="h-4 w-4 text-white" />
                                </div>
                              ) : (
                                <div className="w-full h-full bg-muted rounded-full flex items-center justify-center">
                                  <Bot className="h-4 w-4 text-foreground" />
                                </div>
                              )}
                            </div>
                            
                            <div className="space-y-2">
                              <div className={`p-4 rounded-lg relative ${
                                message.role === "user" 
                                  ? "bg-emerald-600 text-white shadow-lg border-2 border-emerald-500/30" 
                                  : "bg-muted/80 dark:bg-muted/60 text-foreground dark:text-foreground border-2 border-emerald-500/30 dark:border-emerald-400/40 shadow-md"
                              }`}>
                                {/* RAG Flag for AI messages with sources */}
                                {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                                  <div className="absolute -top-2 -left-2">
                                    <Badge className="bg-emerald-500 dark:bg-emerald-400 text-white text-xs px-2 py-1 shadow-md border-0">
                                      RAG
                                    </Badge>
                                  </div>
                                )}
                                
                                <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
                                  {message.content}
                                  {message.isTyping && (
                                    <span className="inline-block w-0.5 h-4 bg-current ml-1 animate-pulse"></span>
                                  )}
                                </p>
                              </div>
                              
                              {/* Sources */}
                              {message.sources && message.sources.length > 0 && (
                                <div className="space-y-3">
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => toggleSources(index)}
                                    className="h-auto p-2 justify-start text-xs text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 hover:bg-emerald-50 dark:hover:bg-emerald-950/20"
                                  >
                                    <span>Sources ({message.sources.length})</span>
                                    {expandedSources.has(index) ? (
                                      <ChevronUp className="h-3 w-3 ml-2" />
                                    ) : (
                                      <ChevronDown className="h-3 w-3 ml-2" />
                                    )}
                                  </Button>
                                  
                                  {expandedSources.has(index) && (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                      {message.sources.map((source, sourceIndex) => (
                                        <div key={sourceIndex} className="p-3 bg-background/80 dark:bg-background/60 border-2 border-emerald-500/30 dark:border-emerald-400/40 rounded-lg hover:bg-muted/30 dark:hover:bg-muted/20 transition-colors shadow-md">
                                          <div className="flex items-center gap-2 mb-2">
                                            <FileText className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
                                            <span className="font-medium text-xs truncate text-foreground dark:text-foreground">{source.filename}</span>
                                            <Badge variant="secondary" className="text-xs bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-700">
                                              Page {source.page_number}
                                            </Badge>
                                          </div>
                                          
                                          {/* Additional source information */}
                                          <div className="space-y-1 mb-2">
                                            <div className="flex items-center gap-3 text-xs text-muted-foreground dark:text-muted-foreground/70">
                                              <span>Rank: {source.rank}</span>
                                              <span>Score: {source.cosine_similarity_score?.toFixed(3) || 'N/A'}</span>
                                            </div>
                                            <div className="flex items-center gap-3 text-xs text-muted-foreground dark:text-muted-foreground/70">
                                              <span>Chunk: {source.chunk_index}</span>
                                              <span>Lines: {source.start_line}-{source.end_line}</span>
                                            </div>
                                          </div>
                                          
                                          <p className="text-xs text-muted-foreground dark:text-muted-foreground/80 line-clamp-3 leading-relaxed border-t border-emerald-200/30 dark:border-emerald-700/30 pt-2">
                                            {source.chunk_text}
                                          </p>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                    
                    {isChatting && (
                      <div className="flex gap-3">
                        <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center">
                          <Bot className="h-4 w-4 text-foreground animate-pulse" />
                        </div>
                        <div className="p-4 bg-muted rounded-lg">
                          <ThinkingMessage />
                        </div>
                      </div>
                    )}
                    
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>

                <div className="border-t border-emerald-500/30 dark:border-emerald-400/40 px-4 py-3 bg-muted/20 dark:bg-muted/10">
                  <div className="flex gap-3">
                    <Textarea
                      placeholder="Type your message..."
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault()
                          handleChat()
                        }
                      }}
                      className="flex-1 min-h-[50px] resize-none border-emerald-500/30 dark:border-emerald-400/40 focus:border-emerald-500 dark:focus:border-emerald-400 bg-background dark:bg-background/80"
                      disabled={isChatting}
                    />
                    <Button 
                      onClick={handleChat} 
                      disabled={isChatting || !chatInput.trim()}
                      className="self-end bg-emerald-600 hover:bg-emerald-700 dark:bg-emerald-500 dark:hover:bg-emerald-600 shadow-md min-w-[50px] h-[50px]"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
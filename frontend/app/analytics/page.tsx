"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { BarChart3, DollarSign, Zap, TrendingUp, RefreshCw, Calculator } from "lucide-react"
import { getUserAnalytics, getPricingConfig } from "@/lib/api"

interface Operation {
  id: string
  operation_type: string
  model: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
  estimated_cost: number
  timestamp: string
}

interface Analytics {
  total_operations: number
  total_tokens: number
  total_cost: number
  by_operation: Record<string, { count: number; tokens: number; cost: number }>
  recent_operations: Operation[]
}

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [pricing, setPricing] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [userId] = useState(() => "demo_user_123")

  const loadAnalytics = async () => {
    setLoading(true)
    try {
             // Check analytics for all operation types
       const [userAnalytics, uploadAnalytics, searchAnalytics, pricingResult] = await Promise.all([
         getUserAnalytics(userId),
         getUserAnalytics("upload_operations"),
         getUserAnalytics("search_operations"),
         getPricingConfig()
       ])
       
       console.log('🔍 Raw API Responses:')
       console.log('  User Analytics:', userAnalytics)
       console.log('  Upload Analytics:', uploadAnalytics)
       console.log('  Search Analytics:', searchAnalytics)
       console.log('  Pricing Result:', pricingResult)
      
      // Combine analytics from all sources
      let combinedAnalytics = {
        total_operations: 0,
        total_tokens: 0,
        total_cost: 0,
        by_operation: {},
        recent_operations: []
      }
      
             const sources = [
         { data: userAnalytics.data?.data, label: "User" },
         { data: uploadAnalytics.data?.data, label: "Upload" },
         { data: searchAnalytics.data?.data, label: "Search" }
       ]
      
      sources.forEach(source => {
        console.log(`🔍 Processing ${source.label} source:`, source.data)
        console.log(`🔍 ${source.label} source.data type:`, typeof source.data)
        console.log(`🔍 ${source.label} source.data keys:`, Object.keys(source.data || {}))
        console.log(`🔍 ${source.label} source.data.total_operations:`, source.data?.total_operations)
        console.log(`🔍 ${source.label} source.data.total_tokens:`, source.data?.total_tokens)
        console.log(`🔍 ${source.label} source.data.total_cost:`, source.data?.total_cost)
        
        if (source.data) {
          const operations = source.data.total_operations || 0
          const tokens = source.data.total_tokens || 0
          const cost = source.data.total_cost || 0
          
          console.log(`📊 ${source.label} - Operations: ${operations}, Tokens: ${tokens}, Cost: ${cost}`)
          
          combinedAnalytics.total_operations += operations
          combinedAnalytics.total_tokens += tokens
          combinedAnalytics.total_cost += cost
          
          console.log(`📈 Running totals - Operations: ${combinedAnalytics.total_operations}, Tokens: ${combinedAnalytics.total_tokens}, Cost: ${combinedAnalytics.total_cost}`)
          
          // Merge operations
          Object.entries(source.data.by_operation || {}).forEach(([op, data]) => {
            console.log(`🏷️ Processing operation: ${op}`, data)
            
            if (!combinedAnalytics.by_operation[op]) {
              combinedAnalytics.by_operation[op] = { count: 0, tokens: 0, cost: 0 }
            }
            combinedAnalytics.by_operation[op].count += data.count
            combinedAnalytics.by_operation[op].tokens += data.tokens
            combinedAnalytics.by_operation[op].cost += data.cost
            
            console.log(`✅ Updated ${op}: count=${combinedAnalytics.by_operation[op].count}, tokens=${combinedAnalytics.by_operation[op].tokens}, cost=${combinedAnalytics.by_operation[op].cost}`)
          })
          
          // Merge recent operations
          const recentOps = source.data.recent_operations || []
          console.log(`📝 Adding ${recentOps.length} recent operations from ${source.label}`)
          combinedAnalytics.recent_operations.push(...recentOps)
        } else {
          console.log(`❌ ${source.label} source has no data`)
        }
      })
      
      // Sort recent operations by timestamp
      combinedAnalytics.recent_operations.sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      ).slice(0, 10) // Keep only 10 most recent
      
      console.log('🔍 Analytics Data:', {
        userAnalytics: userAnalytics.data,
        uploadAnalytics: uploadAnalytics.data,
        searchAnalytics: searchAnalytics.data,
        combinedAnalytics
      })
      
      console.log('📊 Final Combined Analytics:', {
        total_operations: combinedAnalytics.total_operations,
        total_tokens: combinedAnalytics.total_tokens,
        total_cost: combinedAnalytics.total_cost,
        by_operation: combinedAnalytics.by_operation,
        recent_operations_count: combinedAnalytics.recent_operations.length
      })
      
      setAnalytics(combinedAnalytics)
      if (pricingResult.data) setPricing(pricingResult.data)
    } catch (error) {
      console.error("Failed to load analytics:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAnalytics()
  }, [userId])

  const formatCost = (cost: number) => `$${cost.toFixed(4)}`
  const formatNumber = (num: number) => num.toLocaleString()

  const getOperationIcon = (type: string) => {
    switch (type) {
      case 'simple_chat': return '💬'
      case 'rag_chat': return '🤖'
      case 'search': return '🔍'
      case 'upload': return '📁'
      default: return '⚡'
    }
  }

  const getOperationColor = (type: string) => {
    switch (type) {
      case 'simple_chat': return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
      case 'rag_chat': return 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
      case 'search': return 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300'
      case 'upload': return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300'
      default: return 'bg-gray-100 dark:bg-gray-900/30 text-gray-700 dark:text-gray-300'
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-center h-64">
            <div className="flex items-center gap-2">
              <RefreshCw className="h-6 w-6 animate-spin" />
              <span>Loading analytics...</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-serif font-bold text-foreground dark:text-foreground mb-3">
              Token Usage Analytics
            </h1>
            <p className="text-lg text-muted-foreground dark:text-muted-foreground/80">
              Track your API usage, costs, and performance metrics
            </p>
          </div>
          <Button onClick={loadAnalytics} variant="outline" className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Operations</CardTitle>
              <BarChart3 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{analytics?.total_operations || 0}</div>
              <p className="text-xs text-muted-foreground">API calls made</p>
            </CardContent>
          </Card>

          <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Tokens</CardTitle>
              <Zap className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(analytics?.total_tokens || 0)}</div>
              <p className="text-xs text-muted-foreground">Input + Output tokens</p>
            </CardContent>
          </Card>

          <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
              <DollarSign className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatCost(analytics?.total_cost || 0)}</div>
              <p className="text-xs text-muted-foreground">Estimated spend</p>
            </CardContent>
          </Card>

          <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Cost/Op</CardTitle>
              <TrendingUp className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {analytics?.total_operations ? formatCost(analytics.total_cost / analytics.total_operations) : '$0.0000'}
              </div>
              <p className="text-xs text-muted-foreground">Per operation</p>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Operations Breakdown */}
          <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
            <CardHeader className="border-b border-emerald-500/20 dark:border-emerald-400/30">
              <CardTitle className="flex items-center gap-2 text-foreground dark:text-foreground">
                <BarChart3 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                Usage by Operation Type
              </CardTitle>
              <CardDescription className="text-muted-foreground dark:text-muted-foreground/80">
                Breakdown of tokens and costs by operation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {analytics?.by_operation && Object.entries(analytics.by_operation).map(([type, data]) => (
                  <div key={type} className="flex items-center justify-between p-3 border border-emerald-200/50 dark:border-emerald-700/50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <span className="text-lg">{getOperationIcon(type)}</span>
                      <div>
                        <Badge className={getOperationColor(type)}>
                          {type.replace('_', ' ')}
                        </Badge>
                        <p className="text-sm text-muted-foreground mt-1">
                          {data.count} operations
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium">{formatNumber(data.tokens)} tokens</div>
                      <div className="text-sm text-muted-foreground">{formatCost(data.cost)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Operations */}
          <Card className="border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
            <CardHeader className="border-b border-emerald-500/20 dark:border-emerald-400/30">
              <CardTitle className="flex items-center gap-2 text-foreground dark:text-foreground">
                <TrendingUp className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                Recent Operations
              </CardTitle>
              <CardDescription className="text-muted-foreground dark:text-muted-foreground/80">
                Latest API calls and their token usage
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                <div className="space-y-3">
                  {analytics?.recent_operations?.map((op) => (
                    <div key={op.id} className="p-3 border border-emerald-200/50 dark:border-emerald-700/50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{getOperationIcon(op.operation_type)}</span>
                          <Badge className={getOperationColor(op.operation_type)} variant="outline">
                            {op.operation_type}
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            {op.model}
                          </Badge>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {new Date(op.timestamp).toLocaleString()}
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span className="text-muted-foreground">Input:</span> {formatNumber(op.input_tokens)}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Output:</span> {formatNumber(op.output_tokens)}
                        </div>
                        <div>
                          <span className="text-muted-foreground">Cost:</span> {formatCost(op.estimated_cost)}
                        </div>
                      </div>
                    </div>
                  ))}
                  {(!analytics?.recent_operations || analytics.recent_operations.length === 0) && (
                    <div className="text-center text-muted-foreground py-8">
                      <Calculator className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>No operations recorded yet</p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Pricing Configuration */}
        {pricing && (
          <Card className="mt-8 border-2 border-emerald-500/30 dark:border-emerald-400/40 bg-card/95 dark:bg-card/90 shadow-lg">
            <CardHeader className="border-b border-emerald-500/20 dark:border-emerald-400/30">
              <CardTitle className="flex items-center gap-2 text-foreground dark:text-foreground">
                <DollarSign className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                Current Pricing Configuration
              </CardTitle>
              <CardDescription className="text-muted-foreground dark:text-muted-foreground/80">
                Cost per 1M tokens by model
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(pricing).map(([model, rates]: [string, any]) => (
                  <div key={model} className="p-3 border border-emerald-200/50 dark:border-emerald-700/50 rounded-lg">
                    <div className="font-medium text-sm mb-2">{model}</div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Input:</span>
                        <span>${rates.input}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Output:</span>
                        <span>${rates.output}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

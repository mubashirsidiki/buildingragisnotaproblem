import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowRight, Upload, Search, MessageSquare, Code, Zap, Shield, Github, Heart } from "lucide-react"
import Link from "next/link"

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background decorative elements - Minimal */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Just one subtle center accent */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-emerald-400/25 to-emerald-500/25 rounded-full blur-3xl"></div>
        
        {/* One subtle grid line */}
        <div className="absolute top-1/2 left-0 w-full h-px bg-gradient-to-r from-transparent via-emerald-400/40 to-transparent"></div>
      </div>
      
      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <Badge variant="secondary" className="mb-6 backdrop-blur-sm bg-background/80 border-border/50">
            <Heart className="h-3 w-3 mr-1" />
            Open Source & Free
          </Badge>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-serif font-bold text-foreground mb-6">
            Building RAG is <span className="text-emerald-600 bg-gradient-to-r from-emerald-600 to-emerald-500 bg-clip-text text-transparent">Not A Problem</span>
          </h1>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Skip the complexity and get started with RAG in minutes. A complete, customizable RAG system that you can
            run locally, modify, and learn from — perfect for developers at any level.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/upload">
              <Button size="lg" className="bg-emerald-600 hover:bg-emerald-500 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                Try It Now
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
            <Link href="/docs">
              <Button size="lg" variant="outline" className="backdrop-blur-sm bg-background/80 border-border/50 hover:bg-background/90 transition-all duration-300 transform hover:-translate-y-1">
                <Github className="mr-2 h-4 w-4" />
                View on GitHub
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-muted/30 relative">
        {/* Background elements for features section - Minimal */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Just one subtle accent */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-to-r from-emerald-400/20 to-emerald-500/20 rounded-full blur-2xl"></div>
        </div>
        
        <div className="max-w-6xl mx-auto relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-serif font-bold text-foreground mb-4">
              Everything you need to learn RAG
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              A complete RAG implementation with clear examples, documentation, and customizable components. Perfect for
              learning, prototyping, or building your own RAG applications.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-border/50 hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 backdrop-blur-sm bg-background/80">
              <CardHeader>
                <Upload className="h-8 w-8 text-emerald-600 mb-2" />
                <CardTitle>PDF Processing</CardTitle>
                <CardDescription>
                  Upload and process PDFs with customizable chunking strategies and embedding generation
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Multiple chunking strategies</li>
                  <li>• Automatic embedding generation</li>
                  <li>• Vector database integration</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-border/50 hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 backdrop-blur-sm bg-background/80">
              <CardHeader>
                <Search className="h-8 w-8 text-emerald-600 mb-2" />
                <CardTitle>Smart Search</CardTitle>
                <CardDescription>
                  Semantic search with similarity scoring and intelligent result ranking
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Vector similarity search</li>
                  <li>• Configurable parameters</li>
                  <li>• Result explanations</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-border/50 hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 backdrop-blur-sm bg-background/80">
              <CardHeader>
                <MessageSquare className="h-8 w-8 text-emerald-600 mb-2" />
                <CardTitle>AI Chat</CardTitle>
                <CardDescription>
                  Context-aware chat with automatic knowledge retrieval and source citations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• RAG-enhanced responses</li>
                  <li>• Source citations</li>
                  <li>• Conversation memory</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-border/50 hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 backdrop-blur-sm bg-background/80">
              <CardHeader>
                <Code className="h-8 w-8 text-emerald-600 mb-2" />
                <CardTitle>Developer Friendly</CardTitle>
                <CardDescription>
                  Clean code, comprehensive docs, and examples to help you understand and extend
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Well-documented APIs</li>
                  <li>• Code examples</li>
                  <li>• Easy to customize</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-border/50 hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 backdrop-blur-sm bg-background/80">
              <CardHeader>
                <Zap className="h-8 w-8 text-emerald-600 mb-2" />
                <CardTitle>Ready to Run</CardTitle>
                <CardDescription>Get started immediately with Docker or run locally with minimal setup</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Docker support</li>
                  <li>• Local development</li>
                  <li>• Quick setup guide</li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-border/50 hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 backdrop-blur-sm bg-background/80">
              <CardHeader>
                <Shield className="h-8 w-8 text-emerald-600 mb-2" />
                <CardTitle>Open & Transparent</CardTitle>
                <CardDescription>
                  Fully open source with no hidden costs, vendor lock-in, or usage limits
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• MIT licensed</li>
                  <li>• Community driven</li>
                  <li>• No usage limits</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 relative">
        {/* Background elements for CTA section - Minimal */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Just one subtle accent */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-gradient-to-r from-emerald-400/20 to-emerald-500/20 rounded-full blur-3xl"></div>
        </div>
        
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <h2 className="text-3xl sm:text-4xl font-serif font-bold text-foreground mb-4">Ready to dive into RAG?</h2>
          <p className="text-lg text-muted-foreground mb-8">
            Join the community of developers learning and building with RAG. No signup required, no limits, just start
            building.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/workspace">
              <Button size="lg" className="bg-emerald-600 hover:bg-emerald-500 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                Start Exploring
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
            <Link href="/docs">
              <Button size="lg" variant="outline" className="backdrop-blur-sm bg-background/80 border-border/50 hover:bg-background/90 transition-all duration-300 transform hover:-translate-y-1">
                Read the Docs
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 py-12 px-4 sm:px-6 lg:px-8 relative">
        {/* Background elements for footer - Minimal */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Just one subtle accent */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-to-r from-emerald-400/20 to-emerald-500/20 rounded-full blur-2xl"></div>
        </div>
        
        <div className="max-w-6xl mx-auto relative z-10">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <h3 className="text-lg font-serif font-bold text-foreground">BuildingRAGisNotAProblem</h3>
              <p className="text-sm text-muted-foreground">Making RAG accessible for every developer</p>
            </div>
            <div className="flex gap-6 text-sm text-muted-foreground">
              <Link href="/docs" className="hover:text-foreground transition-colors">
                Documentation
              </Link>
              <Link href="/workspace" className="hover:text-foreground transition-colors">
                Try Demo
              </Link>
              <Link href="/upload" className="hover:text-foreground transition-colors">
                Upload PDFs
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

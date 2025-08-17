"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Upload, FileText, CheckCircle, AlertCircle } from "lucide-react"
import { uploadPdf } from "@/lib/api"

interface UploadConfig {
  chunking_mode: string
  max_chunk_size: number
  breakpoint_threshold_type: string
  breakpoint_threshold_amount: number
  chunk_overlap: number
  sentence_group_size: number
}

interface ProcessingResult {
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

export default function UploadPage() {
  const [files, setFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [results, setResults] = useState<ProcessingResult[]>([])
  const [errors, setErrors] = useState<string[]>([])
  const [config, setConfig] = useState<UploadConfig>({
    chunking_mode: "sentence",
    max_chunk_size: 1500,
    breakpoint_threshold_type: "percentile",
    breakpoint_threshold_amount: 95.0,
    chunk_overlap: 200,
    sentence_group_size: 3,
  })

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const pdfFiles = acceptedFiles.filter((file) => file.type === "application/pdf")
    setFiles((prev) => [...prev, ...pdfFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [],
    },
    multiple: true,
    noClick: false,
    noKeyboard: false,
  })

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleUpload = async () => {
    if (files.length === 0) return

    setIsUploading(true)
    setUploadProgress(0)
    setResults([])
    setErrors([])

    try {
      for (let i = 0; i < files.length; i++) {
        try {
          const file = files[i]
          const formData = new FormData()
          formData.append("file", file)
          formData.append("chunking_mode", config.chunking_mode)
          formData.append("max_chunk_size", config.max_chunk_size.toString())
          formData.append("breakpoint_threshold_type", config.breakpoint_threshold_type)
          formData.append("breakpoint_threshold_amount", config.breakpoint_threshold_amount.toString())
          formData.append("chunk_overlap", config.chunk_overlap.toString())
          formData.append("sentence_group_size", config.sentence_group_size.toString())

          // Use the API service
          const result = await uploadPdf(formData)
          
          if (result.error) {
            console.error('Upload error:', result.error)
            throw new Error(result.error)
          }
          
          if (result.data) {
            console.log('Upload response:', result.data)
            setResults((prev) => [...prev, result.data as ProcessingResult])
          } else {
            console.error('No data in upload response')
            throw new Error('Upload failed: No response data')
          }

          setUploadProgress(((i + 1) / files.length) * 100)
        } catch (fileError) {
          console.error(`Failed to upload ${files[i].name}:`, fileError)
          const errorMessage = `Failed to upload ${files[i].name}: ${fileError instanceof Error ? fileError.message : 'Unknown error'}`
          setErrors(prev => [...prev, errorMessage])
          // Continue with next file instead of stopping all uploads
          setUploadProgress(((i + 1) / files.length) * 100)
        }
      }
    } catch (error) {
      console.error("Upload failed:", error)
    } finally {
      setIsUploading(false)
      setFiles([])
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* File Drop Zone */}
            <Card>
              <CardHeader>
                <CardTitle>Upload PDF Files</CardTitle>
                <CardDescription>
                  Drag and drop your PDF files here or click to browse. Files will be processed into semantic chunks.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                    isDragActive
                      ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20"
                      : "border-border hover:border-emerald-500"
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  {isDragActive ? (
                    <p className="text-emerald-600">Drop the files here...</p>
                  ) : (
                    <div>
                      <p className="text-foreground font-medium mb-2">Drop PDF files here, or click to select</p>
                      <p className="text-sm text-muted-foreground">Supports multiple file upload</p>
                    </div>
                  )}
                </div>

                {/* File List */}
                {files.length > 0 && (
                  <div className="mt-6 space-y-2">
                    <h4 className="font-medium text-foreground">Selected Files:</h4>
                    {files.map((file, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                        <div className="flex items-center gap-3">
                          <FileText className="h-5 w-5 text-emerald-600" />
                          <div>
                            <p className="font-medium text-foreground">{file.name}</p>
                            <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                          </div>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => removeFile(index)}>
                          Remove
                        </Button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Upload Progress */}
                {isUploading && (
                  <div className="mt-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-foreground">Processing files...</span>
                      <span className="text-sm text-muted-foreground">{Math.round(uploadProgress)}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-2" />
                  </div>
                )}

                {/* Upload Button */}
                <div className="mt-6">
                  <Button
                    onClick={handleUpload}
                    disabled={files.length === 0 || isUploading}
                    className="w-full bg-emerald-600 hover:bg-emerald-500"
                  >
                    {isUploading ? "Processing..." : `Upload ${files.length} file${files.length !== 1 ? "s" : ""}`}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Results */}
            {results.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-emerald-600" />
                    Processing Results
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {results.map((result, index) => (
                    <div key={index} className="p-4 bg-muted rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-medium text-foreground">{result.filename}</h4>
                        <Badge variant="secondary">{result.chunks_created} chunks</Badge>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Pages</p>
                          <p className="font-medium text-foreground">{result.pdf_stats?.total_pages || 0}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Words</p>
                          <p className="font-medium text-foreground">{result.pdf_stats?.total_words?.toLocaleString() || '0'}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Sentences</p>
                          <p className="font-medium text-foreground">
                            {result.pdf_stats?.total_sentences?.toLocaleString() || '0'}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Chunking Mode</p>
                          <p className="font-medium text-foreground">{result.chunking_mode || 'Unknown'}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Errors */}
            {errors.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertCircle className="h-5 w-5 text-red-600" />
                    Upload Errors
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {errors.map((error, index) => (
                    <div key={index} className="p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg">
                      <p className="text-red-700 dark:text-red-300 text-sm">{error}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </div>

          {/* Configuration Panel */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Processing Configuration</CardTitle>
                <CardDescription>Choose between semantic analysis or fixed-size chunking</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="chunking_mode">Chunking Mode</Label>
                  <Select
                    value={config.chunking_mode}
                    onValueChange={(value) => setConfig((prev) => ({ ...prev, chunking_mode: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sentence">Sentence (Semantic)</SelectItem>
                      <SelectItem value="length">Length (Fixed Size)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="max_chunk_size">Max Chunk Size</Label>
                  <Input
                    id="max_chunk_size"
                    type="number"
                    value={config.max_chunk_size}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, max_chunk_size: Number.parseInt(e.target.value) }))
                    }
                  />
                </div>

                <div>
                  <Label htmlFor="breakpoint_threshold_type">Breakpoint Threshold Type</Label>
                  <Select
                    value={config.breakpoint_threshold_type}
                    onValueChange={(value) => setConfig((prev) => ({ ...prev, breakpoint_threshold_type: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="percentile">Percentile</SelectItem>
                      <SelectItem value="standard_deviation">Standard Deviation</SelectItem>
                      <SelectItem value="interquartile">Interquartile</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="breakpoint_threshold_amount">Breakpoint Threshold Amount</Label>
                  <Input
                    id="breakpoint_threshold_amount"
                    type="number"
                    step="0.01"
                    value={config.breakpoint_threshold_amount}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, breakpoint_threshold_amount: Number.parseFloat(e.target.value) }))
                    }
                  />
                </div>

                <div>
                  <Label htmlFor="chunk_overlap">Chunk Overlap</Label>
                  <Input
                    id="chunk_overlap"
                    type="number"
                    value={config.chunk_overlap}
                    onChange={(e) => setConfig((prev) => ({ ...prev, chunk_overlap: Number.parseInt(e.target.value) }))}
                  />
                </div>

                <div>
                  <Label htmlFor="sentence_group_size">Sentence Group Size</Label>
                  <Input
                    id="sentence_group_size"
                    type="number"
                    value={config.sentence_group_size}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, sentence_group_size: Number.parseInt(e.target.value) }))
                    }
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-amber-500" />
                  Configuration Tips
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground space-y-2">
                <p>
                  • <strong>Sentence mode</strong> uses AI to preserve meaning boundaries
                </p>
                <p>
                  • <strong>Length mode</strong> creates fixed-size chunks quickly
                </p>
                <p>
                  • <strong>Sentence mode</strong> is default for best quality
                </p>
                <p>
                  • <strong>Length mode</strong> is faster and cheaper
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

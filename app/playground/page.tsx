'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { Navbar } from '@/components/navbar'
import { Footer } from '@/components/footer'
import { Button } from '@/components/ui/button'
import { Play, Sliders, Cpu, ArrowRight, RotateCcw, AlertCircle, Leaf, ScanLine } from 'lucide-react'
import './studio.css'
import { ModelUploadZone } from '@/components/playground/model-upload-zone'
import { InputMediaTabs } from '@/components/playground/input-media-tabs'
import { ResultsPanel } from '@/components/playground/results-panel'
import { CountEvaluation } from '@/components/playground/count-evaluation'
import { GroundTruthBox } from '@/lib/count-evaluation'
import { ModelInfo, InputImageMedia, InferenceResult } from '@/components/playground/types'
import { runCVModelInference } from '@/lib/cv-inference-engine'

export default function PlaygroundPage() {
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [inferenceError, setInferenceError] = useState<string | null>(null)
  const [inputMedia, setInputMedia] = useState<InputImageMedia | null>(null)
  const [imageVersion, setImageVersion] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<InferenceResult | null>(null)
  const [annotation, setAnnotation] = useState<{ image: string; boxes: GroundTruthBox[] | null } | null>(null)

  // Inference Settings
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.4)
  const [showBbox, setShowBbox] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const activeRun = useRef<AbortController | null>(null)
  const runVersion = useRef(0)
  useEffect(() => () => { runVersion.current++; activeRun.current?.abort() }, [])

  const cancelRun = () => {
    runVersion.current++
    activeRun.current?.abort()
    activeRun.current = null
    setIsRunning(false)
  }

  const canRunModel = !!model && !!inputMedia

  const handleRunModel = async () => {
    if (!canRunModel || !model || !inputMedia || activeRun.current) return
    const controller = new AbortController()
    activeRun.current = controller
    const version = ++runVersion.current
    const isCurrent = () => version === runVersion.current && !controller.signal.aborted
    const finishRun = () => { if (isCurrent()) { activeRun.current = null; setIsRunning(false) } }

    setIsRunning(true)
    setResults(null)
    setInferenceError(null)

    const runBrowserFallback = async (modeLabel?: string) => {
      try {
        const imageSrc =
          inputMedia.appliedFilters && inputMedia.appliedFilters.length > 0 && inputMedia.previewUrl
            ? inputMedia.previewUrl
            : inputMedia.previewUrl || (inputMedia.file ? URL.createObjectURL(inputMedia.file) : '')

        const clientResult = await runCVModelInference(imageSrc, model, confidenceThreshold)
        if (modeLabel) {
          clientResult.device = modeLabel
        }
        if (isCurrent()) setResults(clientResult)
      } catch (clientErr: any) {
        console.error('Browser CV Engine error:', clientErr)
        if (isCurrent()) setInferenceError(`Lỗi xử lý hình ảnh: ${clientErr.message}`)
      } finally {
        finishRun()
      }
    }

    if (!model.file && !model.serverModelId) {
      await runBrowserFallback('Client CV Engine (WebGPU / Wasm)')
      return
    }

    // 2. If user uploaded a custom model file, try server-side Python first
    try {
      const formData = new FormData()
      if (model.serverModelId) formData.append('modelId', model.serverModelId)
      else if (model.file) formData.append('modelFile', model.file)
      if (model.labelsFile) {
        formData.append('labelsFile', model.labelsFile)
      }
      if (model.imgsz && model.imgsz > 0) {
        formData.append('imgsz', String(model.imgsz))
      }
      if (model.norm && model.norm !== 'auto') {
        formData.append('norm', model.norm)
      }

      if (inputMedia.appliedFilters && inputMedia.appliedFilters.length > 0 && inputMedia.previewUrl) {
        formData.append('imageBase64', inputMedia.previewUrl)
      } else if (inputMedia.file) {
        formData.append('imageFile', inputMedia.file)
      } else if (inputMedia.previewUrl) {
        formData.append('imageBase64', inputMedia.previewUrl)
      }

      formData.append('confThreshold', String(confidenceThreshold))
      formData.append('taskType', model.type)

      const res = await fetch('/api/inference', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      const data = await res.json()
      if (!isCurrent()) return

      if (data && data.success) {
        if (typeof data.detectionThreshold === 'number') setConfidenceThreshold(data.detectionThreshold)
        setResults(data)
        finishRun()
      } else {
        const errorMsg = data?.error || 'Quá trình suy luận mô hình trên máy chủ thất bại.'
        console.error('Backend inference failed:', errorMsg)
        setInferenceError(errorMsg)
        finishRun()
      }
    } catch (err: any) {
      if (!isCurrent()) return
      console.error('Backend inference error:', err)
      setInferenceError(`Không thể kết nối tới máy chủ AI: ${err.message || err}`)
      finishRun()
    }
  }

  const handleReset = () => {
    setResults(null)
    setInferenceError(null)
  }

  return (
    <div className="fruitcount-studio flex flex-col min-h-screen bg-background">
      <Navbar fruitcount />

      <main className="flex-1 py-7">
        <div className="studio-container mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="studio-heading mb-7">
            <div>
              <p className="studio-eyebrow"><Leaf className="size-3.5" /> COMPUTER VISION · FRUITCOUNT</p>
              <h1 className="text-3xl sm:text-4xl font-extrabold text-foreground mt-2">Live Counting <span className="studio-title-accent">Studio</span></h1>
              <p className="text-sm text-muted-foreground mt-2 max-w-2xl">Từ ảnh vườn cây đến số đếm. Chọn mô hình, thêm ảnh và đối chiếu kết quả với dữ liệu thực tế.</p>
            </div>
            <div className="flex flex-wrap gap-3 items-center"><Link href="/compare" className="rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold text-primary">So sánh 2 model →</Link>
            <span className="studio-session"><span className={isRunning ? 'studio-dot is-busy' : 'studio-dot'} />{isRunning ? 'Đang suy luận' : 'Không gian thử nghiệm'}</span></div>
          </div>

          <div className="studio-workspace">
          <aside className="studio-sidebar" aria-label="Cấu hình thử nghiệm">
            <div className="studio-intro-card">
              <span className="studio-eyebrow">ĐẾM QUẢ BẰNG AI</span>
              <p>Một bức ảnh.<br /><strong>Nhiều góc nhìn về khu vườn.</strong></p>
              <div className="studio-intro-steps"><span>01 Chọn model</span><span>02 Thêm ảnh</span><span>03 Đánh giá</span></div>
            </div>
            <fieldset disabled={isRunning} inert={isRunning} className="min-w-0 space-y-4">
            {/* 1. Model Upload Section */}
            <div className="studio-config-card border border-border rounded-xl p-4 bg-card space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center">
                    1
                  </span>
                  <h2 className="font-semibold text-foreground text-base">
                    Mô hình suy luận
                  </h2>
                </div>
                {model && (
                  <span className="text-xs text-muted-foreground font-mono">
                    {model.fileName}
                  </span>
                )}
              </div>

              <ModelUploadZone
                currentModel={model}
                onModelSelect={(selected) => {
                  if (activeRun.current) return
                  runVersion.current++
                  setInferenceError(null)
                  setModel(selected)
                  setResults(null)
                }}
              />
            </div>

            {/* 2. Image Upload Section */}
            <div className="studio-config-card border border-border rounded-xl p-4 bg-card space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-accent text-accent-foreground text-xs font-bold flex items-center justify-center">
                    2
                  </span>
                  <h2 className="font-semibold text-foreground text-base">
                    Ảnh đầu vào
                  </h2>
                </div>
                {inputMedia && (
                  <span className="text-xs text-muted-foreground font-mono truncate max-w-[150px]">
                    {inputMedia.fileName}
                  </span>
                )}
              </div>

              <InputMediaTabs
                inputMedia={inputMedia}
                setInputMedia={(media) => {
                  if (activeRun.current) return
                  runVersion.current++
                  setImageVersion(version => version + 1)
                  setAnnotation(null)
                  setInferenceError(null)
                  setInputMedia(media)
                  setResults(null)
                }}
              />
            </div>
          {inputMedia && <div className="studio-ground-truth"><CountEvaluation key={imageVersion} imageName={inputMedia.fileName} imageUrl={inputMedia.previewUrl} results={results} threshold={confidenceThreshold}
            onBoxesChange={boxes => setAnnotation({ image: inputMedia.previewUrl, boxes })} /></div>}
            </fieldset>
          {/* 3. Settings & Execution Control Bar */}
          <div className="studio-run-card border border-border rounded-xl p-4 bg-card flex flex-col gap-4">
            <div className="flex flex-wrap items-center gap-4 text-xs">
              <div className="flex items-center gap-2 font-medium text-foreground">
                <Cpu className="w-4 h-4 text-primary" />
                <span>Thiết bị phần cứng:</span>
                <span className="text-xs text-muted-foreground">{model?.file || model?.serverModelId ? 'Máy chủ tự chọn CUDA nếu khả dụng, nếu không dùng CPU' : 'Xử lý ảnh trong trình duyệt'}</span>
              </div>

              <div className="h-4 w-px bg-border hidden sm:block" />

              <div className="flex items-center gap-2 text-muted-foreground">
                <Sliders className="w-3.5 h-3.5" />
                <span>Ngưỡng tin cậy ban đầu:</span>
                <span className="font-mono font-bold text-foreground">
                  {(confidenceThreshold * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2 w-full">
              {results && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  className="text-xs h-10 px-3"
                  title="Xoá kết quả để chạy lại"
                >
                  <RotateCcw className="w-3.5 h-3.5 mr-1" />
                  Đặt lại
                </Button>
              )}

              {isRunning && <Button variant="outline" onClick={cancelRun}>Hủy chạy</Button>}
              <Button
                size="lg"
                disabled={!canRunModel || isRunning}
                onClick={handleRunModel}
                className="studio-run-button flex-1 h-12 font-bold bg-primary hover:bg-primary/90 text-primary-foreground transition-all disabled:opacity-50"
              >
                {isRunning ? (
                  <div className="flex items-center gap-2 text-xs">
                    <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
                    <span>Đang chạy suy luận...</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-xs">
                    <Play className="w-4 h-4 fill-current" />
                    <span>Chạy đếm quả</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </div>
                )}
              </Button>
            </div>
          </div>
          </aside>

          <section className="studio-output" aria-label="Kết quả đếm quả">
            <div className="studio-output-heading">
              <div className="flex items-center gap-2"><ScanLine className="size-5" /><h2 className="font-bold">Không gian kết quả</h2></div>
              <span>{results ? model?.name : 'Sẵn sàng cho lần chạy tiếp theo'}</span>
            </div>

          {/* Running Progress Bar */}
          {isRunning && (
            <div className="mb-6 space-y-1.5 animate-fadeInUp">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Đang tải trọng số và phân tích tensor ảnh...</span>
                <span className="font-mono">Processing...</span>
              </div>
              <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
                <div className="h-full bg-primary animate-pulse w-full" />
              </div>
            </div>
          )}

          {/* Inference Error Display */}
          {inferenceError && !isRunning && (
            <div className="mb-6 flex items-start gap-3 bg-red-500/10 border border-red-500/25 rounded-xl p-4 animate-fadeInUp">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
              <div className="text-sm">
                <p className="font-semibold text-red-600 dark:text-red-400 mb-0.5">Lỗi khi chạy mô hình</p>
                <p className="text-muted-foreground text-xs leading-relaxed">{inferenceError}</p>
              </div>
            </div>
          )}

          <div className="studio-results border border-border rounded-xl p-4 sm:p-5 bg-card">
            {results && inputMedia && model ? (
              <ResultsPanel
                groundTruthBoxes={annotation?.image === inputMedia.previewUrl ? annotation.boxes : null}
                inputMedia={inputMedia}
                model={model}
                results={results}
                confidenceThreshold={confidenceThreshold}
                onConfidenceChange={setConfidenceThreshold}
                showBbox={showBbox}
                onToggleBbox={setShowBbox}
                showLabels={showLabels}
                onToggleLabels={setShowLabels}
              />
            ) : (
              <div className="studio-empty text-center py-16 px-4 mx-auto space-y-5">
                <div className="studio-empty-art" aria-hidden="true"><Leaf className="studio-empty-leaf" /><ScanLine className="studio-empty-scan" /><span className="studio-empty-fruit fruit-one" /><span className="studio-empty-fruit fruit-two" /><span className="studio-empty-fruit fruit-three" /></div>
                <div className="studio-eyebrow justify-center">YOUR ORCHARD, IN FOCUS</div>
                <div>
                  <h3 className="font-bold text-lg text-foreground mb-1">
                    Bắt đầu từ một bức ảnh
                  </h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Chọn <strong>mô hình</strong> và thêm <strong>ảnh vườn cây</strong> ở bảng bên cạnh, rồi nhấn <strong>Chạy đếm quả</strong>. Kết quả và bản đồ mật độ sẽ hiển thị tại đây theo mô hình đã chọn.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-2 text-left text-xs font-medium">
                  <div className={`p-2.5 rounded-lg border flex items-center gap-2 ${
                    model ? 'border-green-500/30 bg-green-500/10 text-green-600 dark:text-green-400' : 'border-border bg-muted/30 text-muted-foreground'
                  }`}>
                    <span>{model ? '✓' : '○'}</span>
                    <span>1. Mô hình: {model ? model.name : 'Chưa chọn'}</span>
                  </div>
                  <div className={`p-2.5 rounded-lg border flex items-center gap-2 ${
                    inputMedia ? 'border-green-500/30 bg-green-500/10 text-green-600 dark:text-green-400' : 'border-border bg-muted/30 text-muted-foreground'
                  }`}>
                    <span>{inputMedia ? '✓' : '○'}</span>
                    <span>2. Ảnh test: {inputMedia ? inputMedia.fileName : 'Chưa tải'}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
          </section>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}

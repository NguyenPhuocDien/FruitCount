'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, GitCompareArrows, Play, RefreshCw } from 'lucide-react'
import { Navbar } from '@/components/navbar'
import { Footer } from '@/components/footer'
import { Button } from '@/components/ui/button'
import { InputMediaTabs } from '@/components/playground/input-media-tabs'
import { CountEvaluation } from '@/components/playground/count-evaluation'
import { ResultsPanel } from '@/components/playground/results-panel'
import { InferenceResult, InputImageMedia, ModelInfo, TaskType } from '@/components/playground/types'
import { GroundTruthBox, countErrors } from '@/lib/count-evaluation'
import { requestComparisonInference, runModelPair } from '@/lib/compare-inference'
import '../playground/studio.css'
import './compare.css'

interface Entry { id: string; name: string; family?: string; task?: TaskType; classes?: string[]; ready: boolean; error?: string }
interface Outcome { status: 'waiting' | 'running' | 'success' | 'error'; result: InferenceResult | null; error: string | null }
const empty = (): Outcome[] => [0, 1].map(() => ({ status: 'waiting', result: null, error: null }))
const number = (value: number) => new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 }).format(value)

function ModelResult({ model, image, outcome, boxes, threshold, onThreshold }: {
  model: Entry | undefined; image: InputImageMedia | null; outcome: Outcome; boxes: GroundTruthBox[] | null;
  threshold: number; onThreshold: (value: number) => void
}) {
  const [showBbox, setShowBbox] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const info: ModelInfo | null = model?.task ? { id: model.id, name: model.name, fileName: model.id,
    serverModelId: model.id, type: model.task, sizeMB: 0, inputShape: 'Theo cấu hình model',
    architecture: model.family || '', classesCount: model.classes?.length || 0, classes: model.classes } : null
  return <article className="arena-result-card">
    <header><h2>{model?.name || 'Chưa chọn model'}</h2><span>{({ waiting: 'Chờ chạy', running: 'Đang chạy…', success: 'Hoàn tất', error: 'Có lỗi' })[outcome.status]}</span></header>
    {outcome.error && <p role="alert" className="m-4 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-500">{outcome.error}</p>}
    {outcome.result && image && info ? <div className="p-4"><ResultsPanel model={info} inputMedia={image}
      results={outcome.result} groundTruthBoxes={boxes} confidenceThreshold={threshold} onConfidenceChange={onThreshold}
      showBbox={showBbox} onToggleBbox={setShowBbox} showLabels={showLabels} onToggleLabels={setShowLabels} /></div> :
      !outcome.error && <div className="arena-result-placeholder"><GitCompareArrows className={`size-10 ${outcome.status === 'running' ? 'animate-pulse' : ''}`} />
        <p>{outcome.status === 'running' ? 'Đang nạp model và xử lý ảnh…' : 'Kết quả sẽ xuất hiện ở đây sau khi chạy.'}</p></div>}
  </article>
}

export default function ComparePage() {
  const [models, setModels] = useState<Entry[]>([])
  const [ids, setIds] = useState<[string, string]>(['', ''])
  const [image, setImage] = useState<InputImageMedia | null>(null)
  const [imageVersion, setImageVersion] = useState(0)
  const [annotation, setAnnotation] = useState<{ image: string; boxes: GroundTruthBox[] | null } | null>(null)
  const [outcomes, setOutcomes] = useState<Outcome[]>(empty)
  const [thresholds, setThresholds] = useState([0.4, 0.4])
  const [running, setRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)
  const libraryRequest = useRef(0)
  const boxes = annotation?.image === image?.previewUrl ? annotation?.boxes ?? null : null
  const selected = ids.map(id => models.find(model => model.id === id))
  const refresh = async () => {
    const request = ++libraryRequest.current
    setLoading(true); setError('')
    try {
      const response = await fetch('/api/local-models', { cache: 'no-store' })
      const data = await response.json()
      if (!response.ok || !Array.isArray(data.models)) throw new Error(data.error || 'Không đọc được danh sách model.')
      if (request === libraryRequest.current) {
        setModels(data.models); setOutcomes(empty())
        setIds(current => current.map(id => data.models.some((m: Entry) => m.id === id && m.ready) ? id : '') as [string, string])
      }
    } catch (err) { if (request === libraryRequest.current) setError((err as Error).message) }
    finally { if (request === libraryRequest.current) setLoading(false) }
  }
  useEffect(() => { void refresh(); return () => { controller.current?.abort(); libraryRequest.current++ } }, [])
  const canRun = Boolean(image && ids[0] && ids[1] && ids[0] !== ids[1] && selected.every(m => m?.ready) && !running && !loading)
  const setThreshold = (index: number, value: number) => setThresholds(current => current.map((v, i) => i === index ? value : v))
  const run = async () => {
    if (!canRun || !image) return
    const active = new AbortController()
    controller.current = active
    setRunning(true); setOutcomes(empty()); setError('')
    try {
      await runModelPair(ids, id => requestComparisonInference(id, image, active.signal),
        index => setOutcomes(current => current.map((v, i) => i === index ? { ...v, status: 'running' } : v)),
        (index, result, failure) => {
          setOutcomes(current => current.map((v, i) => i === index ? { status: failure ? 'error' : 'success', result, error: failure } : v))
          if (result) setThreshold(index, result.detectionThreshold ?? 0.4)
        }, active.signal)
    } finally { if (!active.signal.aborted) setRunning(false) }
  }
  const counts = outcomes.map((o, i) => o.result?.predicted_count ?? (o.result?.modelType === 'detection'
    ? o.result.detections.filter(d => d.confidence >= thresholds[i]).length : null))
  return <div className="fruitcount-studio min-h-screen flex flex-col">
    <Navbar fruitcount />
    <main className="studio-container arena-container w-full mx-auto px-4 sm:px-6 lg:px-8 py-7 flex-1">
      <div className="studio-heading mb-6"><div><p className="studio-eyebrow"><GitCompareArrows className="size-4" /> MODEL ARENA</p>
        <h1 className="text-3xl font-extrabold mt-2">Hai model. <span className="studio-title-accent">Cùng một ảnh.</span></h1>
        <p className="text-sm text-muted-foreground mt-2">Đối chiếu số đếm, thời gian xử lý và kết quả trực quan trên cùng dữ liệu.</p></div>
        <Link href="/playground" className="flex items-center gap-2 text-sm text-primary"><ArrowLeft className="size-4" /> Live Studio</Link></div>
      <fieldset disabled={running} className="arena-setup" aria-label="Cấu hình so sánh hai model">
        <section className="arena-model-selection">
          <div className="flex justify-between items-center gap-2"><h2 className="font-bold">1 · Chọn hai model</h2>
            <button type="button" disabled={loading} onClick={refresh} className="text-xs text-primary flex items-center gap-1"><RefreshCw className="size-3" />{loading ? 'Đang đọc…' : 'Làm mới'}</button></div>
          {[0, 1].map(index => <label key={index} className="block text-sm mt-4">Model {index === 0 ? 'A' : 'B'}
            <select value={ids[index]} onChange={e => { setIds(current => current.map((id, i) => i === index ? e.target.value : id) as [string, string]); setOutcomes(empty()) }} className="block w-full mt-2 bg-background border border-border">
              <option value="">Chọn model đã cài…</option>
              {models.filter(model => model.id !== 'dino' && model.id !== 'resnet').map(model => <option key={model.id} value={model.id} disabled={!model.ready || model.id === ids[1 - index]}>{model.name}{model.ready ? '' : ` — ${model.error}`}</option>)}
            </select></label>)}
          <p className="text-xs text-muted-foreground mt-4">Model được chạy lần lượt A → B để hạn chế tranh chấp bộ nhớ GPU. Mỗi model giữ cấu hình suy luận riêng.</p>
        </section>
        <section className="arena-image-selection"><h2 className="font-bold mb-4">2 · Ảnh dùng chung</h2>
          <InputMediaTabs inputMedia={image} setInputMedia={media => { setImage(media); setImageVersion(version => version + 1); setOutcomes(empty()); setAnnotation(null) }} />
        </section>
        {image && <div className="arena-ground-truth"><CountEvaluation key={imageVersion} imageName={image.fileName} imageUrl={image.previewUrl}
          results={null} threshold={0.4} onBoxesChange={value => setAnnotation({ image: image.previewUrl, boxes: value })} /></div>}
      </fieldset>
      {error && <p role="alert" className="my-3 text-sm text-red-500">{error}</p>}
      <div className="arena-run-bar"><p className="text-sm text-muted-foreground">{running ? 'Đang chạy so sánh; kết quả từng model sẽ xuất hiện khi hoàn tất.' : 'Chọn hai model khác nhau và thêm ảnh để bắt đầu. Label/mask là tùy chọn.'}</p>
        <Button onClick={run} disabled={!canRun} className="studio-run-button h-12 px-6"><Play className="size-4" />{running ? 'Đang so sánh…' : 'Chạy so sánh'}</Button></div>
      <section className="arena-summary" aria-label="Bảng so sánh số đếm">
        <h2 className="font-bold mb-3">Tổng quan so sánh</h2>
        <div className="overflow-x-auto"><table><thead><tr><th>Chỉ số</th>{selected.map((m, i) => <th key={i}>{m?.name || `Model ${i === 0 ? 'A' : 'B'}`}</th>)}</tr></thead>
          <tbody>
            <tr><th>Dự đoán cuối</th>{counts.map((count, i) => <td key={i} className="arena-count">{count === null ? '—' : `${number(count)} quả`}</td>)}</tr>
            <tr><th>Thực tế từ label / mask</th>{[0, 1].map(i => <td key={i}>{boxes === null ? 'Chưa tải ground truth' : `${boxes.length} quả`}</td>)}</tr>
            <tr><th>Sai số tuyệt đối</th>{counts.map((count, i) => <td key={i}>{count === null || boxes === null ? '—' : `${number(countErrors(count, boxes.length).absolute)} quả`}</td>)}</tr>
            <tr><th>Đếm trực tiếp từ box</th>{outcomes.map((o, i) => <td key={i}>{o.result?.count_raw === undefined ? '—' : `${o.result.count_raw} quả`}</td>)}</tr>
            <tr><th>Thời gian xử lý model</th>{outcomes.map((o, i) => <td key={i}>{o.result ? `${number(o.result.latencyMs)} ms` : '—'}</td>)}</tr>
            <tr><th>Thiết bị thực tế</th>{outcomes.map((o, i) => <td key={i}>{o.result?.device || '—'}</td>)}</tr>
          </tbody></table></div>
        <p className="text-xs text-muted-foreground mt-3">Thời gian theo backend, gồm các bước mà runner đo; không phải benchmark thuần mạng. So sánh này áp dụng cho một ảnh, không kết luận chất lượng trên toàn dataset.</p>
      </section>
      <div className="arena-results">{outcomes.map((outcome, index) => <ModelResult key={index} model={selected[index]} image={image} outcome={outcome}
        boxes={boxes} threshold={thresholds[index]} onThreshold={value => setThreshold(index, value)} />)}</div>
    </main><Footer />
  </div>
}

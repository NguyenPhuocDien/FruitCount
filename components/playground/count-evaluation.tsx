'use client'

import { useEffect, useRef, useState } from 'react'
import { countErrors, parseYoloBoxes, GroundTruthBox } from '@/lib/count-evaluation'
import { InferenceResult } from './types'
import { readMask, MaskMode } from '@/lib/mask-evaluation'

export function CountEvaluation({ imageName, imageUrl, results, threshold, onBoxesChange }: {
  imageName: string; results: InferenceResult | null; threshold: number
  imageUrl: string
  onBoxesChange: (boxes: GroundTruthBox[] | null) => void
}) {
  const [groundTruth, setGroundTruth] = useState<{ name: string; count: number } | null>(null)
  const [error, setError] = useState('')
  const request = useRef(0)
  useEffect(() => () => { request.current++ }, [])
  const [source, setSource] = useState<'label' | 'mask'>('label')
  const [maskMode, setMaskMode] = useState<MaskMode>('instance')
  const [loading, setLoading] = useState(false)
  const reset = () => { request.current++; setLoading(false); setError(''); setGroundTruth(null); onBoxesChange(null) }
  const predicted = results?.predicted_count ?? (results?.modelType === 'detection'
    ? results.detections.filter(d => d.confidence >= threshold).length : undefined)
  const metrics = groundTruth && predicted !== undefined ? countErrors(predicted, groundTruth.count) : null
  return <section className="rounded-xl border border-border p-4 space-y-3 bg-card">
    <h3 className="text-sm font-semibold">Ground truth để đánh giá số đếm</h3>
    <label className="text-sm">Nguồn nhãn:
      <select value={source} onChange={e => { reset(); setSource(e.target.value as 'label' | 'mask') }} className="ml-2 rounded border border-border bg-background p-2">
        <option value="label">Label YOLO (.txt)</option><option value="mask">Mask (.png)</option>
      </select>
    </label>
    {source === 'mask' && <div className="space-y-2">
      <label className="text-sm">Loại mask:
        <select value={maskMode} onChange={e => { reset(); setMaskMode(e.target.value as MaskMode) }} className="ml-2 rounded border border-border bg-background p-2">
          <option value="instance">Instance: mỗi ID/màu là một quả</option>
          <option value="binary">Nhị phân: đếm vùng liên thông</option>
        </select>
      </label>
      <p className="text-xs text-muted-foreground">Dùng PNG gốc cùng kích thước ảnh, nền đen (0). Instance đếm mỗi ID/màu khác nền một lần, kể cả vùng bị tách rời. Nhị phân gộp vùng chạm nhau theo 8 hướng, có thể sai khi quả dính nhau; semantic mask không thể tự xác định số quả. Không dùng heatmap hoặc ảnh mask đã làm mượt.</p>
    </div>}
    <p className="text-xs text-muted-foreground">Ảnh đang đánh giá: {imageName}. {source === 'label' ? 'Mỗi dòng YOLO là một quả: class xc yc width height. File rỗng tương ứng 0 quả; đếm tất cả lớp.' : 'Mask được đọc thành số quả và bbox từng vùng để dùng trong chế độ Nghi bỏ sót.'}</p>
    <input key={`${source}-${maskMode}`} aria-label="Tải ground truth của ảnh" type="file" accept={source === 'label' ? '.txt' : '.png'} className="text-sm max-w-full" onChange={async event => {
      const id = ++request.current
      const file = event.target.files?.[0]
      setGroundTruth(null); setError('')
      onBoxesChange(null)
      if (!file) return
      setLoading(true)
      try {
        if (source === 'label' && file.size > 5 * 1024 * 1024) throw new Error('File label tối đa 5 MB.')
        const boxes = source === 'mask' ? await readMask(file, imageUrl, maskMode) : parseYoloBoxes(await file.text())
        if (id === request.current) {
          setGroundTruth({ name: file.name, count: boxes.length })
          onBoxesChange(boxes)
        }
      } catch (err) { if (id === request.current) setError((err as Error).message) }
      finally { if (id === request.current) setLoading(false) }
    }} />
    {loading && <p className="text-xs">Đang đọc ground truth…</p>}
    {error && <p role="alert" className="text-xs text-red-500">{error}</p>}
    {groundTruth && <p className="text-sm">{groundTruth.name} — Số quả thực tế: <strong>{groundTruth.count}</strong></p>}
    {groundTruth && groundTruth.name.replace(/\.[^.]+$/, '') !== imageName.replace(/\.[^.]+$/, '') &&
      <p className="text-xs text-amber-600">Tên ground truth khác tên ảnh; hãy kiểm tra chúng có cùng ảnh hay không.</p>}
    {metrics && <div className="grid sm:grid-cols-3 gap-3 text-sm">
      <p>Dự đoán: <strong>{predicted!.toFixed(2)}</strong></p>
      <p>Sai số tuyệt đối: <strong>{metrics.absolute.toFixed(2)} quả</strong></p>
      <p>Sai số tương đối: <strong>{metrics.percentage === null ? 'Không xác định (GT = 0)' : `${metrics.percentage.toFixed(2)}%`}</strong></p>
    </div>}
    {metrics && <p className="text-xs text-muted-foreground">Bảng so sánh đầy đủ nằm ở đầu khung kết quả. Đây là sai số một ảnh, không phải mAP hay MAE toàn dataset. Số đếm cuối dùng cấu hình model; ngưỡng hiển thị chỉ lọc box.</p>}
  </section>
}

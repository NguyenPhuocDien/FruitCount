'use client'

import { useEffect, useState } from 'react'
import { ModelInfo, TaskType } from './types'

interface Entry { id: string; name: string; family?: string; task?: TaskType; classes?: string[]; ready: boolean; error?: string }

export function LocalModelPicker({ onSelect, selectedId = '' }: { onSelect: (model: ModelInfo) => void; selectedId?: string }) {
  const [models, setModels] = useState<Entry[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      const response = await fetch('/api/local-models', { cache: 'no-store' })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Không đọc được thư viện model.')
      setModels(data.models)
    } catch (err) { setError((err as Error).message) }
    finally { setLoading(false) }
  }
  useEffect(() => { void refresh() }, [])
  return <div className="space-y-3 rounded-lg border border-border p-3 mb-4">
    <div className="flex justify-between items-center gap-2">
      <label htmlFor="local-model" className="text-sm font-medium">Thư viện FruitCount trên máy chủ</label>
      <button type="button" disabled={loading} onClick={refresh} className="text-xs text-primary disabled:opacity-50">{loading ? 'Đang đọc…' : 'Làm mới'}</button>
    </div>
    <select id="local-model" value={selectedId} className="w-full rounded border border-border bg-background p-2 text-sm" onChange={event => {
      const m = models.find(item => item.id === event.target.value)
      if (!m?.ready || !m.task) return
      onSelect({ id: `local-${m.id}`, serverModelId: m.id, name: m.name, fileName: m.id,
        type: m.task, sizeMB: 0, inputShape: 'Theo cấu hình model', architecture: m.family || '',
        classesCount: m.classes?.length || 0, classes: m.classes, framework: 'Python server' })
    }}>
      <option value="">Chọn model đã cài…</option>
      {models.filter(m => m.id !== 'dino' && m.id !== 'resnet').map(m => <option key={m.id} value={m.id} disabled={!m.ready}>{m.name}{m.ready ? '' : ` — ${m.error}`}</option>)}
    </select>
    <p className="text-xs text-muted-foreground">Thêm thư mục model vào backend FruitCount rồi bấm Làm mới. Trạng thái sẵn sàng chỉ kiểm tra các file cấu hình, chưa kiểm thử trọng số.</p>
    {error && <p role="alert" className="text-xs text-red-500">{error}</p>}
  </div>
}

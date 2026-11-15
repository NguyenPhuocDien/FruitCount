import type { InferenceResult, InputImageMedia } from '@/components/playground/types'

export async function requestComparisonInference(modelId: string, image: InputImageMedia, signal: AbortSignal): Promise<InferenceResult> {
  const form = new FormData()
  form.append('modelId', modelId)
  if (image.appliedFilters?.length || !image.file) form.append('imageBase64', image.previewUrl)
  else form.append('imageFile', image.file)
  const response = await fetch('/api/inference', { method: 'POST', body: form, signal })
  const data = await response.json()
  if (!response.ok || !data.success) throw new Error(data.error || 'Model không trả kết quả hợp lệ.')
  if (!Array.isArray(data.detections)) throw new Error('Kết quả model thiếu danh sách detections.')
  return data
}

/** Sequential runs prevent two Python models competing for GPU memory; one failure does not skip the other. */
export async function runModelPair<T>(ids: [string, string], run: (id: string) => Promise<T>,
  onStart: (index: number) => void, onResult: (index: number, result: T | null, error: string | null) => void,
  signal: AbortSignal) {
  for (let index = 0; index < ids.length; index++) {
    if (signal.aborted) return
    onStart(index)
    try {
      const result = await run(ids[index])
      if (!signal.aborted) onResult(index, result, null)
    } catch (error) {
      if (signal.aborted) return
      onResult(index, null, (error as Error).message || 'Không chạy được model.')
    }
  }
}

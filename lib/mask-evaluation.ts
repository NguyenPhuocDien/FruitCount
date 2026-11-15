import type { GroundTruthBox } from './count-evaluation'

export type MaskMode = 'instance' | 'binary'

/** Black is background. Instance IDs are exact RGB values, never brightness thresholds. */
export function maskToBoxes(pixels: Uint8ClampedArray, width: number, height: number, mode: MaskMode): GroundTruthBox[] {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 || pixels.length !== width * height * 4) {
    throw new Error('Kích thước mask không hợp lệ.')
  }
  const keyAt = (index: number) => {
    const i = index * 4
    return pixels[i + 3] === 0 ? 0 : (pixels[i] << 16) | (pixels[i + 1] << 8) | pixels[i + 2]
  }
  type Region = { id: number; x0: number; y0: number; x1: number; y1: number }
  const regions: Region[] = []
  if (mode === 'instance') {
    const byId = new Map<number, Region>()
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const id = keyAt(y * width + x)
      if (!id) continue
      let region = byId.get(id)
      if (!region) {
        if (byId.size >= 10000) throw new Error('Mask có quá nhiều ID; hãy dùng mask gốc không anti-alias, không phải ảnh màu/heatmap.')
        region = { id, x0: x, y0: y, x1: x, y1: y }
        byId.set(id, region)
      } else {
        region.x0 = Math.min(region.x0, x); region.y0 = Math.min(region.y0, y)
        region.x1 = Math.max(region.x1, x); region.y1 = Math.max(region.y1, y)
      }
    }
    regions.push(...byId.values())
  } else {
    const visited = new Uint8Array(width * height)
    const queue = new Int32Array(width * height)
    for (let start = 0; start < visited.length; start++) {
      if (visited[start] || !keyAt(start)) continue
      const region = { id: regions.length + 1, x0: width, y0: height, x1: 0, y1: 0 }
      let head = 0, tail = 1
      queue[0] = start; visited[start] = 1
      while (head < tail) {
        const index = queue[head++], x = index % width, y = Math.floor(index / width)
        region.x0 = Math.min(region.x0, x); region.x1 = Math.max(region.x1, x)
        region.y0 = Math.min(region.y0, y); region.y1 = Math.max(region.y1, y)
        // Eight-connected foreground; diagonal neighbours belong to the same region.
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue
          const next = ny * width + nx
          if (!visited[next] && keyAt(next)) { visited[next] = 1; queue[tail++] = next }
        }
      }
      regions.push(region)
      if (regions.length > 10000) throw new Error('Mask có quá nhiều vùng; kiểm tra nhiễu trong file gốc.')
    }
  }
  return regions.map(r => ({ classId: 0, x: r.x0 / width, y: r.y0 / height,
    width: (r.x1 - r.x0 + 1) / width, height: (r.y1 - r.y0 + 1) / height }))
}

export async function readMask(file: File, imageUrl: string, mode: MaskMode) {
  if (file.size > 25 * 1024 * 1024) throw new Error('File mask tối đa 25 MB.')
  const header = new Uint8Array(await file.slice(0, 26).arrayBuffer())
  if (header.length < 26 || [137, 80, 78, 71, 13, 10, 26, 10].some((v, i) => header[i] !== v)) {
    throw new Error('Hãy chọn mask PNG gốc; không dùng JPEG vì mất ID pixel.')
  }
  if (header[24] === 16) throw new Error('Mask PNG 16-bit chưa được hỗ trợ; cần bộ đọc giữ nguyên ID 16-bit, không đổi trực tiếp sang 8-bit.')
  const dimensions = (offset: number) => new DataView(header.buffer).getUint32(offset)
  if (dimensions(16) * dimensions(20) > 16_000_000) throw new Error('Mask tối đa 16 triệu pixel.')
  const load = (url: string) => new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Không đọc được ảnh hoặc mask.'))
    image.src = url
  })
  const url = URL.createObjectURL(file)
  try {
    const [mask, image] = await Promise.all([load(url), load(imageUrl)])
    if (mask.naturalWidth !== image.naturalWidth || mask.naturalHeight !== image.naturalHeight) {
      throw new Error(`Mask ${mask.naturalWidth}×${mask.naturalHeight} không khớp ảnh ${image.naturalWidth}×${image.naturalHeight}; cần mask đúng ảnh, không tự resize ID.`)
    }
    const canvas = document.createElement('canvas')
    canvas.width = mask.naturalWidth; canvas.height = mask.naturalHeight
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new Error('Không mở được canvas để đọc mask.')
    ctx.drawImage(mask, 0, 0)
    return maskToBoxes(ctx.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height, mode)
  } finally { URL.revokeObjectURL(url) }
}

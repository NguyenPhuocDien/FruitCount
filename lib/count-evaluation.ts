/** YOLO ground truth: one class xc yc width height row per annotated fruit. */
export function parseYoloGroundTruth(text: string): number {
  return parseYoloBoxes(text).length
}

export interface GroundTruthBox { classId: number; x: number; y: number; width: number; height: number }

export function parseYoloBoxes(text: string): GroundTruthBox[] {
  const rows = text.replace(/^\uFEFF/, '').split(/\r?\n/)
  const boxes: GroundTruthBox[] = []
  rows.forEach((row, index) => {
    if (!row.trim()) return
    const tokens = row.trim().split(/\s+/)
    const values = tokens.map(Number)
    if (tokens.length !== 5 || values.some(v => !Number.isFinite(v)) ||
      !Number.isInteger(values[0]) || values[0] < 0 ||
      values.slice(1).some(v => v < 0 || v > 1) || values[3] <= 0 || values[4] <= 0) {
      throw new Error(`Dòng ${index + 1}: cần định dạng YOLO class xc yc width height, tọa độ chuẩn hóa 0–1.`)
    }
    boxes.push({ classId: values[0], x: values[1] - values[3] / 2,
      y: values[2] - values[4] / 2, width: values[3], height: values[4] })
  })
  return boxes
}

/** Maximum relative density inside each annotated box, not detection confidence. */
export function densityBoxResponses(boxes: GroundTruthBox[], pixels: Uint8ClampedArray, width: number, height: number) {
  return boxes.map(box => {
    const x0 = Math.max(0, Math.floor(box.x * width)), y0 = Math.max(0, Math.floor(box.y * height))
    const x1 = Math.min(width, Math.ceil((box.x + box.width) * width))
    const y1 = Math.min(height, Math.ceil((box.y + box.height) * height))
    let peak = 0
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) peak = Math.max(peak, pixels[(y * width + x) * 4])
    return peak / 255
  })
}

export function countErrors(predicted: number, actual: number) {
  const signed = predicted - actual
  return { signed, absolute: Math.abs(signed), squared: signed ** 2,
    percentage: actual > 0 ? Math.abs(signed) / actual * 100 : null }
}

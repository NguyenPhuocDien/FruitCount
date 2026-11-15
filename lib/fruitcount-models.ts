import fs from 'fs/promises'
import path from 'path'

const DEFAULT_MODEL_ROOT = path.join(/*turbopackIgnore: true*/ process.cwd(), 'backend FruitCount')
export const MODEL_ROOT = process.env.FRUITCOUNT_MODEL_ROOT ? path.resolve(process.env.FRUITCOUNT_MODEL_ROOT) : DEFAULT_MODEL_ROOT

export async function getLocalModel(id: string) {
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error('Model ID không hợp lệ.')
  const directory = path.join(MODEL_ROOT, id)
  const root = await fs.realpath(MODEL_ROOT)
  const realDirectory = await fs.realpath(directory)
  if (path.dirname(realDirectory) !== root) throw new Error('Thư mục model nằm ngoài thư viện.')
  const manifestPath = path.join(realDirectory, 'model.json')
  const config = JSON.parse(await fs.readFile(manifestPath, 'utf8'))
  if (!['yolo', 'resnet', 'csrnet', 'dino', 'deimv2', 'ecdet'].includes(config.family)) throw new Error('Loại model không hợp lệ.')
  if (!['detection', 'classification', 'segmentation'].includes(config.task)) throw new Error('Task không hợp lệ.')
  if (config.requiredFiles !== undefined && !Array.isArray(config.requiredFiles)) throw new Error('requiredFiles phải là danh sách file.')
  const required = [config.weights, config.adapter, config.deployConfig, config.labelsFile, ...(config.requiredFiles || [])].filter(Boolean)
  const missing: string[] = []
  for (const file of required) {
    if (typeof file !== 'string' || path.isAbsolute(file)) throw new Error('Đường dẫn model phải tương đối.')
    const target = path.resolve(realDirectory, file)
    if (!target.startsWith(realDirectory + path.sep)) throw new Error('File phải nằm trong thư mục model.')
    try {
      const real = await fs.realpath(target)
      if (!real.startsWith(realDirectory + path.sep)) throw new Error('File trỏ ra ngoài thư mục model.')
      if (!(await fs.stat(real)).isFile()) missing.push(file)
    } catch { missing.push(file) }
  }
  if (!config.weights) missing.push('weights trong model.json')
  if (['csrnet', 'dino', 'deimv2', 'ecdet'].includes(config.family) && !config.adapter) missing.push('adapter trong model.json')
  return { id, config, manifestPath, directory: realDirectory, missing }
}

export async function listLocalModels() {
  const entries = await fs.readdir(MODEL_ROOT, { withFileTypes: true })
  return Promise.all(entries.filter(e => e.isDirectory() && /^[a-zA-Z0-9_-]+$/.test(e.name)).map(async e => {
    try {
      const m = await getLocalModel(e.name)
      return { id: m.id, name: m.config.name || m.id, family: m.config.family, task: m.config.task,
        classes: m.config.classes || [], ready: m.missing.length === 0, error: m.missing.length ? `Thiếu: ${m.missing.join(', ')}` : undefined }
    } catch (error) {
      return { id: e.name, name: e.name, ready: false, error: (error as Error).message }
    }
  }))
}

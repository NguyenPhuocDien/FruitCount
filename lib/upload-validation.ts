export class InvalidUpload extends Error {}

export function validateUploadForm(form: FormData) {
  for (const field of ['modelFile', 'imageFile', 'labelsFile']) {
    const entries = form.getAll(field)
    if (entries.length > 1 || entries.some(value => typeof value === 'string')) {
      throw new InvalidUpload(`${field} phải là một file tải lên.`)
    }
  }
  for (const field of ['modelId', 'imageBase64', 'confThreshold', 'iouThreshold', 'imgsz', 'norm', 'taskType']) {
    const entries = form.getAll(field)
    if (entries.length > 1 || entries.some(value => typeof value !== 'string')) {
      throw new InvalidUpload(`${field} phải là chuỗi văn bản.`)
    }
  }
}

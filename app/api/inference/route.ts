import { InvalidUpload, validateUploadForm } from '@/lib/upload-validation'
import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import { ensureTempDir, newRunId, runModelScript, safeUnlink, saveUpload } from '@/lib/python-runner'
import { getLocalModel } from '@/lib/fruitcount-models'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

export async function POST(req: NextRequest) {
  let modelPath: string | null = null
  let imagePath: string | null = null
  let labelsPath: string | null = null
  try {
    let form: FormData
    try { form = await req.formData() }
    catch { throw new InvalidUpload('Không đọc được dữ liệu multipart của request.') }
    validateUploadForm(form)
    const modelFile = form.get('modelFile') as File | null
    const imageFile = form.get('imageFile') as File | null
    const imageBase64 = form.get('imageBase64') as string | null
    const labelsFile = form.get('labelsFile') as File | null
    const modelId = form.get('modelId')
    let localModel = null
    if (typeof modelId === 'string' && modelId) {
      try {
        localModel = await getLocalModel(modelId)
        if (localModel.missing.length) throw new Error(`Thiếu: ${localModel.missing.join(', ')}`)
      } catch (error) {
        return NextResponse.json({ success: false, error: (error as Error).message }, { status: 400 })
      }
    }

    if (!localModel && (!modelFile || modelFile.size === 0)) {
      return NextResponse.json(
        { success: false, error: 'Chưa có file mô hình được tải lên. Vui lòng tải file mô hình (.pt, .onnx, ...) từ máy của bạn.' },
        { status: 400 }
      )
    }

    const runId = newRunId('infer')
    if (!localModel) modelPath = await saveUpload(modelFile!, runId, 'model')

    if (imageFile && imageFile.size > 0) {
      imagePath = await saveUpload(imageFile, runId, 'input')
    } else if (imageBase64 && imageBase64.includes('base64,')) {
      const mime = imageBase64.slice(5, imageBase64.indexOf(';'))
      const ext = mime.includes('png') ? '.png' : mime.includes('webp') ? '.webp' : '.jpg'
      imagePath = path.join(ensureTempDir(), `input-${runId}${ext}`)
      fs.writeFileSync(imagePath, Buffer.from(imageBase64.split('base64,')[1], 'base64'))
    } else {
      return NextResponse.json({ success: false, error: 'Chưa có hình ảnh đầu vào để kiểm thử.' }, { status: 400 })
    }

    const args = [
      '--model', localModel ? path.join(localModel.directory, localModel.config.weights) : modelPath!,
      '--image', imagePath,
      '--name', localModel ? localModel.config.name || localModel.id : modelFile!.name,
      '--conf', String(form.get('confThreshold') ?? '0.25'),
      '--iou', String(form.get('iouThreshold') ?? '0.45'),
    ]
    if (localModel) args.push('--manifest', localModel.manifestPath)
    const imgsz = form.get('imgsz')
    const norm = form.get('norm')
    if (imgsz && Number(imgsz) > 0) args.push('--imgsz', String(Number(imgsz)))
    if (norm) args.push('--norm', String(norm))
    if (labelsFile && labelsFile.size > 0) {
      labelsPath = await saveUpload(labelsFile, runId, 'labels')
      args.push('--labels', labelsPath)
    }

    const r = await runModelScript(args, 300_000, req.signal)
    if (req.signal.aborted) return new NextResponse(null, { status: 499 })
    if (!r.ok) return NextResponse.json({ success: false, error: r.error }, { status: 500 })
    return NextResponse.json(r.data, { status: r.data?.success ? 200 : 422 })
  } catch (err: any) {
    if (err instanceof InvalidUpload) return NextResponse.json({ success: false, error: err.message }, { status: 400 })
    return NextResponse.json({ success: false, error: err.message || 'Lỗi server khi xử lý inference.' }, { status: 500 })
  } finally {
    safeUnlink(modelPath, imagePath, labelsPath)
  }
}

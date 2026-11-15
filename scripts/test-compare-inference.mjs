import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../lib/compare-inference.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext } }).outputText
const { runModelPair, requestComparisonInference } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
const events = []
await runModelPair(['a', 'b'], async id => {
  events.push(`run-${id}`)
  if (id === 'a') throw new Error('checkpoint failed')
  return { id }
}, index => events.push(`start-${index}`), (index, result, error) => events.push([index, result, error]), new AbortController().signal)
assert.deepEqual(events, ['start-0', 'run-a', [0, null, 'checkpoint failed'], 'start-1', 'run-b', [1, { id: 'b' }, null]])
const cancelled = new AbortController()
const calls = []
await runModelPair(['a', 'b'], async id => { calls.push(id); cancelled.abort(); return id }, () => {}, () => assert.fail('Must not publish cancelled result'), cancelled.signal)
assert.deepEqual(calls, ['a'])
const originalFetch = globalThis.fetch
const requests = []
try {
  globalThis.fetch = async (url, options) => {
    requests.push({ url, model: options.body.get('modelId'), file: options.body.get('imageFile'), base64: options.body.get('imageBase64') })
    return { ok: true, json: async () => ({ success: true, detections: [], predicted_count: 12 }) }
  }
  const file = new File(['image-data'], 'fruit.png', { type: 'image/png' })
  const image = { file, previewUrl: 'data:image/png;base64,dGVzdA==', fileName: 'fruit.png' }
  await runModelPair(['a', 'b'], id => requestComparisonInference(id, image, new AbortController().signal), () => {}, () => {}, new AbortController().signal)
  assert.deepEqual(requests.map(r => r.model), ['a', 'b'])
  assert.equal(await requests[0].file.text(), await requests[1].file.text())
  assert.ok(requests.every(r => r.url === '/api/inference' && r.base64 === null))
  await requestComparisonInference('a', { ...image, appliedFilters: ['brightness'] }, new AbortController().signal)
  assert.equal(requests[2].file, null)
  assert.equal(requests[2].base64, image.previewUrl)
  globalThis.fetch = async () => ({ ok: false, json: async () => ({ error: 'Missing model' }) })
  await assert.rejects(requestComparisonInference('a', image, new AbortController().signal), /Missing model/)
} finally { globalThis.fetch = originalFetch }
console.log('Comparison tests passed: sequential requests, independent failures, cancellation, identical input and filtered image payload.')

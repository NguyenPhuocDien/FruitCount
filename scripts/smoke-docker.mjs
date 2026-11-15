// Run on the host after docker compose up; no Python or extra npm dependencies required.
const base = new URL(process.argv[2] || 'http://127.0.0.1:3000')
async function request(path, options = {}) {
  return fetch(new URL(path, base), { ...options, signal: AbortSignal.timeout(15000) })
}
const health = await request('/api/health')
if (!health.ok || (await health.json()).status !== 'ok') throw new Error('Healthcheck failed')
console.log('Healthcheck OK')
const listing = await request('/api/local-models')
if (!listing.ok) throw new Error(`Model library returned ${listing.status}`)
const { models } = await listing.json()
if (!Array.isArray(models)) throw new Error('Invalid model library response')
console.log('Ready models:', models.filter(model => model.ready).map(model => model.id).join(', ') || '(none; install model files)')
for (const route of ['/playground', '/compare']) {
  const response = await request(route)
  if (!response.ok) throw new Error(`${route}: ${response.status}`)
  const html = await response.text()
  const assets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:css|js)(?:\?[^"]*)?)"/g)]
    .map(match => match[1]).filter(path => path.startsWith('/_next/'))
  if (!assets.length) throw new Error(`${route}: no built static assets found`)
  for (const path of new Set(assets)) {
    const asset = await request(path)
    if (!asset.ok) throw new Error(`${path}: ${asset.status}`)
  }
  console.log(`${route}: HTML and ${assets.length} assets OK`)
}
const icon = await request('/icon.svg')
if (!icon.ok) throw new Error('Public assets not served')
const invalid = await request('/api/inference', { method: 'POST', body: new FormData() })
if (invalid.status !== 400) throw new Error('Inference validation did not return 400 for missing input')
console.log('Public assets and inference input validation OK (no actual model execution).')

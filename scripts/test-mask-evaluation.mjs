import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'
import sharp from 'sharp'

const source = fs.readFileSync(new URL('../lib/mask-evaluation.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } }).outputText
const { maskToBoxes } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
function rgba(values) {
  return new Uint8ClampedArray(values.flatMap(v => [v, v, v, 255]))
}
assert.equal(maskToBoxes(rgba([1, 0, 1, 0]), 2, 2, 'instance').length, 1)
assert.equal(maskToBoxes(rgba([1, 0, 0, 2]), 2, 2, 'instance').length, 2)
assert.equal(maskToBoxes(rgba([1, 0, 0, 2]), 2, 2, 'binary').length, 1)
assert.equal(maskToBoxes(rgba([1, 0, 1]), 3, 1, 'binary').length, 2)
assert.equal(maskToBoxes(rgba([0, 0]), 2, 1, 'instance').length, 0)
assert.deepEqual(maskToBoxes(rgba([0, 1, 0, 0]), 2, 2, 'instance')[0],
  { classId: 0, x: 0.5, y: 0, width: 0.5, height: 0.5 })
assert.throws(() => maskToBoxes(rgba([1]), 2, 2, 'instance'))
const maskPath = new URL('../docs/mask1.png', import.meta.url)
if (fs.existsSync(maskPath)) {
  const { data, info } = await sharp(fs.readFileSync(maskPath)).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const pixels = new Uint8ClampedArray(data)
  const instances = maskToBoxes(pixels, info.width, info.height, 'instance')
  assert.equal(instances.length, 95)
  console.log(`docs/mask1.png: ${info.width}x${info.height}; ${instances.length} instance IDs; ${maskToBoxes(pixels, info.width, info.height, 'binary').length} binary components.`)
}
console.log('Mask tests passed: instance IDs, touching/disconnected regions, empty mask and bounding boxes.')

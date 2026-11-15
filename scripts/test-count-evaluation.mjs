import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../lib/count-evaluation.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const { parseYoloGroundTruth, parseYoloBoxes, densityBoxResponses, countErrors } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
assert.equal(parseYoloGroundTruth('\uFEFF0 0.5 0.6 0.1 0.2\r\n1 0.2 0.3 0.1 0.1\r\n'), 2)
assert.equal(parseYoloGroundTruth(' \r\n'), 0)
for (const invalid of ['fruit', '0 0.5 0.5 0.1', '0 NaN 0.5 0.1 0.2', '0 1.2 0.5 0.1 0.2', '0 0.5 0.5 0 0.2', '-1 0.5 0.5 0.1 0.2']) {
  assert.throws(() => parseYoloGroundTruth(invalid))
}
assert.deepEqual(countErrors(12.5, 10), { signed: 2.5, absolute: 2.5, squared: 6.25, percentage: 25 })
assert.equal(countErrors(2, 0).percentage, null)
const pixels = new Uint8ClampedArray(4 * 4 * 4)
pixels[(1 * 4 + 1) * 4] = 255
const boxes = parseYoloBoxes('0 0.25 0.25 0.5 0.5\n0 0.75 0.75 0.5 0.5')
assert.deepEqual(densityBoxResponses(boxes, pixels, 4, 4), [1, 0])
assert.deepEqual(densityBoxResponses([], pixels, 4, 4), [])
assert.deepEqual(densityBoxResponses(parseYoloBoxes('0 0 0 0.1 0.1'), pixels, 4, 4), [0])
console.log('Count evaluation: valid/invalid YOLO labels, fractional predictions and zero ground truth passed.')
console.log('Density comparison: strong/zero response, empty annotations and image boundary passed.')

import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { printLanUrls } from './lan-addresses.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
// Shell environment takes precedence over .env: this command explicitly requests LAN access.
const env = { ...process.env, FRUITCOUNT_BIND_ADDRESS: '0.0.0.0' }
const args = ['compose', 'up', '-d', '--wait', '--wait-timeout', '180']
args.push(process.argv.includes('--no-build') ? '--no-build' : '--build')
const started = spawnSync('docker', args, { cwd: root, env, stdio: 'inherit' })
if (started.error || started.status !== 0) {
  console.error(started.error?.message || 'Triển khai chưa thành công. Kiểm tra Docker Desktop, log build hoặc cổng đang bị chiếm.')
  process.exit(1)
}
const published = spawnSync('docker', ['compose', 'port', 'web', '3000'], { cwd: root, env, encoding: 'utf8' })
const port = published.stdout?.trim().match(/:(\d+)\s*$/)?.[1]
if (published.status !== 0 || !port) {
  console.error('Container đã khởi động nhưng chưa đọc được cổng. Chạy docker compose ps.')
  process.exit(1)
}
printLanUrls(port)
if (process.platform === 'win32') console.log(`Nếu thiết bị khác chưa truy cập được, chạy PowerShell Administrator:\npowershell -ExecutionPolicy Bypass -File scripts/allow-lan.ps1 -Port ${port}`)

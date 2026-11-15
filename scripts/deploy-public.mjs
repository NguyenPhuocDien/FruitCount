import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const compose = ['compose', '-f', 'compose.yaml', '-f', 'compose.tunnel.yaml']
const options = { cwd: root, encoding: 'utf8' }
const up = spawnSync('docker', [...compose, 'up', '-d', '--no-build'], { cwd: root, stdio: 'inherit' })
if (up.error || up.status !== 0) {
  console.error(up.error?.message || 'Không khởi động được web/tunnel. Build image web trước bằng npm run deploy:lan.')
  process.exit(1)
}
for (let attempt = 0; attempt < 20; attempt++) {
  const logs = spawnSync('docker', [...compose, 'logs', '--no-color', '--tail', '150', 'tunnel'], options)
  const matches = (logs.stdout + logs.stderr).match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/g)
  if (matches?.length) {
    const url = matches[matches.length - 1]
    console.log(`\nLink thử từ Internet: ${url}/playground\nSo sánh model: ${url}/compare`)
    console.log('Đây là link demo công khai tạm thời. Giữ máy chủ, Docker và tunnel hoạt động; URL có thể đổi khi tunnel khởi động lại.')
    console.log('Dừng chia sẻ: docker compose -f compose.yaml -f compose.tunnel.yaml stop tunnel')
    process.exit(0)
  }
  await new Promise(resolve => setTimeout(resolve, 3000))
}
console.error('Chưa nhận được URL. Xem log: docker compose -f compose.yaml -f compose.tunnel.yaml logs tunnel')
process.exit(1)

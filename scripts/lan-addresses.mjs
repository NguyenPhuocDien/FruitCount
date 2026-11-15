import os from 'node:os'

export function lanAddresses(interfaces = os.networkInterfaces()) {
  return [...new Set(Object.entries(interfaces)
    .filter(([name]) => !/vEthernet|WSL|VirtualBox|VMware|Docker|Loopback/i.test(name))
    .flatMap(([, entries]) => (entries || []).filter(entry => !entry.internal && entry.family === 'IPv4').map(entry => entry.address)))]
}

export function printLanUrls(port) {
  console.log(`\nMáy hiện tại: http://localhost:${port}/playground`)
  const addresses = lanAddresses()
  for (const address of addresses) {
    console.log(`Thiết bị cùng mạng: http://${address}:${port}/playground`)
    console.log(`So sánh model:       http://${address}:${port}/compare`)
  }
  if (!addresses.length) console.log('Không tìm thấy IPv4 LAN. Kiểm tra kết nối Wi-Fi/Ethernet và chạy ipconfig.')
  console.log('Giữ máy chủ và Docker hoạt động. IP LAN có thể thay đổi khi đổi mạng.\n')
}

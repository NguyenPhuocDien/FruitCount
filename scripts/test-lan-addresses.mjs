import assert from 'node:assert/strict'
import { lanAddresses } from './lan-addresses.mjs'

const entry = (address, internal = false, family = 'IPv4') => ({ address, internal, family })
assert.deepEqual(lanAddresses({
  'Wi-Fi': [entry('192.168.1.15'), entry('fe80::1', false, 'IPv6')],
  Ethernet: [entry('192.168.1.15'), entry('10.0.0.8')],
  Loopback: [entry('127.0.0.1', true)],
  'vEthernet (WSL)': [entry('172.20.0.1')],
  'VirtualBox Host-Only': [entry('192.168.56.1')],
}), ['192.168.1.15', '10.0.0.8'])
assert.deepEqual(lanAddresses({}), [])
console.log('LAN address tests passed: physical IPv4 candidates, duplicate removal, loopback/IPv6/virtual adapter exclusions.')

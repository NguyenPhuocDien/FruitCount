param(
    [ValidateRange(1, 65535)][int]$Port = 3000,
    [ValidateSet('Private', 'Public')][string]$NetworkProfile = 'Private',
    [string]$LocalAddress = 'Any'
)
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Mo PowerShell bang Run as Administrator de them quy tac Firewall.'
}
$ruleName = "FruitCount-LAN-TCP-$Port"
if ($NetworkProfile -eq 'Public') {
    if ($LocalAddress -eq 'Any') { throw 'Public profile requires the exact local Wi-Fi IPv4 address via -LocalAddress.' }
    $parsedAddress = [System.Net.IPAddress]::Parse($LocalAddress)
    if ($parsedAddress.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) { throw 'LocalAddress must be IPv4.' }
    $ruleName += '-Public'
}
if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule -Name $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -Profile $NetworkProfile -RemoteAddress LocalSubnet -LocalAddress $LocalAddress | Out-Null
    Enable-NetFirewallRule -Name $ruleName | Out-Null
} else {
    New-NetFirewallRule -Name $ruleName -DisplayName "FruitCount LAN TCP $Port ($NetworkProfile)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -Profile $NetworkProfile -RemoteAddress LocalSubnet -LocalAddress $LocalAddress | Out-Null
}
Write-Output "Da cho phep TCP $Port tu cung subnet tren mang $NetworkProfile, IP $LocalAddress."
Write-Output 'Quy tac khong mo cong cho Internet va khong thay doi profile cua mang.'

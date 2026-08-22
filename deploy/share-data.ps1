$share = "plant-data"
$path = "D:\dev\plant-processing-image\data"
$user = "KINGSMAN\perez"
if (-not (Test-Path $path)) {
    throw "Missing $path"
}
$existing = Get-SmbShare -Name $share -ErrorAction SilentlyContinue
if ($existing) {
    Set-SmbShare -Name $share -Description "Plant health photos (read-only)" -Force
    Grant-SmbShareAccess -Name $share -AccountName $user -AccessRight Read -Force
    Revoke-SmbShareAccess -Name $share -AccountName "Everyone" -Force -ErrorAction SilentlyContinue
} else {
    New-SmbShare -Name $share -Path $path -ReadAccess $user -Description "Plant health photos (read-only)"
}
Get-SmbShare -Name $share | Format-List Name, Path, Description
Get-SmbShareAccess -Name $share | Format-Table AccountName, AccessControlType, AccessRight

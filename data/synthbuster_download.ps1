# Script to download Synthbuster dataset.
# Synthbuster dataset can be used only for non-commercial purposes.
# It is obtained from the RAISE dataset available on the web page: http://mmlab.science.unitn.it/RAISE/
# To use this dataset, cite these two papers:
# [1] Quentin Bammey. 
#    "Synthbuster: Towards detection of diffusion model generated images." IEEE OJSP, 2023.
# [2] Duc-Tien Dang-Nguyen, Cecilia Pasquini, Valentina Conotter, and Giulia Boato.
#    "RAISE: A Raw Images Dataset for Digital Image Forensics." In ACM MMSys, page 219–224, 2015.
#

Write-Host "Downloading Synthbuster dataset..."

# Download files using Invoke-WebRequest
if (-not (Test-Path "synthbuster.zip")) {
    Write-Host "Downloading synthbuster.zip..."
    Invoke-WebRequest -Uri "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1" -OutFile "synthbuster.zip"
} else {
    Write-Host "synthbuster.zip already exists, skipping download."
}

if (-not (Test-Path "real_RAISE_1k.zip")) {
    Write-Host "Downloading real_RAISE_1k.zip..."
    Invoke-WebRequest -Uri "https://www.grip.unina.it/download/prog/DMimageDetection/real_RAISE_1k.zip" -OutFile "real_RAISE_1k.zip"
} else {
    Write-Host "real_RAISE_1k.zip already exists, skipping download."
}

Write-Host "Verifying checksums..."
# Verify MD5 checksums (requires Get-FileHash which outputs SHA256 by default in PowerShell)
# For MD5 verification, we'll use a custom function
function Verify-MD5 {
    param($checksumFile)
    
    if (Test-Path $checksumFile) {
        $checksums = Get-Content $checksumFile
        foreach ($line in $checksums) {
            if ($line -match '(\S+)\s+(.+)') {
                $expectedHash = $matches[1]
                $filename = $matches[2] -replace '\*', ''
                
                if (Test-Path $filename) {
                    $actualHash = (Get-FileHash -Path $filename -Algorithm MD5).Hash.ToLower()
                    if ($actualHash -eq $expectedHash) {
                        Write-Host "[OK] $filename" -ForegroundColor Green
                    } else {
                        Write-Host "[FAILED] $filename" -ForegroundColor Red
                        Write-Host "  Expected: $expectedHash" -ForegroundColor Red
                        Write-Host "  Got:      $actualHash" -ForegroundColor Red
                    }
                } else {
                    Write-Host "[NOT FOUND] $filename" -ForegroundColor Yellow
                }
            }
        }
    } else {
        Write-Host "Checksum file not found, skipping verification..." -ForegroundColor Yellow
    }
}

Verify-MD5 "synthbuster_checksums.md5"

Write-Host "Unzipping files..."
# Expand archives (PowerShell 5.0+)
if (Test-Path "synthbuster.zip") {
    Write-Host "Extracting synthbuster.zip..."
    Expand-Archive -Path "synthbuster.zip" -DestinationPath "." -Force
} else {
    Write-Host "synthbuster.zip not found, skipping extraction." -ForegroundColor Yellow
}

if (Test-Path "real_RAISE_1k.zip") {
    Write-Host "Extracting real_RAISE_1k.zip..."
    Expand-Archive -Path "real_RAISE_1k.zip" -DestinationPath "synthbuster" -Force
} else {
    Write-Host "real_RAISE_1k.zip not found, skipping extraction." -ForegroundColor Yellow
}

Write-Host "Done." -ForegroundColor Green

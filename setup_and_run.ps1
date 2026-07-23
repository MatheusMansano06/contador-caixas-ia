# Script de Setup e Inicialização - Contador de Caixas IA

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Contador de Caixas IA - Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# 1. Criar ambiente virtual se não existir
if (-not (Test-Path ".venv")) {
    Write-Host "`n[1/4] Criando ambiente virtual..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "`n[1/4] Ambiente virtual já existe." -ForegroundColor Green
}

# 2. Ativar ambiente
Write-Host "[2/4] Ativando ambiente virtual..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# 3. Instalar dependências
Write-Host "[3/4] Instalando dependências..." -ForegroundColor Yellow
pip install -q -r 04_backend_api_contagem/requirements.txt

# 4. Criar diretório de dados
if (-not (Test-Path "06_dados_e_sessoes")) {
    New-Item -ItemType Directory -Path "06_dados_e_sessoes" | Out-Null
}

Write-Host "`n=====================================" -ForegroundColor Green
Write-Host "Setup concluído! Iniciando API..." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

Write-Host "`nAPI rodando em: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Painel web (Desktop): http://localhost:8000" -ForegroundColor Cyan
Write-Host "Painel mobile: http://localhost:8000/index-mobile.html" -ForegroundColor Cyan
Write-Host "`nNota: Para usar no celular, configure um tunnel HTTPS (ngrok, localtunnel, etc)" -ForegroundColor Yellow
Write-Host "`nPressione Ctrl+C para parar o servidor`n" -ForegroundColor Gray

# Rodar servidor
Set-Location 04_backend_api_contagem
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

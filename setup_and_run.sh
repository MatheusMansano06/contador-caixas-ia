#!/bin/bash

echo "====================================="
echo "Contador de Caixas IA - Setup"
echo "====================================="

# 1. Criar ambiente virtual se não existir
if [ ! -d ".venv" ]; then
    echo -e "\n[1/4] Criando ambiente virtual..."
    python3 -m venv .venv
else
    echo -e "\n[1/4] Ambiente virtual já existe."
fi

# 2. Ativar ambiente
echo "[2/4] Ativando ambiente virtual..."
source .venv/bin/activate

# 3. Instalar dependências
echo "[3/4] Instalando dependências..."
pip install -q -r 04_backend_api_contagem/requirements.txt

# 4. Criar diretório de dados
mkdir -p 06_dados_e_sessoes

echo -e "\n====================================="
echo "Setup concluído! Iniciando API..."
echo "====================================="

echo -e "\nAPI rodando em: http://localhost:8000"
echo "Painel web (Desktop): http://localhost:8000"
echo "Painel mobile: http://localhost:8000/index-mobile.html"
echo -e "\nNota: Para usar no celular, configure um tunnel HTTPS (ngrok, localtunnel, etc)"
echo -e "\nPressione Ctrl+C para parar o servidor\n"

# Rodar servidor
cd 04_backend_api_contagem
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

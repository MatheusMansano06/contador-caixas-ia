# 🚀 Quick Start - Contador de Caixas IA

## Setup Automático (Windows)

```powershell
.\setup_and_run.ps1
```

**Ou manual (qualquer SO):**

```bash
# 1. Criar ambiente virtual
python -m venv .venv

# 2. Ativar
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instalar dependências
pip install -r 04_backend_api_contagem/requirements.txt

# 4. Rodar servidor
cd 04_backend_api_contagem
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🎯 Usando no Celular

**Opção 1: Rede Local (Rápido)**
1. Descubra o IP da sua máquina: `ipconfig` (Windows) ou `ifconfig` (Linux/Mac)
2. No celular (mesma rede), acesse: `http://SEU_IP:8000/index-mobile.html`
3. Clique em "Iniciar" para capturar câmera

**Opção 2: Tunnel HTTPS (Necessário se quiser getUserMedia)**
```bash
# Instale ngrok: https://ngrok.com
ngrok http 8000
# Copie a URL HTTPS gerada
# No celular, acesse a URL HTTPS do ngrok + /index-mobile.html
```

## 🎬 Testando Localmente (Desktop)

```
http://localhost:8000         # Painel desktop clássico (webcam servidor)
http://localhost:8000/index-mobile.html  # Painel mobile (câmera do cliente)
```

## 📌 Endpoints da API

- `POST /api/start` - Inicia sessão
- `POST /api/stop` - Para sessão
- `POST /api/frame` - Processa frame da câmera do cliente
- `POST /api/line` - Atualiza linha virtual
- `GET /api/status` - Status atual
- `GET /api/sessions` - Histórico de sessões
- `GET /api/events/{session_id}` - Eventos de uma sessão

## 💾 Banco de Dados

Salvo automaticamente em:
```
06_dados_e_sessoes/contador_caixas.sqlite3
```

## 🔧 Troubleshooting

**"Nenhuma permissão de câmera"**
- No celular: permitir acesso à câmera no prompt do navegador
- Usar HTTPS (ngrok ou similar) para userMedia funcionar

**"Frames muito lentos"**
- Reduzir resolução da câmera
- Reduzir qualidade JPEG (alterar app.js)
- Usar conexão WiFi de 5GHz

**"Objetos não detectados"**
- Melhorar iluminação do ambiente
- Ajustar a linha virtual nos controles
- Próximo: treinar modelo YOLO com dataset próprio

## 📚 Próximos Passos

1. ✅ Teste ao vivo com caixas reais
2. 📊 Colete vídeos para treinar modelo YOLO customizado
3. 🤖 Implemente detector YOLO melhorado
4. 📱 Deploy em edge box ou Jetson

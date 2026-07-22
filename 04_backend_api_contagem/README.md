# Backend API de Contagem

Backend local do MVP ao vivo do contador de caixas.

Esta primeira versao entrega:

- API FastAPI;
- painel web servido pelo proprio backend;
- captura ao vivo via webcam USB ou camera IP/RTSP;
- detector local inicial baseado em movimento e contornos;
- tracking por centroide;
- contagem por cruzamento de linha virtual;
- sessoes e eventos salvos em SQLite.

## Como rodar

Na raiz do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r 04_backend_api_contagem/requirements.txt
uvicorn app.main:app --app-dir 04_backend_api_contagem --reload --host 0.0.0.0 --port 8000
```

Depois acesse:

```text
http://localhost:8000
```

## Fonte da camera

No painel, use:

- `0` para webcam padrao;
- `1`, `2` etc. para outras webcams locais;
- URL RTSP para camera IP, por exemplo `rtsp://usuario:senha@ip:554/caminho`.

## Observacao sobre IA

O detector atual e propositalmente simples. Ele serve para iniciar a validacao ao vivo, testar fluxo de camera, tracking, linha virtual, contador e historico.

O projeto ja separa a camada de visao computacional em `app/vision.py`, para futura troca por:

- YOLO;
- ONNX Runtime;
- modelo customizado treinado com imagens reais da operacao.

## Endpoints principais

- `GET /` painel web;
- `GET /api/status` estado atual;
- `POST /api/start` inicia sessao;
- `POST /api/stop` para sessao;
- `POST /api/line` atualiza linha virtual;
- `GET /api/stream` stream MJPEG com overlay;
- `GET /api/sessions` lista sessoes locais;
- `GET /api/events/{session_id}` lista eventos de uma sessao.

## Banco local

O SQLite fica em:

```text
06_dados_e_sessoes/contador_caixas.sqlite3
```

Este arquivo nao deve ser commitado.

## Testes

Para instalar dependencias de teste:

```bash
pip install -r 04_backend_api_contagem/requirements-dev.txt
pytest 09_testes_validacao
```

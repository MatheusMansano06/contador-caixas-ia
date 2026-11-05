# Contador de Caixas com IA

Contagem de caixas por câmera comum (celular, webcam ou câmera IP/RTSP), com **visão computacional clássica + YOLO**. Alternativa em software a câmeras dedicadas com IA embarcada, pensada para recebimento, separação e expedição.

| | |
|---|---|
| **Visão computacional** | OpenCV · YOLO (Ultralytics) · NumPy |
| **Backend** | Python · FastAPI · SQLite |
| **Frontend** | Painel web, versão mobile com a câmera do celular e monitor em outra tela |
| **Testes** | pytest com cenários sintéticos e adversariais |

## Como funciona

Dois modos de contagem:

1. **Linha virtual:** detecta objetos em movimento, acompanha cada um entre frames (tracking por centroide) e conta quando o centro cruza a linha no sentido configurado.
2. **Vigia de área:** o operador desenha a zona, calibra com ela vazia e o sistema conta as caixas presentes.
   - A diferença em relação ao fundo encontra as "manchas" físicas, e o YOLO **confirma** cada uma. Assim o modelo não inventa caixa onde não há nada.
   - Caixas encostadas ou empilhadas são separadas pela aresta escura entre elas.
   - Detecções sobrepostas da mesma caixa são fundidas.
   - Uma guarda de movimento por fração de pixels evita "caixas fantasma".
   - A contagem da sessão persiste quando uma caixa é tapada por um instante (oclusão).

Sessões e eventos de contagem ficam salvos em SQLite, com API para consultar o histórico.

## Treino e dados

- `10_scripts_ferramentas/gerar_dataset_sintetico.py`: gera imagens rotuladas de caixas.
- `10_scripts_ferramentas/coletar_amostras.py` e `POST /api/dataset/capture`: coletam amostras reais da câmera.
- `05_ia_visao_computacional/treinamentos/treinar.py`: treina (fine-tuning) o detector YOLO.
- `10_scripts_ferramentas/testar_modelo.py`: avalia o modelo treinado.

## Estrutura

```
03_frontend_mobile_camera/   painel, versão mobile e monitor
04_backend_api_contagem/     API FastAPI, contagem, vigia de área, detector YOLO
05_ia_visao_computacional/   treino do modelo
09_testes_validacao/         testes de contagem e cenários da vigia
10_scripts_ferramentas/      dataset, coleta e avaliação
01, 02, 07, 08, 11           documentação, requisitos, câmeras, deploy, operação
```

## Rodar localmente

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r 04_backend_api_contagem/requirements.txt
uvicorn app.main:app --app-dir 04_backend_api_contagem --reload --host 0.0.0.0 --port 8000
```

Abra `http://localhost:8000` (ou `/index-mobile.html` no celular). O `ultralytics` puxa o PyTorch (cerca de 2 GB) e só é necessário para a detecção por YOLO.

## Testes

```bash
pip install -r 04_backend_api_contagem/requirements-dev.txt
python -m pytest 09_testes_validacao
```

Documentação completa do produto: [README_DO_PROJETO.md](README_DO_PROJETO.md).

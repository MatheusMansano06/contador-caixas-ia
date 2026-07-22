# Contador de Caixas com IA por Camera

Data de criacao: 22 de julho de 2026
Status atual: MVP ao vivo em desenvolvimento inicial.

Implementacao inicial criada:

- backend local com FastAPI;
- painel web operacional;
- captura ao vivo por webcam USB ou camera IP/RTSP;
- detector local inicial baseado em movimento e contornos;
- tracking por centroide;
- contagem por linha virtual;
- persistencia local em SQLite.

Para rodar a primeira versao, veja:

```text
04_backend_api_contagem/README.md
```

## 1. Resumo do projeto

Este projeto tem como objetivo criar uma solucao de contagem de caixas usando camera e inteligencia artificial. A ideia inicial e permitir que um celular, uma camera IP ou outro dispositivo com camera fique apontado para uma area de passagem de caixas e conte automaticamente a quantidade de caixas que cruzam uma linha ou regiao virtual.

O primeiro foco nao e contar pilhas complexas de caixas paradas. O foco inicial e contar caixas em movimento, passando pela camera de forma controlada, porque esse cenario e mais confiavel para um MVP e mais proximo das tecnicas atuais de contagem por visao computacional.

## 2. Problema que o projeto resolve

Em operacoes logisticas, estoque, separacao, recebimento ou expedicao, a contagem manual de caixas pode ser lenta e sujeita a erro. Uma solucao com camera pode acelerar esse processo, registrar evidencias visuais e criar dados historicos para auditoria, produtividade e controle operacional.

O projeto busca reduzir:

- tempo gasto em contagem manual;
- erros humanos de contagem;
- retrabalho na conferencia de volumes;
- dependencia de equipamentos caros logo no inicio;
- dificuldade de registrar historico das contagens.

## 3. Produto de referencia pesquisado

Foi analisada a camera Intelbras VIP 9320 OBJ FT vendida como camera IP para contagem de caixas.

Pagina de referencia:
https://www.sillis.com.br/produto/camera-intelbras-ip-para-contagem-de-caixas-vip-9320-obj-ft/

Datasheet oficial Intelbras:
https://backend.intelbras.com/sites/default/files/2022-05/datasheet-vip-9320-obj-ft.pdf

Funcionalidades identificadas no produto de referencia:

- contagem de caixas por area;
- contagem de entradas e saidas;
- gerenciamento de filas;
- ate 4 areas configuraveis;
- IA embarcada na camera;
- acesso via web, iOS e Android;
- integracoes por RTSP, ONVIF, API e SDK;
- uso industrial com protecao fisica IP67 e IK10.

Aprendizado importante: existe mercado real para esse tipo de solucao. Porem, a camera Intelbras e um hardware dedicado com IA embarcada. Nosso projeto deve comecar como software flexivel, validando a ideia com celular/camera comum e evoluindo depois para cameras IP ou dispositivos dedicados.

## 4. Como a solucao deve funcionar

Fluxo previsto do MVP:

1. O usuario abre uma interface web no celular.
2. A interface solicita permissao para usar a camera.
3. A camera fica apontada para a area onde as caixas passam.
4. O usuario define uma linha virtual de contagem na tela.
5. A aplicacao captura frames do video.
6. Um modelo de visao computacional detecta caixas nos frames.
7. Um mecanismo de tracking acompanha cada caixa entre frames.
8. Quando o centro de uma caixa cruza a linha virtual no sentido configurado, o contador aumenta.
9. A sessao registra total, horario de inicio, horario de fim e eventos de contagem.

Fluxo visual simplificado:

```text
Camera do celular ou camera IP
        |
        v
Captura de video / frames
        |
        v
Detector de caixas
        |
        v
Tracking dos objetos
        |
        v
Regra de cruzamento de linha
        |
        v
Contador + historico da sessao
```

## 5. Escopo do MVP

O MVP deve entregar:

- captura da camera em tempo real;
- preview do video;
- linha virtual de contagem;
- contador total da sessao;
- deteccao de caixas passando pela camera;
- reducao de dupla contagem usando tracking;
- reset manual da contagem;
- exportacao simples dos resultados da sessao;
- logs basicos para entender erros de deteccao.

O MVP nao deve tentar resolver no primeiro momento:

- contagem perfeita de caixas empilhadas;
- multiplas cameras simultaneas;
- reconhecimento de cliente, pedido ou nota fiscal;
- dashboard gerencial completo;
- integracao com ERP ou WMS;
- funcionamento 100% offline com IA pesada no proprio navegador.

## 6. Arquitetura tecnica recomendada

Arquitetura inicial recomendada:

```text
Frontend Mobile Web
        |
        | envia frames ou stream reduzido
        v
Backend API de Contagem
        |
        | chama detector/tracker
        v
Modulo de IA / Visao Computacional
        |
        v
Resultado: caixas detectadas, IDs rastreados e eventos de cruzamento
```

Componentes principais:

- Frontend Mobile Web: tela usada no celular para camera, linha de contagem e contador.
- Backend API: recebe frames, coordena sessao e devolve resultados.
- Modulo de IA: detecta caixas e acompanha objetos ao longo do tempo.
- Banco ou armazenamento local: guarda sessoes, eventos e metricas.
- Integracoes futuras: cameras IP, RTSP, ONVIF, SDKs e sistemas externos.

## 7. Tecnologias candidatas

Frontend:

- HTML, CSS e JavaScript no prototipo simples;
- React ou Vite se o projeto crescer;
- API getUserMedia para acesso a camera no navegador.

Backend:

- Python;
- FastAPI;
- WebSocket ou HTTP para envio de frames/resultados;
- OpenCV para manipulacao de imagem;
- Ultralytics/YOLO para deteccao e tracking.

IA e visao computacional:

- YOLO para deteccao de caixas;
- tracking por ID para evitar dupla contagem;
- regra de cruzamento de linha;
- dataset proprio quando tivermos imagens reais do ambiente.

Banco e dados:

- SQLite no inicio;
- PostgreSQL em fase mais madura;
- arquivos JSON/CSV para exportacao rapida de sessoes.

Deploy e ambiente:

- ambiente local no inicio;
- HTTPS necessario para camera no celular via navegador;
- possibilidade de usar tunnel HTTPS para testes;
- deploy cloud quando o backend precisar ficar disponivel fora da rede local.

## 8. Limites tecnicos conhecidos

Camera no navegador:

- navegadores modernos exigem contexto seguro para camera;
- em celular, normalmente sera necessario HTTPS;
- testes locais podem exigir tunnel, rede local configurada ou instalacao como app/PWA.

Deteccao:

- caixas muito parecidas com o fundo podem ser ignoradas;
- sombras e baixa iluminacao prejudicam a leitura;
- caixas coladas ou sobrepostas aumentam o risco de erro;
- angulo da camera influencia muito a precisao.

Contagem:

- a linha virtual precisa estar bem posicionada;
- se a caixa parar em cima da linha, pode gerar incerteza;
- se duas caixas passarem grudadas, podem virar uma deteccao so;
- tracking ruim pode causar contagem duplicada.

## 9. Decisoes iniciais de produto

Decisao 1: comecar por caixas em movimento.

Motivo: e o cenario mais viavel para contagem por linha virtual e tracking.

Decisao 2: separar frontend, backend e IA.

Motivo: isso permite trocar o modelo, melhorar o backend e crescer para camera IP sem refazer toda a interface.

Decisao 3: nao depender inicialmente de hardware proprietario.

Motivo: o celular ajuda a validar o valor antes de investir em camera dedicada.

Decisao 4: preparar o projeto para integracao futura com cameras IP.

Motivo: produtos como a Intelbras VIP 9320 OBJ FT mostram que o caminho industrial natural usa RTSP, ONVIF, API e SDK.

## 10. Estrutura de pastas

```text
contador-caixas-ia/
  README_DO_PROJETO.md
  01_documentacao_tecnica/
  02_produto_e_requisitos/
  03_frontend_mobile_camera/
  04_backend_api_contagem/
  05_ia_visao_computacional/
    datasets/
    modelos/
    treinamentos/
    avaliacoes/
  06_dados_e_sessoes/
  07_integracoes_cameras/
  08_infraestrutura_deploy/
  09_testes_validacao/
  10_scripts_ferramentas/
  11_operacao_suporte/
```

Descricao das pastas:

- `01_documentacao_tecnica`: arquitetura, diagramas, decisoes tecnicas e especificacoes detalhadas.
- `02_produto_e_requisitos`: regras de negocio, requisitos, fluxos do usuario e backlog.
- `03_frontend_mobile_camera`: aplicacao web usada no celular para abrir camera, configurar linha e visualizar contagem.
- `04_backend_api_contagem`: API responsavel por sessoes, recebimento de frames, contagem e retorno de resultados.
- `05_ia_visao_computacional`: modelos, datasets, rotulagem, treinamento e avaliacao da IA.
- `05_ia_visao_computacional/datasets`: imagens e videos de treino, validacao e teste.
- `05_ia_visao_computacional/modelos`: pesos de modelos treinados ou baixados.
- `05_ia_visao_computacional/treinamentos`: configuracoes, scripts e saidas de treinamento.
- `05_ia_visao_computacional/avaliacoes`: metricas, comparativos e resultados de testes dos modelos.
- `06_dados_e_sessoes`: arquivos de sessoes de contagem, historicos, exportacoes e amostras.
- `07_integracoes_cameras`: estudos e implementacoes futuras para RTSP, ONVIF, cameras IP e SDKs.
- `08_infraestrutura_deploy`: Docker, configuracoes de servidor, HTTPS, deploy e variaveis de ambiente.
- `09_testes_validacao`: testes automatizados, videos de validacao, criterios de aceite e cenarios reais.
- `10_scripts_ferramentas`: scripts auxiliares para preparar videos, extrair frames, converter datasets e medir resultados.
- `11_operacao_suporte`: guias de uso, suporte, calibracao em campo e procedimentos para operadores.

## 11. Convencoes recomendadas para desenvolvimento

Nomes:

- usar nomes de pastas e arquivos em minusculo;
- usar hifen ou underline de forma consistente;
- evitar acentos em arquivos de codigo;
- manter documentos em Markdown quando possivel.

Codigo:

- separar regras de negocio da interface;
- manter o modulo de IA isolado da API;
- versionar configuracoes importantes;
- registrar parametros de deteccao usados em cada teste;
- evitar depender de um modelo unico desde o inicio.

Dados:

- nunca misturar dataset bruto com dataset tratado;
- registrar origem e data dos videos/imagens;
- separar treino, validacao e teste;
- evitar commitar arquivos grandes sem uma decisao clara de armazenamento.

## 12. Criterios de sucesso da primeira versao

Uma primeira versao sera considerada util se:

- abrir a camera no celular;
- permitir posicionar ou configurar a linha de contagem;
- detectar caixas em movimento em uma cena controlada;
- contar corretamente a maioria das caixas passando uma por vez;
- evitar dupla contagem em casos simples;
- gerar um registro basico da sessao.

Metricas sugeridas:

- precisao de contagem em videos reais;
- taxa de dupla contagem;
- taxa de caixas nao detectadas;
- tempo medio de resposta;
- estabilidade em celular comum.

## 13. Roteiro sugerido de evolucao

Fase 0: planejamento e estrutura.

- definir pastas;
- documentar objetivo;
- mapear referencias;
- decidir escopo do MVP.

Fase 1: prototipo funcional.

- criar frontend de camera;
- criar API local;
- detectar caixas em frames;
- implementar linha virtual;
- contar cruzamentos.

Fase 2: validacao real.

- gravar videos reais;
- medir erros;
- ajustar angulo de camera;
- comparar modelos;
- definir parametros de confianca.

Fase 3: modelo customizado.

- coletar dataset proprio;
- rotular caixas;
- treinar modelo;
- avaliar precisao;
- otimizar desempenho.

Fase 4: produto operacional.

- adicionar historico;
- criar painel;
- exportar relatorios;
- integrar camera IP;
- preparar deploy.

## 14. Perguntas tecnicas ainda abertas

- A camera sera usada na mao ou fixa em um suporte?
- As caixas passam em esteira, bancada, chao ou porta de entrada?
- As caixas passam uma por vez ou podem passar agrupadas?
- A contagem precisa diferenciar entrada e saida?
- O ambiente tem internet durante o uso?
- A solucao precisa funcionar fora da rede local?
- Qual nivel de erro e aceitavel no primeiro teste?

## 15. Referencias externas uteis

- Ultralytics Object Counting:
  https://docs.ultralytics.com/guides/object-counting

- Roboflow Line Crossing Counter:
  https://blog.roboflow.com/count-objects-crossing-lines/

- Roboflow Line Counter Block:
  https://inference.roboflow.com/workflows/blocks/line_counter/

- MDN getUserMedia:
  https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia

- Google MediaPipe Object Detector Web:
  https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/web_js

- Google MediaPipe Object Detector Customization:
  https://developers.google.com/edge/mediapipe/solutions/customization/object_detector

- Produto de referencia Sillis / Intelbras:
  https://www.sillis.com.br/produto/camera-intelbras-ip-para-contagem-de-caixas-vip-9320-obj-ft/

- Datasheet oficial Intelbras VIP 9320 OBJ FT:
  https://backend.intelbras.com/sites/default/files/2022-05/datasheet-vip-9320-obj-ft.pdf

## 16. Observacao final para desenvolvedores

Este repositorio ainda nao deve assumir uma solucao definitiva. A decisao tecnica mais importante neste momento e validar a contagem por camera em uma cena real simples. Depois disso, o projeto pode evoluir para modelos customizados, cameras IP, dashboard, banco de dados e integracoes com sistemas de operacao logistica.

Antes de programar, o ideal e confirmar o cenario fisico de teste: posicao da camera, distancia das caixas, iluminacao, fundo, velocidade de passagem e se as caixas passam uma por vez ou agrupadas.

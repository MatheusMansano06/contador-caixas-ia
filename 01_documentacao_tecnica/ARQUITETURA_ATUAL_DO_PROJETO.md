# Arquitetura Atual do Projeto

Data: 22 de julho de 2026
Status: arquitetura atual aprovada para inicio do MVP ao vivo

## 1. Objetivo desta arquitetura

Este documento registra a arquitetura atual escolhida para o projeto de contagem de caixas com IA.

A decisao atual e construir uma solucao que funcione ao vivo, em tempo real, sem depender de app nativo como base do produto inicial e sem depender de LLMs pagos por token no motor principal de contagem.

## 2. Decisao principal

A arquitetura atual do projeto sera:

- camera fixa como fonte de video;
- processamento local em tempo real;
- painel web para operacao;
- IA de visao computacional rodando localmente;
- backend local coordenando sessoes, contagem e eventos.

Resumo da decisao:

```text
Camera IP ou webcam USB
        |
        v
Backend local de captura e contagem
        |
        +--> Modulo local de IA e tracking
        |
        +--> Banco local de sessoes e eventos
        |
        v
Painel web no celular ou computador
```

## 3. O que foi descartado neste momento

Neste momento, decidimos nao usar como base principal:

- app nativo mobile;
- contagem por envio de imagens para Codex, Claude ou outros LLMs;
- operacao baseada somente em videos gravados;
- inferencia pesada dentro do navegador como base do produto comercial.

Motivos:

- app nativo aumenta custo e manutencao cedo demais;
- LLM por token tende a ficar caro para video ao vivo e escala comercial;
- video gravado ajuda em teste e treino, mas nao atende o objetivo principal;
- inferencia centralizada no navegador tende a ser menos previsivel para venda em operacao.

## 4. Principios da arquitetura

- tempo real antes de tudo;
- processamento local para reduzir latencia e custo recorrente;
- estrutura preparada para futura venda como produto;
- separacao clara entre interface, backend e IA;
- compatibilidade futura com camera IP e edge box;
- uso de nuvem apenas como complemento, nao como dependencia do contador.

## 5. Arquitetura funcional atual

### 5.1 Fonte de video

O sistema deve aceitar inicialmente:

- webcam USB;
- camera IP com RTSP.

No MVP, a camera deve ficar fixa e apontada para uma area controlada de passagem de caixas.

### 5.2 Backend local

O backend local sera o centro do sistema.

Responsabilidades:

- receber o stream da camera;
- capturar frames;
- enviar frames para o modulo de IA;
- controlar sessoes de contagem;
- aplicar regras de negocio da contagem;
- registrar eventos e resultados;
- disponibilizar API e atualizacoes em tempo real para o painel web.

### 5.3 Modulo local de IA

O modulo de IA sera responsavel por:

- detectar caixas nos frames;
- acompanhar cada caixa com tracking por ID;
- identificar cruzamento da linha virtual;
- evitar dupla contagem em casos simples;
- retornar eventos de contagem para o backend.

Estrutura logica:

```text
Frame
  -> detector de caixas
  -> tracking por ID
  -> regra de cruzamento de linha
  -> evento IN/OUT ou incremento simples
```

### 5.4 Painel web

O painel web sera usado no celular ou computador, sem necessidade de app nativo no inicio.

Funcoes iniciais:

- iniciar e encerrar sessao;
- visualizar o video;
- configurar linha virtual;
- acompanhar contador em tempo real;
- ver estado da camera e da sessao;
- resetar contagem quando permitido;
- consultar historico basico da sessao.

### 5.5 Banco local e historico

O projeto deve manter armazenamento local no inicio.

Dados previstos:

- sessoes de contagem;
- horario de inicio e fim;
- totais por sessao;
- eventos de cruzamento;
- configuracao usada na camera;
- parametros da linha virtual.

## 6. Fluxo ao vivo da operacao

```text
1. Operador acessa o painel web
2. Sistema conecta a camera
3. Operador posiciona ou ajusta a linha virtual
4. Backend inicia sessao
5. Frames entram no motor local de IA
6. Objetos sao detectados e rastreados
7. Cruzamentos validos incrementam a contagem
8. Painel mostra contador em tempo real
9. Sessao e eventos ficam registrados localmente
```

## 7. Stack tecnica inicial recomendada

### 7.1 Backend

- Python
- FastAPI
- WebSocket para atualizacao em tempo real
- OpenCV para captura e manipulacao de frames

### 7.2 IA

- detector de objetos voltado para caixas
- tracking por ID
- regra de line crossing
- exportacao futura para ONNX

Observacao:

O motor de IA deve ser pensado para rodar localmente e poder evoluir para edge hardware no futuro.

### 7.3 Frontend

- painel web
- HTML, CSS e JavaScript no inicio ou frontend leve com Vite
- interface acessivel por navegador de celular e desktop

### 7.4 Dados

- SQLite no inicio
- exportacao simples para JSON ou CSV

## 8. Estrutura de deploy prevista

### 8.1 MVP interno

Ambiente inicial:

- notebook ou computador local;
- webcam USB ou camera IP;
- rede local;
- painel web acessado via navegador.

### 8.2 Produto comercial futuro

Ambiente futuro mais provavel:

- mini PC ou edge box;
- camera IP fixa;
- painel web local;
- sincronizacao opcional com servidor central.

## 9. Papel da nuvem no futuro

A nuvem nao sera responsavel pela contagem em tempo real.

A nuvem, se existir depois, deve servir para:

- cadastro de clientes;
- atualizacao de modelos;
- monitoramento de instalacoes;
- backups e relatorios;
- suporte remoto;
- gestao de licencas.

## 10. Beneficios desta arquitetura

- elimina custo por token no processo principal;
- reduz dependencia de internet;
- melhora latencia;
- facilita venda por unidade instalada;
- aproxima o MVP da arquitetura comercial final;
- permite evolucao para camera IP e edge appliance.

## 11. Limitacoes assumidas no inicio

- foco em uma unica camera por sessao;
- ambiente controlado;
- camera fixa;
- caixas em movimento;
- casos simples primeiro, nao cenarios extremos;
- calibracao manual inicial da linha de contagem.

## 12. Escopo do MVP desta arquitetura

O MVP deve entregar:

- conexao com uma camera ao vivo;
- preview do fluxo de video;
- definicao da linha virtual;
- deteccao de caixas em tempo real;
- tracking basico;
- contagem ao vivo;
- armazenamento local da sessao;
- painel web funcional.

## 13. Fora do escopo imediato

- app iOS ou Android nativo;
- multiplas cameras simultaneas;
- integracao com ERP ou WMS;
- dashboard gerencial completo;
- analise multiunidade em nuvem;
- cobranca por uso baseada em IA de terceiros.

## 14. Direcao comercial

Se a validacao operacional funcionar, esta arquitetura permite evoluir o projeto para um produto vendavel com este desenho:

- cliente instala uma camera fixa;
- cliente usa um mini PC ou edge box com o sistema;
- operador acessa tudo pelo navegador;
- contagem acontece localmente;
- servicos centralizados entram depois como camada de gestao.

## 15. Proximo passo tecnico sugerido

Com base nesta arquitetura, o proximo passo de implementacao deve ser:

1. criar backend local de captura;
2. criar painel web minimo;
3. integrar deteccao e tracking local;
4. implementar linha virtual e logica de contagem;
5. registrar sessoes localmente.

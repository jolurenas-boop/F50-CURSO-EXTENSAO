# Fadiga sEMG — bíceps e tríceps braquial, coleta única (Streamlit)

Análise de fadiga em isometria sustentada (ex.: 30 s a 50% da CIVM) a partir de **um único arquivo .txt**
com todos os canais.

| Col. | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| | Tempo | Bíceps D | Bíceps E | Tríceps D | Tríceps E | Carga D | Carga E |

O layout é editável em *Colunas e canais* (com prévia numerada do arquivo). Separador, decimal e linhas de
cabeçalho são detectados automaticamente ou definidos à mão.

## Como executar
```bash
pip install -r requirements.txt
streamlit run app.py
```
PDF: WeasyPrint (no Streamlit Cloud, mantenha o `packages.txt`); sem ele, o app usa um Chrome/Chromium/Edge
instalado; sem nenhum dos dois, entrega o relatório em HTML.

## O que é calculado
- **Força (por lado)**: tara pelo repouso antes do onset; onset; pico inicial e overshoot; início do patamar quando a
  média móvel (0,5 s) entra e permanece na faixa **média ± 2 DP** por ≥ 2 s, com verificação de estabilidade
  (CV ≤ 10%, deriva ≤ 20%) — se não estabilizar, o lado é sinalizado e usa onset + 3 s (ou o início que você
  definir à mão); força média, inicial e final, variação e inclinação, DP, CV, CV sem tendência, % do tempo na faixa.
- **EMG (por músculo, no patamar do seu lado)**: filtro 20–450 Hz + notch 60 Hz; janelas de **0,5 s com 50% de
  sobreposição** (0–0,5; 0,25–0,75; …); MDF e MNF (Welch, segmentos de 0,25 s), RMS e ENM por janela; inclinação
  (regressão, R², p corrigido pela sobreposição), valores inicial/final, queda da MDF pelos extremos e pela
  regressão, variação do RMS e da ENM; exclusão de janelas com artefato (limite ±5000 µV, pico > 15 DP robustos,
  RMS > 2,5× a mediana).
- **Assimetria D−E** da queda e da inclinação da MDF por músculo.
- **Integridade**: colunas com sinal idêntico (canal gravado em duplicidade) são sinalizadas.

## Saídas
Tabelas na tela, CSV (EMG, força, todas as janelas) e **relatório em PDF** (identificação, síntese, mecânica,
fadiga eletromiográfica, figuras de força/MDF/RMS/ENM/EMG, controle de qualidade, métodos).

Observação: na flexão do cotovelo o tríceps é antagonista — leia o RMS dele como coativação; a ENM do antagonista
não é eficiência daquele músculo.

## Estrutura
| Arquivo | Função |
|---|---|
| `app.py` | Interface |
| `coleta_unica.py` | Leitura do arquivo de 7 colunas, montagem da coleta, relatório da coleta |
| `fadiga_nucleo.py` | Processamento (mesmo núcleo do app de fadiga do supino) |
| `fadiga_graficos.py`, `fadiga_individual.py`, `fadiga_relatorio.py` | Figuras, tabelas de síntese e HTML → PDF |

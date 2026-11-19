"""Fadiga em COLETA ÚNICA a partir de um arquivo com todas as colunas.

Layout padrão (editável):  Col. 1 Tempo | 2 Bíceps D | 3 Bíceps E | 4 Tríceps D | 5 Tríceps E | 6 Carga D | 7 Carga E
A análise é a mesma do app de fadiga (tara, patamar por lado com média ± k·DP + verificação de CV/deriva,
janelas de 0,5 s com 50% de sobreposição, MDF/MNF/RMS/ENM, artefatos, regressão, assimetria D−E).
"""
from __future__ import annotations

import hashlib
import html
import io
import re
from datetime import datetime

import numpy as np
import pandas as pd

import fadiga_nucleo as fn
import fadiga_individual as fi
from fadiga_graficos import fig_forca, fig_emg, fig_tendencia
from fadiga_relatorio import CSS, _figura, _tabela, _tabela_txt, html_para_pdf

# código -> coluna (1 = Tempo)
COLUNAS_PADRAO = {'BBD': 2, 'BBE': 3, 'TBD': 4, 'TBE': 5, 'CCD': 6, 'CCE': 7}
SEPARADORES = {'auto': None, 'tab': '\t', 'ponto_virgula': ';', 'espaco': r'\s+', 'virgula': ','}


def _detecta_sep(linhas: list[str]) -> str:
    amostra = [l for l in linhas if l.strip()][:30]
    for chave, sep in (('tab', '\t'), ('ponto_virgula', ';')):
        if amostra and all(sep in l for l in amostra):
            return chave
    if amostra and all(len(l.split()) > 1 for l in amostra):
        return 'espaco'
    return 'virgula'


def le_tabela(conteudo: bytes, separador: str = 'auto', decimal: str = 'auto', cabecalho: int = 0) -> pd.DataFrame:
    """Lê o arquivo inteiro como tabela numérica; colunas numeradas a partir de 1."""
    texto = conteudo.decode('utf-8-sig', errors='replace')
    linhas = texto.splitlines()[cabecalho:]
    chave = _detecta_sep(linhas) if separador == 'auto' else separador
    if decimal == 'auto':
        amostra = ' '.join(linhas[:50])
        decimal = ',' if (chave != 'virgula' and ',' in amostra and '.' not in amostra) else '.'
    corpo = '\n'.join(linhas)
    if chave == 'tab':
        df = pd.read_csv(io.StringIO(corpo), sep='\t', header=None, decimal=decimal)
    else:
        df = pd.read_csv(io.StringIO(corpo), sep=SEPARADORES[chave], header=None, decimal=decimal, engine='python')
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna(axis=0, how='any')
    if len(df) < 100:
        raise ValueError(f'só {len(df)} linhas numéricas: confira o separador, o decimal e as linhas de cabeçalho.')
    df.columns = range(1, df.shape[1] + 1)
    return df.reset_index(drop=True)


def previa(conteudo: bytes, separador='auto', decimal='auto', cabecalho=0, n=8) -> pd.DataFrame:
    df = le_tabela(conteudo, separador, decimal, cabecalho).head(n)
    df.columns = [f'Col. {c}' for c in df.columns]
    return df


def monta_teste(conteudo: bytes, nome_arquivo: str, id_vol: str, coluna_tempo: int, colunas: dict,
                separador='auto', decimal='auto', cabecalho=0) -> tuple[fn.Teste, list[str]]:
    """Cria um Teste (um 'canal' por coluna) para o núcleo de fadiga. colunas: {código: nº da coluna}."""
    df = le_tabela(conteudo, separador, decimal, cabecalho)
    usadas = [coluna_tempo] + list(colunas.values())
    if len(set(usadas)) != len(usadas):
        raise ValueError('cada canal (e o Tempo) precisa apontar para uma coluna diferente.')
    if max(usadas) > df.shape[1]:
        raise ValueError(f'o arquivo tem {df.shape[1]} colunas, mas o mapeamento usa a coluna {max(usadas)}.')
    t = df[coluna_tempo].to_numpy(float)
    avisos = []
    canais = {}
    for cod, col in colunas.items():
        x = df[col].to_numpy(float)
        h = hashlib.sha256(x.tobytes()).hexdigest()
        canais[cod] = (f'{nome_arquivo} [col. {col}]', h, np.column_stack([t, x]))
    return fn.Teste(id=id_vol, cond='U', coleta='1', canais=canais), avisos


def processa(teste: fn.Teste, mapa: dict, p: fn.Parametros, inicio_manual: dict | None = None,
             excluir_manual: list | None = None) -> fn.Resultado:
    excl, msgs = fn.verifica_integridade({teste.chave: teste}, mapa)
    excluir = dict(excl.get(teste.chave, {}))
    for c in excluir_manual or []:
        excluir[c] = 'excluído manualmente pelo usuário'
    r = fn.processa_teste(teste, mapa, p, {'1': 'única'}, inicio_manual, excluir)
    r.condicao, r.momento = 'Coleta', 'única'
    r.avisos[:0] = [m.replace('MESMO arquivo', 'MESMA coluna (sinal idêntico)') for m in msgs]
    return r


def nome_pdf(r, nome_arquivo: str) -> str:
    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', nome_arquivo.rsplit('.', 1)[0]) or 'registro'
    return f'relatorio_fadiga_{stem}.pdf'


def monta_html(r: fn.Resultado, nome_arquivo: str, meta: dict | None = None) -> str:
    meta = meta or {}
    p = fn.Parametros(**r.parametros)
    ue, uf = p.unidade_emg, p.unidade_forca
    agora = datetime.now().strftime('%d/%m/%Y %H:%M')
    ident = [('Arquivo analisado', nome_arquivo), ('Voluntário', r.id),
             ('Protocolo', meta.get('protocolo') or 'Fadiga isométrica sustentada'),
             ('Avaliador', meta.get('avaliador') or '—'), ('Observações', meta.get('observacoes') or '—'),
             ('Duração do registro', f'{fn.fmt(r.duracao_s, 2)} s ({r.n_amostras} amostras a {p.fs:g} Hz)'),
             ('Canais', ', '.join(f'{c} (col. {v[0].split("col. ")[-1].rstrip("]")})' for c, v in r.arquivos.items()))]
    frases = [f for f in fi.sintese([r]) if 'Comparações entre coletas' not in f]
    lados = list(r.forca)
    mec = ''.join(f'<h3>Lado {fn.NOME_LADO[l] or "único"}</h3>' + _tabela(fi.tabela_mecanica([r], l), 'Variável')
                  for l in lados)
    assim = (('<h3>Assimetria bilateral (direito − esquerdo)</h3>' + _tabela(r.df_assimetria, 'Grupo muscular')
              + '<p class="nota">Negativo na diferença da queda = queda maior no lado direito.</p>')
             if r.df_assimetria is not None and len(r.df_assimetria) else '')
    avisos = ''.join(f'<div class="aviso">{html.escape(a)}</div>' for a in r.avisos) or \
        '<div class="ok">Nenhum aviso de qualidade foi gerado nas verificações automáticas.</div>'
    atenu = _tabela(r.atenuacao_notch.to_frame(), 'Canal') if len(r.atenuacao_notch) else ''
    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório de fadiga — {html.escape(r.id)}</title><style>{CSS}</style></head><body>
<h1>Relatório de fadiga sEMG — isometria sustentada</h1>
<div class="sub">Bíceps e tríceps braquial + células de carga · coleta única · gerado em {agora} · versão {fn.APP_VERSION}</div>

<h2>1. Identificação</h2>
{_tabela_txt(ident)}

<h2>2. Síntese</h2>
{''.join(f'<p>{f}</p>' for f in frases)}
{_tabela(fi.tabela_padrao([r]), 'Músculo')}
<p class="nota">↓/↑ = inclinação ao longo do patamar com p &lt; 0,05 (p corrigido pela sobreposição das janelas);
= = sem tendência significativa. "Padrão clássico" de fadiga = MDF cai e RMS sobe ao mesmo tempo.</p>

<h2>3. Fadiga mecânica (células de carga)</h2>
{mec}
<p class="nota">Patamar: a média móvel ({p.media_movel_s:g} s) da força entra e permanece na faixa média ± {p.k_dp:g} DP por
≥ {p.dur_min_estavel_s:g} s, em cada lado, com verificação de estabilidade (CV ≤ {p.cv_max_pct:g}% e deriva ≤
{p.deriva_max_pct:g}%). Inicial/final = média dos primeiros/últimos {p.extremos_s:g} s do patamar. CV sem tendência =
variabilidade após remover a reta de regressão.</p>
{_figura(fig_forca(r), 'Figura 1. Força tarada, faixa média ± k·DP do patamar, onset e início do patamar de cada lado.')}

<h2>4. Fadiga eletromiográfica</h2>
{_tabela(r.df_emg, 'Músculo')}
<h3>Métricas complementares</h3>
{_tabela(r.df_emg_extra, 'Canal')}
<p class="nota">Janelas de {p.janela_espectral_s:g} s com {p.sobreposicao:.0%} de sobreposição dentro do patamar do lado
correspondente. Slope = inclinação da regressão linear vs. tempo (p corrigido pela sobreposição). Queda MDF (tabela
principal) = extremos; "Queda pela regressão" usa o patamar inteiro e é mais robusta. ENM = força do lado ÷ RMS
(no antagonista, leia o RMS como coativação, não como eficiência).</p>
{assim}
{_figura(fig_tendencia(r, 'MDF_Hz', 'MDF (Hz)'), 'Figura 2. Frequência mediana por janela no patamar e reta de regressão (x = janela excluída).')}
{_figura(fig_tendencia(r, 'RMS', f'RMS ({ue})'), 'Figura 3. RMS por janela no patamar e reta de regressão.')}
{_figura(fig_tendencia(r, 'ENM', f'ENM ({uf}/{ue})'), 'Figura 4. Eficiência neuromuscular por janela.')}
{_figura(fig_emg(r), 'Figura 5. EMG filtrado; verde = patamar analisado; vermelho = janelas com artefato.')}

<h2>5. Controle de qualidade</h2>
{avisos}
<h3>Auditoria espectral</h3>
{atenu}

<h2>6. Métodos e rastreabilidade</h2>
{_tabela_txt([(f'{g} — {n}', v) for g, n, v in fn.descricao_parametros(p)])}
{_tabela_txt(list(fn.versoes().items()))}
</body></html>"""


def gera_pdf(r, nome_arquivo, meta=None) -> bytes:
    return html_para_pdf(monta_html(r, nome_arquivo, meta))

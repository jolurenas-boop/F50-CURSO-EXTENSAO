"""Relatório INDIVIDUAL de fadiga: compara as coletas de um mesmo voluntário (Pré × Pós de cada condição e,
quando houver as 4 coletas, Intervenção × Controle).

Tudo aqui é descritivo (um único indivíduo): diferenças entre coletas são apresentadas como Δ, sem teste
de hipótese entre sessões. Dentro de cada coleta, a inclinação de cada variável tem p-valor corrigido pela
sobreposição das janelas.
"""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd

import fadiga_nucleo as fn
from fadiga_graficos import fig_forca_sessoes, fig_tendencia_sessoes, fig_barras_sessoes
from fadiga_relatorio import CSS, _figura, _tabela, _tabela_txt

ALFA = 0.05

VARS_MECANICA = [
    ('origem_tara', 'Origem da tara'),
    ('i_onset', 'Instante do onset (s)'),
    ('tempo_estabilizacao_s', 'Tempo até estabilizar (s)'),
    ('estabilizou', 'Estabilizou'),
    ('pico_inicial', 'Pico inicial ({uf})'),
    ('overshoot_pct', 'Overshoot inicial (%)'),
    ('media', 'Força média no patamar ({uf})'),
    ('forca_ini', 'Força inicial ({uf})'),
    ('forca_fim', 'Força final ({uf})'),
    ('var_forca_pct', 'Variação da força no patamar (%)'),
    ('slope', 'Inclinação da força ({uf}/s)'),
    ('dp', 'DP no patamar ({uf})'),
    ('cv', 'CV no patamar (%)'),
    ('cv_sem_tendencia', 'CV sem tendência (%)'),
    ('pct_na_faixa', '% do tempo na faixa média ± k·DP'),
    ('duracao_patamar_s', 'Duração do patamar (s)'),
]

VARS_EMG = [   # (chave, rótulo, casas decimais)
    ('MDF_ini', 'MDF inicial (Hz)', 1),
    ('MDF_slope', 'Inclinação da MDF (Hz/s)', 3),
    ('MDF_slope_norm', 'Inclinação normalizada da MDF (%/s)', 3),
    ('Queda_MDF_reg', 'Queda da MDF pela regressão (%)', 1),
    ('Queda_MDF', 'Queda da MDF pelos extremos (%)', 1),
    ('MNF_slope', 'Inclinação da MNF (Hz/s)', 3),
    ('RMS_ini', 'RMS inicial ({ue})', 1),
    ('RMS_slope', 'Inclinação do RMS ({ue}/s)', 2),
    ('Var_RMS', 'Variação do RMS (%)', 1),
    ('Var_ENM', 'Variação da ENM (%)', 1),
    ('pct_excluidas', 'Janelas excluídas por artefato (%)', 0),
]


# ------------------------------------------------------------------ utilidades
def ordena(resultados: list) -> list:
    return sorted(resultados, key=lambda r: (r.chave.split('_', 1)[1][0] != 'C', r.chave.split('_', 1)[1][0],
                                             int(r.coleta) if r.coleta.isdigit() else 99))


def rotulo(r) -> str:
    return f'{r.condicao} {r.momento}'


def pares_pre_pos(resultados: list) -> list[tuple]:
    """[(condição, Pré, Pós)] para cada condição com coletas 1 e 2 (ou as duas primeiras disponíveis)."""
    por_cond = {}
    for r in resultados:
        por_cond.setdefault(r.condicao, []).append(r)
    pares = []
    for cond, lst in por_cond.items():
        lst = sorted(lst, key=lambda r: int(r.coleta) if r.coleta.isdigit() else 99)
        if len(lst) >= 2:
            pares.append((cond, lst[0], lst[1]))
    return pares


def _num(v):
    if isinstance(v, (bool, np.bool_)):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _valor_forca(r, lado, chave):
    fr = r.forca.get(lado)
    if fr is None:
        return np.nan
    v = fr.get(chave, np.nan)
    if chave in ('i_onset', 'tempo_estabilizacao_s') and not fr.get('repouso_ok', True):
        return 'n.c. (registro começou com contração)'
    if chave == 'i_onset':
        return v / r.parametros['fs']
    return v


# ------------------------------------------------------------------ tabelas
def tabela_sessoes(resultados) -> pd.DataFrame:
    linhas = {}
    for r in resultados:
        excl = [a.split(':')[0].replace('Canal ', '').replace(' excluído', '') for a in r.avisos
                if a.startswith('Canal ') and 'excluído' in a]
        linhas[rotulo(r)] = {
            'Teste': r.chave, 'Duração (s)': r.duracao_s,
            'Canais analisados': ', '.join(sorted(r.arquivos)),
            'Canais excluídos': ', '.join(excl) or '—',
            'Avisos de qualidade': str(len(r.avisos)),
        }
    return pd.DataFrame(linhas).T


def tabela_mecanica(resultados, lado) -> pd.DataFrame:
    uf = resultados[0].parametros['unidade_forca']
    k = resultados[0].parametros['k_dp']
    tab = {}
    for chave, rot in VARS_MECANICA:
        rot = rot.format(uf=uf).replace('k·DP', f'{k:g} DP')
        lin = {}
        for r in resultados:
            v = _valor_forca(r, lado, chave)
            lin[rotulo(r)] = ('sim' if v else 'NÃO') if chave == 'estabilizou' and lado in r.forca else (
                v if lado in r.forca else '—')
        deltas = {}
        for cond, pre, pos in pares_pre_pos(resultados):
            a, b = _num(_valor_forca(pre, lado, chave)), _num(_valor_forca(pos, lado, chave))
            deltas[cond] = b - a
            lin[f'Δ Pós−Pré ({cond})'] = b - a if np.isfinite(a) and np.isfinite(b) else '—'
        if 'Intervenção' in deltas and 'Controle' in deltas:
            dd = deltas['Intervenção'] - deltas['Controle']
            lin['ΔIntervenção − ΔControle'] = dd if np.isfinite(dd) else '—'
        tab[rot] = lin
    return pd.DataFrame(tab).T


def tabela_emg(resultados, chave) -> pd.DataFrame:
    musculos = []
    for r in resultados:
        for i, c in enumerate(r.emg):
            if c not in [m[0] for m in musculos]:
                musculos.append((c, r.df_emg.index[i]))
    pares = pares_pre_pos(resultados)
    tab = {}
    for c, nome in musculos:
        lin = {rotulo(r): (r.emg[c]['resumo'][chave] if c in r.emg else np.nan) for r in resultados}
        deltas = {}
        for cond, pre, pos in pares:
            a = pre.emg[c]['resumo'][chave] if c in pre.emg else np.nan
            b = pos.emg[c]['resumo'][chave] if c in pos.emg else np.nan
            deltas[cond] = b - a
            lin[f'Δ Pós−Pré ({cond})'] = b - a
        if 'Intervenção' in deltas and 'Controle' in deltas:
            lin['ΔIntervenção − ΔControle'] = deltas['Intervenção'] - deltas['Controle']
        tab[nome] = lin
    return pd.DataFrame(tab).T


def _seta(slope, p):
    if not np.isfinite(slope) or not np.isfinite(p) or p >= ALFA:
        return '='
    return '↓' if slope < 0 else '↑'


def tabela_padrao(resultados) -> pd.DataFrame:
    """Tendência de MDF e RMS em cada coleta (↓/↑ = inclinação com p < 0,05; = sem tendência)."""
    tab = {}
    for r in resultados:
        for i, (c, e) in enumerate(r.emg.items()):
            s = e['resumo']
            m, a = _seta(s['MDF_slope'], s['MDF_p']), _seta(s['RMS_slope'], s['RMS_p'])
            if m == '↓' and a == '↑':
                txt = 'MDF↓ RMS↑ (padrão clássico)'
            else:
                txt = f'MDF{m} RMS{a}'
            tab.setdefault(r.df_emg.index[i], {})[rotulo(r)] = txt
    return pd.DataFrame(tab).T


def tabela_assimetria(resultados) -> pd.DataFrame:
    tab = {}
    for r in resultados:
        if r.df_assimetria is None or not len(r.df_assimetria):
            continue
        for g, row in r.df_assimetria.iterrows():
            tab.setdefault(g, {})[rotulo(r)] = row['Diferença D−E da queda (p.p.)']
    return pd.DataFrame(tab).T


# ------------------------------------------------------------------ síntese automática
def sintese(resultados) -> list[str]:
    frases = []
    ue = resultados[0].parametros['unidade_emg']
    uf = resultados[0].parametros['unidade_forca']
    for r in resultados:
        n = len(r.emg)
        n_mdf = sum(1 for e in r.emg.values() if _seta(e['resumo']['MDF_slope'], e['resumo']['MDF_p']) == '↓')
        n_rms = sum(1 for e in r.emg.values() if _seta(e['resumo']['RMS_slope'], e['resumo']['RMS_p']) == '↑')
        frases.append(f'<b>{rotulo(r)}</b>: MDF com queda significativa ao longo do patamar em {n_mdf} de {n} músculos; '
                      f'RMS com aumento significativo em {n_rms} de {n}.')
    for cond, pre, pos in pares_pre_pos(resultados):
        comuns = [c for c in pre.emg if c in pos.emg]
        if not comuns:
            continue
        d_slope = [pos.emg[c]['resumo']['MDF_slope'] - pre.emg[c]['resumo']['MDF_slope'] for c in comuns]
        d_queda = [pos.emg[c]['resumo']['Queda_MDF_reg'] - pre.emg[c]['resumo']['Queda_MDF_reg'] for c in comuns]
        d_rms = [pos.emg[c]['resumo']['RMS_slope'] - pre.emg[c]['resumo']['RMS_slope'] for c in comuns]
        mais_neg = sum(1 for d in d_slope if np.isfinite(d) and d < 0)
        mais_queda = sum(1 for d in d_queda if np.isfinite(d) and d < 0)
        mais_rms = sum(1 for d in d_rms if np.isfinite(d) and d > 0)
        nomes = {c: pre.df_emg.index[list(pre.emg).index(c)] for c in comuns}
        i_max = int(np.nanargmin(d_slope)) if np.isfinite(d_slope).any() else None
        frase = (f'<b>{cond}, Pós × Pré</b> (fadiga eletromiográfica): a inclinação da MDF ficou mais negativa em '
                 f'{mais_neg} de {len(comuns)} músculos (Δ médio {fn.fmt(np.nanmean(d_slope), 3)} Hz/s); a queda da MDF '
                 f'(regressão) foi maior em {mais_queda} de {len(comuns)} (Δ médio {fn.fmt(np.nanmean(d_queda), 1)} p.p.); '
                 f'a inclinação '
                 f'do RMS aumentou em {mais_rms} de {len(comuns)} (Δ médio {fn.fmt(np.nanmean(d_rms), 2)} {ue}/s).')
        if i_max is not None:
            frase += (f' Maior piora da inclinação da MDF: {nomes[comuns[i_max]]} '
                      f'({fn.fmt(d_slope[i_max], 3)} Hz/s).')
        frases.append(frase)
        for lado in pre.forca:
            if lado not in pos.forca:
                frases.append(f'<b>{cond}, mecânica, lado {fn.NOME_LADO[lado] or "único"}</b>: sem dado válido no Pós — '
                              f'comparação mecânica não feita para este lado.')
                continue
            a, b = pre.forca[lado], pos.forca[lado]
            t_est = (f'tempo até estabilizar {fn.fmt(a["tempo_estabilizacao_s"])} → {fn.fmt(b["tempo_estabilizacao_s"])} s; '
                     if a['repouso_ok'] and b['repouso_ok'] else
                     'tempo até estabilizar não comparável (um dos registros começou com a contração em curso); ')
            frases.append(
                f'<b>{cond}, mecânica, lado {fn.NOME_LADO[lado] or "único"}</b>: força média no patamar '
                f'{fn.fmt(a["media"])} → {fn.fmt(b["media"])} {uf} ({fn.fmt(100 * (b["media"] - a["media"]) / a["media"], 1)}%); '
                f'CV {fn.fmt(a["cv"], 1)} → {fn.fmt(b["cv"], 1)}%; variação da força dentro do patamar '
                f'{fn.fmt(a["var_forca_pct"], 1)} → {fn.fmt(b["var_forca_pct"], 1)}%; ' + t_est + 'overshoot '
                f'{fn.fmt(a["overshoot_pct"], 1)} → {fn.fmt(b["overshoot_pct"], 1)}%.')
    pares = {c: (pre, pos) for c, pre, pos in pares_pre_pos(resultados)}
    if 'Intervenção' in pares and 'Controle' in pares:
        (ip, io), (cp, co) = pares['Intervenção'], pares['Controle']
        comuns = [c for c in ip.emg if c in io.emg and c in cp.emg and c in co.emg]
        dd = [(io.emg[c]['resumo']['MDF_slope'] - ip.emg[c]['resumo']['MDF_slope'])
              - (co.emg[c]['resumo']['MDF_slope'] - cp.emg[c]['resumo']['MDF_slope']) for c in comuns]
        melhor = sum(1 for d in dd if np.isfinite(d) and d > 0)
        frases.append(f'<b>Intervenção × Controle</b>: a variação Pós−Pré da inclinação da MDF foi menos negativa na '
                      f'Intervenção do que no Controle em {melhor} de {len(comuns)} músculos (ΔI − ΔC médio '
                      f'{fn.fmt(np.nanmean(dd), 3)} Hz/s; valores positivos = menor piora com a intervenção).')
    for r in resultados:
        div = [r.df_emg.index[i] for i, e in enumerate(r.emg.values())
               if np.isfinite(e['resumo']['Queda_MDF']) and np.isfinite(e['resumo']['Queda_MDF_reg'])
               and abs(e['resumo']['Queda_MDF'] - e['resumo']['Queda_MDF_reg']) > 10]
        if div:
            frases.append(f'<b>{rotulo(r)}</b>: a queda da MDF pelos extremos e pela regressão diferem em mais de 10 p.p. em '
                          f'{", ".join(div)} — valores dos extremos instáveis; use a regressão.')
    # qualidade
    for r in resultados:
        crit = [a for a in r.avisos if 'excluído' in a or 'NÃO estabilizou' in a or 'sem repouso' in a
                or 'Sem célula' in a]
        art = [(r.df_emg.index[i], e['resumo']['pct_excluidas']) for i, e in enumerate(r.emg.values())
               if np.isfinite(e['resumo']['pct_excluidas']) and e['resumo']['pct_excluidas'] >= 20]
        if crit or art:
            itens = [html.escape(a) for a in crit] + [f'{html.escape(m)}: {v:.0f}% das janelas excluídas por artefato'
                                                      for m, v in art]
            frases.append(f'<b>Atenção — {rotulo(r)}</b>: ' + ' | '.join(itens))
    frases.append('Comparações entre coletas são descritivas (um único voluntário, sem teste estatístico entre sessões). '
                  'Diferenças pequenas podem estar dentro da variabilidade dia a dia/teste-reteste do EMG.')
    return frases


# ------------------------------------------------------------------ HTML / PDF
def monta_html_individual(resultados: list, meta: dict | None = None, titulo: str | None = None) -> str:
    meta = meta or {}
    resultados = ordena(resultados)
    r0 = resultados[0]
    p = fn.Parametros(**r0.parametros)
    ue = p.unidade_emg
    agora = datetime.now().strftime('%d/%m/%Y %H:%M')
    lados = []
    for r in resultados:
        for l in r.forca:
            if l not in lados:
                lados.append(l)

    mec_html = ''.join(
        f'<h3>Lado {fn.NOME_LADO[l] or "único"}</h3>' + _tabela(tabela_mecanica(resultados, l), 'Variável')
        for l in lados)
    emg_html = ''.join(f'<h3>{html.escape(rot.format(ue=ue))}</h3>' + _tabela(tabela_emg(resultados, k), 'Músculo', casas)
                       for k, rot, casas in VARS_EMG)
    emg_html += ('<p class="nota">Queda pela regressão = inclinação normalizada × duração do patamar (usa todas as '
                 'janelas). Queda pelos extremos = média dos últimos vs. primeiros '
                 f'{p.extremos_s:g} s — mais sensível a poucas janelas ruidosas. Quando as duas divergem muito, '
                 'prefira a da regressão e confira a Figura 3.</p>')
    assim = tabela_assimetria(resultados)
    assim_html = (_tabela(assim, 'Grupo muscular') + '<p class="nota">Diferença D−E da queda da MDF (pontos percentuais); '
                  'negativo = queda maior no lado direito.</p>') if len(assim) else '<p>Sem pares D/E.</p>'
    qualidade = ''
    for r in resultados:
        av = ''.join(f'<div class="aviso">{html.escape(a)}</div>' for a in r.avisos) or \
            '<div class="ok">Sem avisos.</div>'
        qualidade += f'<h3>{html.escape(rotulo(r))} ({r.chave})</h3>{av}'

    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório individual de fadiga — voluntário {html.escape(r0.id)}</title><style>{CSS}</style></head><body>
<h1>{html.escape(titulo or 'Relatório individual de fadiga')} — voluntário {html.escape(r0.id)}</h1>
<div class="sub">Isometria sustentada · {len(resultados)} coleta(s) · gerado em {agora} · versão {fn.APP_VERSION}
{(' · avaliador: ' + html.escape(meta['avaliador'])) if meta.get('avaliador') else ''}</div>

<h2>1. Coletas analisadas</h2>
{_tabela(tabela_sessoes(resultados), 'Coleta')}
{''.join(f'<div class="aviso">{html.escape(m)}</div>' for m in meta.get('integridade', [])
         if f'Voluntário {r0.id}' in m or any(r.chave in m for r in resultados))}

<h2>2. Síntese</h2>
{''.join(f'<p>{f}</p>' for f in sintese(resultados))}
<h3>Tendência em cada coleta</h3>
{_tabela(tabela_padrao(resultados), 'Músculo')}
<p class="nota">↓/↑ = inclinação ao longo do patamar com p &lt; 0,05 (p corrigido pela sobreposição das janelas);
= = sem tendência significativa. "Padrão clássico" de fadiga = MDF cai e RMS sobe ao mesmo tempo.</p>

<h2>3. Fadiga mecânica (células de carga)</h2>
{mec_html}
<p class="nota">Inicial/final = média dos primeiros/últimos {p.extremos_s:g} s do patamar. CV sem tendência = variabilidade
após remover a reta de regressão (estabilidade da força). Overshoot = pico inicial acima da média do patamar.</p>
{_figura(fig_forca_sessoes(resultados), 'Figura 1. Força tarada (média móvel) de cada lado nas coletas, alinhada no onset; pontilhado = início do patamar.')}

<h2>4. Fadiga eletromiográfica</h2>
{emg_html}
{_figura(fig_barras_sessoes(resultados), 'Figura 2. Principais índices de fadiga por músculo e coleta.')}
{_figura(fig_tendencia_sessoes(resultados, 'MDF_Hz', 'MDF (Hz)'), 'Figura 3. MDF por janela no patamar e retas de regressão de cada coleta (tracejado = Pré, contínuo = Pós).')}
{_figura(fig_tendencia_sessoes(resultados, 'RMS', f'RMS ({ue})'), 'Figura 4. RMS por janela no patamar e retas de regressão de cada coleta.')}
{_figura(fig_tendencia_sessoes(resultados, 'ENM', f'ENM (força/RMS)'), 'Figura 5. Eficiência neuromuscular (força do lado ÷ RMS) por janela.')}

<h2>5. Assimetria bilateral</h2>
{assim_html}

<h2>6. Controle de qualidade</h2>
{qualidade}

<h2>7. Métodos</h2>
{_tabela_txt([(f'{g} — {n}', v) for g, n, v in fn.descricao_parametros(p)])}
<p class="nota">ENM = força média do lado na janela ÷ RMS do músculo (contribuição relativa, não eficiência
intrínseca do músculo). Relatório gerado automaticamente: confira as figuras antes de interpretar.</p>
</body></html>"""


def nome_pdf_individual(resultados) -> str:
    conds = sorted({r.chave.split('_', 1)[1][0] for r in resultados})
    return f'{resultados[0].id}_relatorio_individual_fadiga_{"".join(conds)}.pdf'


def gera_pdf_individual(resultados, meta=None, titulo=None) -> bytes:
    from fadiga_relatorio import html_para_pdf
    return html_para_pdf(monta_html_individual(resultados, meta, titulo))


def relatorios_por_dia(resultados: list) -> dict:
    """{condição: [resultados do dia]} — um relatório por dia (condição)."""
    dias = {}
    for r in ordena(resultados):
        dias.setdefault(r.condicao, []).append(r)
    return dias


def nome_pdf_dia(r, condicao) -> str:
    sem_acento = {'Intervenção': 'Intervencao'}.get(condicao, condicao)
    return f'{r.id}_relatorio_dia_{sem_acento}.pdf'

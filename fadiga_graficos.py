"""Figuras (matplotlib) da análise de fadiga — usadas na tela e no relatório PDF."""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CORES_LADO = {'Dir': '#1f77b4', 'Esq': '#d62728', '': '#2f6f4f'}


def _passo(n, alvo=15000):
    return max(1, n // alvo)


def _grade(n, ncols=2, altura=2.3, sharex=True):
    ncols = 1 if n == 1 else ncols
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, max(2.6, altura * nrows)), sharex=sharex, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n], ncols


def fig_forca(res):
    """Força tarada de cada lado: média móvel, faixa média ± k·DP, onset e início do patamar."""
    p = res.parametros
    fs, k, uf = p['fs'], p['k_dp'], p['unidade_forca']
    lados = list(res.forca)
    fig, axes = plt.subplots(len(lados), 1, figsize=(10, 2.9 * len(lados)), sharex=True, squeeze=False)
    passo = _passo(len(res.t))
    nomes = list(res.df_forca.columns)
    for ax, lado, nome in zip(axes[:, 0], lados, nomes):
        r = res.forca[lado]
        cor = CORES_LADO.get(lado, '#2f6f4f')
        ax.plot(res.t[::passo], r['f'][::passo], color='0.7', lw=0.6)
        ax.plot(res.t[::passo], r['fm'][::passo], color=cor, lw=1.2)
        t_ini, t_fim = r['i_ini'] / fs, r['i_fim'] / fs
        ax.fill_between([t_ini, t_fim], r['media'] - k * r['dp'], r['media'] + k * r['dp'],
                        color='green', alpha=0.15, lw=0)
        ax.axhline(r['media'], xmin=0, xmax=1, color='green', lw=0.6, ls='-', alpha=0.6)
        ax.axvline(r['i_onset'] / fs, color='k', ls=':', lw=1)
        ax.axvline(t_ini, color='red' if not r['estabilizou'] else 'k', ls='--', lw=1.2)
        if r.get('i_estab_auto') is not None and r['i_estab_auto'] != r['i_ini']:
            ax.axvline(r['i_estab_auto'] / fs, color='gray', ls='--', lw=0.8)
        ax.plot(r['i_pico'] / fs, r['f'][r['i_pico']], 'o', color=cor, mec='k', ms=5)
        status = 'estabilizou' if r['estabilizou'] else 'NÃO ESTABILIZOU'
        ax.set_title(f"{nome} — início do patamar {t_ini:.2f} s ({r['modo']}) | média "
                     f"{r['media']:.2f} ± {r['dp']:.2f} {uf}, CV {r['cv']:.1f}% | {status}", fontsize=8.5,
                     color='k' if r['estabilizou'] else 'red')
        ax.set_ylabel(f'Força tarada ({uf})')
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel('Tempo (s)')
    fig.legend(handles=[Line2D([], [], color='0.7', label='Força filtrada'),
                        Line2D([], [], color='k', label=f'Média móvel ({p["media_movel_s"]:g} s)'),
                        Patch(color='green', alpha=0.25, label=f'Média ± {k:g} DP (patamar)'),
                        Line2D([], [], color='k', ls=':', label='Onset'),
                        Line2D([], [], color='k', ls='--', label='Início do patamar'),
                        Line2D([], [], color='k', marker='o', ls='', label='Pico inicial')],
               ncol=6, fontsize=7.5, loc='lower center', frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def fig_emg(res):
    """EMG filtrado de cada músculo; verde = patamar analisado; vermelho = janelas excluídas por artefato."""
    p = res.parametros
    fs, ue = p['fs'], p['unidade_emg']
    cods = list(res.emg)
    fig, axes, ncols = _grade(len(cods))
    passo = _passo(len(res.t))
    for i, (ax, c) in enumerate(zip(axes, cods)):
        e = res.emg[c]
        ax.plot(res.t[::passo], e['xf'][::passo], color=CORES_LADO.get(e['lado'], '#2f6f4f'), lw=0.4)
        ax.axvspan(e['i_ini'] / fs, e['i_fim'] / fs, color='green', alpha=0.08)
        jan = e['janelas']
        if len(jan):
            for _, w in jan[~jan['valida'] | (jan['motivo_artefato'] != '')].iterrows():
                ax.axvspan(w['t_ini_s'], w['t_ini_s'] + p['janela_espectral_s'], color='red', alpha=0.18, lw=0)
        ax.set_title(res.df_emg.index[i], fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=True)
        if i % ncols == 0:
            ax.set_ylabel(f'EMG ({ue})')
    for ax in axes[-ncols:]:
        ax.set_xlabel('Tempo (s)')
    fig.legend(handles=[Patch(color='green', alpha=0.2, label='Patamar analisado'),
                        Patch(color='red', alpha=0.3, label='Janela com artefato')],
               ncol=2, fontsize=7.5, loc='lower center', frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def fig_tendencia(res, coluna='MDF_Hz', rotulo='MDF (Hz)'):
    """Variável por janela vs. tempo no patamar, com a reta de regressão (só janelas válidas)."""
    cods = list(res.emg)
    fig, axes, ncols = _grade(len(cods), altura=2.2, sharex=False)
    for i, (ax, c) in enumerate(zip(axes, cods)):
        e = res.emg[c]
        jan = e['janelas']
        cor = CORES_LADO.get(e['lado'], '#2f6f4f')
        if len(jan):
            t0 = e['i_ini'] / res.parametros['fs']
            t = jan['t_centro_s'] - t0
            ok = jan['valida']
            ax.plot(t[ok], jan.loc[ok, coluna], 'o', color=cor, ms=3.5)
            ax.plot(t[~ok], jan.loc[~ok, coluna], 'x', color='0.5', ms=4)
            if ok.sum() >= 3:
                b, a = np.polyfit(t[ok], jan.loc[ok, coluna], 1)
                xx = np.array([t[ok].min(), t[ok].max()])
                ax.plot(xx, a + b * xx, color='k', lw=1)
                ax.text(0.98, 0.95, f'inclinação {b:+.3f}/s', transform=ax.transAxes, ha='right', va='top',
                        fontsize=7.5, bbox=dict(fc='w', ec='none', alpha=0.8))
        ax.set_title(res.df_emg.index[i], fontsize=8.5)
        ax.grid(alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel(rotulo)
    for ax in axes[-ncols:]:
        ax.set_xlabel('Tempo desde o início do patamar (s)')
    fig.legend(handles=[Line2D([], [], color='k', marker='o', ls='', ms=4, label='Janela válida'),
                        Line2D([], [], color='0.5', marker='x', ls='', label='Excluída (artefato)'),
                        Line2D([], [], color='k', label='Regressão linear')],
               ncol=3, fontsize=7.5, loc='lower center', frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ==============================================================================
# FIGURAS DO RELATÓRIO INDIVIDUAL (várias coletas do mesmo voluntário)
# ==============================================================================
CORES_SESSAO = {'C1': '#8FB3D9', 'C2': '#1F4E79', 'I1': '#F2B880', 'I2': '#C0581C'}
ESTILO_SESSAO = {'1': '--', '2': '-'}


def _cor_sessao(r):
    s = r.chave.split('_', 1)[1]
    return CORES_SESSAO.get(s, None), ESTILO_SESSAO.get(r.coleta, '-')


def rotulo_sessao(r):
    return f'{r.condicao} {r.momento}'


def fig_forca_sessoes(resultados):
    """Força tarada (média móvel) de cada lado, sessões sobrepostas e alinhadas no onset."""
    lados = []
    for r in resultados:
        for l in r.forca:
            if l not in lados:
                lados.append(l)
    fig, axes = plt.subplots(len(lados), 1, figsize=(10, 2.8 * len(lados)), sharex=True, squeeze=False)
    uf = resultados[0].parametros['unidade_forca']
    for ax, lado in zip(axes[:, 0], lados):
        for r in resultados:
            if lado not in r.forca:
                continue
            fr, fs = r.forca[lado], r.parametros['fs']
            cor, ls = _cor_sessao(r)
            t = (np.arange(len(fr['fm'])) - fr['i_onset']) / fs
            k = _passo(len(t))
            ax.plot(t[::k], fr['fm'][::k], color=cor, ls=ls, lw=1.3, label=rotulo_sessao(r))
            ax.axvline((fr['i_ini'] - fr['i_onset']) / fs, color=cor, ls=':', lw=0.9)
        ax.set_title(f'Lado {({"Dir": "direito", "Esq": "esquerdo"}).get(lado, lado or "único")}', fontsize=9)
        ax.set_ylabel(f'Força tarada ({uf})')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, ncol=4, loc='lower right')
    axes[-1, 0].set_xlabel('Tempo desde o onset (s) — pontilhado: início do patamar')
    fig.tight_layout()
    return fig


def fig_tendencia_sessoes(resultados, coluna='MDF_Hz', rotulo='MDF (Hz)'):
    """Variável por janela vs. tempo no patamar, por músculo; sessões sobrepostas com retas de regressão."""
    musculos = []
    for r in resultados:
        for i, c in enumerate(r.emg):
            nome = r.df_emg.index[i]
            if (c, nome) not in musculos:
                musculos.append((c, nome))
    fig, axes, ncols = _grade(len(musculos), altura=2.3, sharex=False)
    for i, (ax, (c, nome)) in enumerate(zip(axes, musculos)):
        for r in resultados:
            if c not in r.emg or not len(r.emg[c]['janelas']):
                continue
            e = r.emg[c]
            jan = e['janelas']
            ok = jan['valida']
            t = jan['t_centro_s'] - e['i_ini'] / r.parametros['fs']
            cor, ls = _cor_sessao(r)
            ax.plot(t[ok], jan.loc[ok, coluna], '.', color=cor, ms=2.5, alpha=0.45)
            if ok.sum() >= 3:
                b, a = np.polyfit(t[ok], jan.loc[ok, coluna], 1)
                xx = np.array([t[ok].min(), t[ok].max()])
                ax.plot(xx, a + b * xx, color=cor, ls=ls, lw=1.8, label=f'{rotulo_sessao(r)}: {b:+.3f}/s')
        ax.set_title(nome, fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc='best')
        if i % ncols == 0:
            ax.set_ylabel(rotulo)
    for ax in axes[-ncols:]:
        ax.set_xlabel('Tempo desde o início do patamar (s)')
    fig.tight_layout()
    return fig


def fig_barras_sessoes(resultados, chaves=(('MDF_slope', 'Inclinação da MDF (Hz/s)'),
                                           ('Queda_MDF', 'Queda da MDF (%)'),
                                           ('RMS_slope', 'Inclinação do RMS (/s)'))):
    """Barras agrupadas por músculo (uma barra por sessão) para as principais variáveis de fadiga."""
    musculos, cods = [], []
    for r in resultados:
        for i, c in enumerate(r.emg):
            if c not in cods:
                cods.append(c)
                musculos.append(r.df_emg.index[i])
    fig, axes = plt.subplots(len(chaves), 1, figsize=(10, 2.6 * len(chaves)), sharex=True)
    axes = np.atleast_1d(axes)
    larg = 0.8 / max(1, len(resultados))
    x = np.arange(len(cods))
    for ax, (k, rot) in zip(axes, chaves):
        for j, r in enumerate(resultados):
            vals = [r.emg[c]['resumo'][k] if c in r.emg else np.nan for c in cods]
            cor, _ = _cor_sessao(r)
            ax.bar(x + (j - (len(resultados) - 1) / 2) * larg, vals, width=larg, color=cor, label=rotulo_sessao(r),
                   edgecolor='k', lw=0.3)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_ylabel(rot, fontsize=8)
        ax.grid(alpha=0.3, axis='y')
    axes[0].legend(fontsize=7.5, ncol=4, loc='lower left')
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([m.replace(' (', '\n(') for m in musculos], fontsize=7.5)
    fig.tight_layout()
    return fig

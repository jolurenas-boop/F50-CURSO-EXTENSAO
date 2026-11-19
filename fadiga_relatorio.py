"""Relatório de fadiga em PDF (HTML + CSS -> WeasyPrint).

A estrutura é compatível com o portal de fusão (app_streamlit_master.py):
  * "Arquivo analisado" e "Duração do registro ... s" na identificação;
  * UMA única tabela com as colunas "Músculo" + "MDF inicial/final/slope", "Queda MDF", "RMS slope", "R²", "p-valor";
  * seção "4. Controle de qualidade" seguida de "Auditoria espectral".
Nome sugerido do PDF: <ID>_F50<C|I><PRE|POS>.pdf (padrão reconhecido pelo portal).
"""
import base64
import html
import io
from datetime import datetime

import matplotlib.pyplot as plt

import fadiga_nucleo as fn
from fadiga_graficos import fig_forca, fig_emg, fig_tendencia

CSS = """
@page { size: A4; margin: 16mm 14mm 17mm 14mm;
  @bottom-left  { content: "Relatório de fadiga sEMG — isometria sustentada"; font-size: 7.5pt; color: #777; }
  @bottom-right { content: "Página " counter(page) " de " counter(pages); font-size: 7.5pt; color: #777; } }
body { font-family: "DejaVu Sans", "Helvetica", "Arial", sans-serif; font-size: 8.6pt; color: #1b1f24; line-height: 1.36; }
h1 { font-size: 16pt; margin: 0 0 2pt 0; color: #0f2a4a; }
.sub { color: #55606d; font-size: 9pt; margin-bottom: 9pt; }
h2 { font-size: 11pt; color: #0f2a4a; border-bottom: 1.2pt solid #0f2a4a; padding-bottom: 2pt; margin: 14pt 0 6pt 0; page-break-after: avoid; }
h3 { font-size: 9.3pt; margin: 8pt 0 3pt 0; color: #22364f; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 7pt 0; page-break-inside: avoid; }
th, td { border: 0.5pt solid #c5ccd6; padding: 2.2pt 4pt; font-size: 7.6pt; }
th { background: #eaf0f7; text-align: center; font-weight: bold; }
td { text-align: right; }
td:first-child, th:first-child { text-align: left; }
table.txt td { text-align: left; }
table.txt td:first-child { width: 32%; font-weight: bold; background: #f6f8fb; }
figure { margin: 5pt 0 8pt 0; page-break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-size: 7.4pt; color: #55606d; }
.aviso { background: #fff6e0; border-left: 3pt solid #e0a000; padding: 3pt 7pt; margin: 2pt 0; font-size: 7.8pt; }
.ok { background: #eaf7ee; border-left: 3pt solid #2f9e5b; padding: 3pt 7pt; font-size: 7.8pt; }
.mono { font-family: "DejaVu Sans Mono", monospace; font-size: 6.6pt; word-break: break-all; }
.nota { font-size: 7.4pt; color: #55606d; }
"""


def _b64(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _tabela(df, cabecalho_indice=' ', casas=None):
    out = fn.formata_tabela(df, casas)
    out.insert(0, cabecalho_indice, [html.escape(str(i)) for i in df.index])
    out.columns = [html.escape(str(c)) for c in out.columns]
    return out.to_html(escape=False, index=False, border=0)


def _tabela_txt(linhas):
    corpo = ''.join(f'<tr><td>{html.escape(str(a))}</td><td>{html.escape(str(b))}</td></tr>' for a, b in linhas)
    return f'<table class="txt">{corpo}</table>'


def _figura(fig, legenda):
    return (f'<figure><img src="data:image/png;base64,{_b64(fig)}">'
            f'<figcaption>{html.escape(legenda)}</figcaption></figure>')


def nome_pdf(res, prefixo='F50') -> str:
    cond = res.chave.split('_', 1)[1][0]
    mom = {'Pré': 'PRE', 'Pós': 'POS'}.get(res.momento, f'C{res.coleta}')
    return f'{res.id}_{prefixo}{cond}{mom}.pdf'


def monta_html(res, meta: dict | None = None) -> str:
    meta = meta or {}
    p = fn.Parametros(**res.parametros)
    agora = datetime.now().strftime('%d/%m/%Y %H:%M')
    arquivos = sorted(v[0] for v in res.arquivos.values())
    # "Arquivo analisado" numa única linha (o portal lê a primeira linha após o rótulo)
    ident = [('Arquivo analisado', f'{res.chave} — {len(arquivos)} canais ({", ".join(sorted(res.arquivos))})'),
             ('Voluntário', res.id), ('Condição', res.condicao), ('Momento', f'{res.momento} (coleta {res.coleta})'),
             ('Avaliador', meta.get('avaliador') or '—'), ('Observações', meta.get('observacoes') or '—'),
             ('Duração do registro', f'{fn.fmt(res.duracao_s, 2)} s ({res.n_amostras} amostras a {p.fs:g} Hz)')]

    avisos_html = (''.join(f'<div class="aviso">{html.escape(a)}</div>' for a in res.avisos) if res.avisos
                   else '<div class="ok">Nenhum aviso de qualidade foi gerado nas verificações automáticas.</div>')
    assim_html = ''
    if res.df_assimetria is not None and len(res.df_assimetria):
        assim_html = ('<h3>Assimetria bilateral (direito − esquerdo)</h3>'
                      + _tabela(res.df_assimetria, 'Grupo muscular')
                      + '<p class="nota">Valores negativos de "Diferença D−E da queda" indicam queda maior no lado '
                        'direito.</p>')
    tab_atenu = _tabela(res.atenuacao_notch.to_frame(), 'Canal') if len(res.atenuacao_notch) else ''
    arq_html = _tabela_txt([(c, f'{n} — SHA-256 {h}') for c, (n, h) in sorted(res.arquivos.items())])

    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório de fadiga {html.escape(res.chave)}</title><style>{CSS}</style></head><body>
<h1>Relatório de fadiga sEMG — isometria sustentada</h1>
<div class="sub">Gerado em {agora} — versão da aplicação {fn.APP_VERSION}</div>

<h2>1. Identificação</h2>
{_tabela_txt(ident)}

<h2>2. Força (células de carga)</h2>
{_tabela(res.df_forca, 'Variável')}
<p class="nota">Tara = média do repouso antes do onset. O patamar começa quando a média móvel da força entra e
permanece na faixa média ± {p.k_dp:g} DP por pelo menos {p.dur_min_estavel_s:g} s, separadamente em cada lado.
Lados que não estabilizam usam onset + {p.inicio_reserva_s:g} s e são sinalizados.</p>
{_figura(fig_forca(res), 'Figura 1. Força tarada, faixa média ± k·DP do patamar, onset e início do patamar de cada lado.')}

<h2>3. Fadiga eletromiográfica</h2>
{_tabela(res.df_emg, 'Músculo')}
<p class="nota">Janelas de {p.janela_espectral_s:g} s com {p.sobreposicao:.0%} de sobreposição, dentro do patamar
do lado correspondente. Inicial/final = média das janelas válidas nos primeiros/últimos {p.extremos_s:g} s.
Slope = inclinação da regressão linear vs. tempo. Queda MDF = (final − inicial)/inicial × 100.</p>
<h3>Métricas complementares</h3>
{_tabela(res.df_emg_extra, 'Canal')}
{assim_html}
{_figura(fig_tendencia(res, 'MDF_Hz', 'MDF (Hz)'), 'Figura 2. Frequência mediana por janela no patamar, com regressão linear (x = janela excluída).')}
{_figura(fig_tendencia(res, 'RMS', f'RMS ({p.unidade_emg})'), 'Figura 3. RMS por janela no patamar, com regressão linear.')}
{_figura(fig_emg(res), 'Figura 4. EMG filtrado; verde = patamar analisado; vermelho = janelas com artefato.')}

<h2>4. Controle de qualidade</h2>
{avisos_html}
<h3>Auditoria espectral</h3>
{tab_atenu}

<h2>5. Parâmetros e rastreabilidade</h2>
{_tabela_txt([(f'{g} — {n}', v) for g, n, v in fn.descricao_parametros(p)])}
<h3>Arquivos de entrada</h3>
<div class="mono">{arq_html}</div>
<h3>Versões de software</h3>
{_tabela_txt(list(fn.versoes().items()))}
<p class="nota">Relatório gerado automaticamente. Confira o início do patamar (Figura 1) e as janelas excluídas
(Figuras 2 e 4) antes de interpretar.</p>
</body></html>"""


def html_para_pdf(doc: str) -> bytes:
    """HTML -> PDF. Tenta WeasyPrint; se faltar, usa um Chrome/Chromium headless instalado; senão, RuntimeError."""
    erros = []
    try:
        from weasyprint import HTML
        return HTML(string=doc).write_pdf()
    except Exception as e:  # noqa: BLE001
        erros.append(f'WeasyPrint: {e}')
    import os
    import shutil
    import subprocess
    import tempfile
    for exe in ('google-chrome', 'chromium', 'chromium-browser', 'chrome', 'msedge'):
        caminho = shutil.which(exe)
        if not caminho:
            continue
        with tempfile.TemporaryDirectory() as d:
            h, pdf = os.path.join(d, 'r.html'), os.path.join(d, 'r.pdf')
            with open(h, 'w', encoding='utf-8') as f:
                f.write(doc)
            subprocess.run([caminho, '--headless=new', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                            f'--print-to-pdf={pdf}', h], capture_output=True, timeout=180)
            if os.path.exists(pdf) and os.path.getsize(pdf) > 0:
                with open(pdf, 'rb') as f:
                    return f.read()
        erros.append(f'{exe}: falhou')
    raise RuntimeError('Não foi possível gerar PDF (instale o WeasyPrint com as bibliotecas Pango — no Streamlit '
                       'Cloud, use o packages.txt — ou tenha o Chrome/Chromium instalado). ' + ' | '.join(erros))


def gera_pdf(res, meta: dict | None = None) -> bytes:
    return html_para_pdf(monta_html(res, meta))

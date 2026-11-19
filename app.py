"""Aplicação Streamlit — FADIGA sEMG em isometria sustentada, coleta única.

Arquivo único com 7 colunas (padrão): Tempo | Bíceps D | Bíceps E | Tríceps D | Tríceps E | Carga D | Carga E.

Executar:  streamlit run app.py
"""
import io
import re
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import coleta_unica as cu
import fadiga_individual as fi
import fadiga_nucleo as fn
from fadiga_graficos import fig_forca, fig_emg, fig_tendencia

st.set_page_config(page_title='Fadiga sEMG — bíceps e tríceps', layout='wide')

SEP = {'Automático': 'auto', 'Tabulação': 'tab', 'Ponto e vírgula': 'ponto_virgula', 'Espaço(s)': 'espaco',
       'Vírgula': 'virgula'}
DEC = {'Automático': 'auto', 'Ponto (.)': '.', 'Vírgula (,)': ','}
SEM_LADO = '— (único)'
LADO_UI = {'Dir': 'Dir', 'Esq': 'Esq', '': SEM_LADO}
LADO_UI_INV = {v: k for k, v in LADO_UI.items()}


def mostra_figura(fig):
    st.pyplot(fig)
    plt.close(fig)


def csv_ptbr(df, index=True) -> bytes:
    return ('﻿' + df.to_csv(sep=';', decimal=',', index=index)).encode('utf-8')


@st.cache_data(show_spinner='Processando…', max_entries=16)
def roda(conteudo: bytes, nome: str, id_vol: str, col_tempo: int, colunas: dict, mapa: dict, fmt_arq: tuple,
         params: dict, manual: dict, excluir: tuple):
    teste, _ = cu.monta_teste(conteudo, nome, id_vol, col_tempo, colunas, *fmt_arq)
    return cu.processa(teste, mapa, fn.Parametros(**params), manual, list(excluir))


ss = st.session_state
ss.setdefault('manual', {})
ss.setdefault('excluir', [])
ss.setdefault('cfg_ver', 0)

# ==============================================================================
# BARRA LATERAL
# ==============================================================================
with st.sidebar:
    st.header('Arquivo')
    arquivo = st.file_uploader('Registro (.txt) — 30 s, todas as colunas', type=['txt', 'csv', 'tsv', 'dat'])
    _m = re.match(r'\s*(\d+)(?=[_\s-])', arquivo.name) if arquivo else None      # ex.: 001_F50.txt -> 001
    id_padrao = _m.group(1).zfill(3) if _m else ''
    id_vol = st.text_input('Voluntário / ID', value=id_padrao)
    with st.expander('Formato do arquivo'):
        sep = st.selectbox('Separador de colunas', list(SEP))
        dec = st.selectbox('Separador decimal', list(DEC))
        n_cab = st.number_input('Linhas de cabeçalho a ignorar', 0, 1000, 0, 1)
    with st.expander('Colunas e canais', expanded=False):
        if st.button('Restaurar padrão (Tempo, BB D/E, TB D/E, Carga D/E)'):
            ss['cfg_ver'] += 1
        col_tempo = st.number_input('Coluna do Tempo', 1, 200, 1, 1, key=f't_{ss["cfg_ver"]}')
        df_canais = st.data_editor(
            pd.DataFrame([{'Código': c, 'Nome': n, 'Lado': LADO_UI[l], 'Tipo': t, 'Coluna': cu.COLUNAS_PADRAO[c]}
                          for c, (n, l, t) in fn.CANAIS_PADRAO.items()]),
            key=f'canais_{ss["cfg_ver"]}', num_rows='dynamic', hide_index=True,
            column_config={'Código': st.column_config.TextColumn(required=True),
                           'Nome': st.column_config.TextColumn(required=True),
                           'Lado': st.column_config.SelectboxColumn(options=list(LADO_UI.values()), required=True),
                           'Tipo': st.column_config.SelectboxColumn(options=list(fn.TIPOS_VALIDOS), required=True),
                           'Coluna': st.column_config.NumberColumn(min_value=1, max_value=200, step=1,
                                                                   required=True)})

    st.header('Parâmetros')
    with st.expander('Aquisição e filtros'):
        fs = st.number_input('Frequência de amostragem (Hz)', 100.0, 50000.0, 2000.0, 100.0)
        u_emg = st.text_input('Unidade do EMG', 'µV')
        u_f = st.text_input('Unidade da força', 'kgf')
        b_lo = st.number_input('EMG passa-alta (Hz)', 1.0, 200.0, 20.0, 1.0)
        b_hi = st.number_input('EMG passa-baixa (Hz)', 50.0, 2000.0, 450.0, 10.0)
        notch = st.checkbox('Notch', True)
        f_notch = st.number_input('Frequência do notch (Hz)', 1.0, 1000.0, 60.0, 1.0)
        fc_f = st.number_input('Força: passa-baixa (Hz)', 1.0, 100.0, 20.0, 1.0)
    with st.expander('Estabilização da força (por lado)', expanded=True):
        k_dp = st.number_input('Faixa: média ± k·DP — k =', 0.5, 5.0, 2.0, 0.25)
        dur_min = st.number_input('Permanecer na faixa por (s)', 0.25, 10.0, 2.0, 0.25)
        mm = st.number_input('Média móvel da força (s)', 0.05, 3.0, 0.5, 0.05)
        cv_max = st.number_input('Verificação: CV máximo no patamar (%)', 1.0, 50.0, 10.0, 1.0)
        der_max = st.number_input('Verificação: deriva máxima (%)', 1.0, 100.0, 20.0, 1.0)
        reserva = st.number_input('Se não estabilizar: início = onset + (s)', 0.0, 20.0, 3.0, 0.5)
        fim_ign = st.number_input('Descartar o fim do registro (s)', 0.0, 10.0, 0.0, 0.25)
        comum = st.checkbox('Mesma janela para os dois lados (início mais tardio)', False)
    with st.expander('Janelas espectrais', expanded=True):
        jan = st.number_input('Duração da janela (s)', 0.1, 5.0, 0.5, 0.05)
        sob = st.number_input('Sobreposição (%)', 0, 90, 50, 5)
        seg = st.number_input('Segmento do Welch (s)', 0.05, 5.0, 0.25, 0.05)
        ext = st.number_input('Inicial/final: média dos primeiros/últimos (s)', 0.25, 10.0, 2.0, 0.25)
        st.caption(f'Janelas: 0–{jan:g}; {jan * (1 - sob / 100):g}–{jan * (1 - sob / 100) + jan:g}; … s')
    with st.expander('Artefatos'):
        excl = st.checkbox('Excluir janelas com artefato', True)
        fe = st.number_input('Limite do conversor (± unidade do EMG)', 0.0, 1e6, 5000.0, 100.0, help='0 = não verificar')
        pico = st.number_input('Pico máximo (nº de DP robustos)', 3.0, 100.0, 15.0, 1.0)
        rmsr = st.number_input('RMS máximo (× mediana das janelas)', 1.2, 20.0, 2.5, 0.1)
    with st.expander('Identificação no relatório'):
        avaliador = st.text_input('Avaliador')
        protocolo = st.text_input('Protocolo', 'Fadiga isométrica sustentada (30 s)')
        observ = st.text_area('Observações', height=70)

# ------------------------------------------------------------------ configuração
try:
    mapa, colunas = {}, {}
    for _, r in df_canais.iterrows():
        cod = str(r.get('Código') or '').strip().upper()
        if not cod or cod == 'NAN':
            continue
        mapa[cod] = (str(r['Nome']).strip(), LADO_UI_INV.get(r['Lado'], ''), str(r['Tipo']))
        colunas[cod] = int(r['Coluna'])
    fn.valida_mapa(mapa)
    params = fn.Parametros(
        fs=float(fs), unidade_emg=u_emg or 'µV', unidade_forca=u_f or 'kgf', banda_emg=(float(b_lo), float(b_hi)),
        usar_notch=bool(notch), f_notch=float(f_notch), fc_forca=float(fc_f), k_dp=float(k_dp),
        media_movel_s=float(mm), dur_min_estavel_s=float(dur_min), inicio_reserva_s=float(reserva),
        fim_ignorado_s=float(fim_ign), janela_comum=bool(comum), cv_max_pct=float(cv_max),
        deriva_max_pct=float(der_max), janela_espectral_s=float(jan), sobreposicao=float(sob) / 100,
        welch_segmento_s=float(seg), extremos_s=float(ext), excluir_artefatos=bool(excl), fundo_escala=float(fe),
        pico_max_dp=float(pico), rms_max_razao=float(rmsr))
    fn.valida_parametros(params)
except Exception as e:  # noqa: BLE001
    st.error(f'Configuração inválida: {e}')
    st.stop()

# ==============================================================================
# CORPO
# ==============================================================================
st.title('Fadiga sEMG — bíceps e tríceps braquial (coleta única)')
layout = ' | '.join([f'Col. {col_tempo}: Tempo'] + [f'Col. {colunas[c]}: {fn.nome_exibicao(c, mapa)}'
                                                    for c in sorted(colunas, key=colunas.get)])
if arquivo is None:
    st.info('Envie o registro de 30 s na barra lateral.')
    st.markdown(f'**Layout esperado:** `{layout}`')
    st.markdown('O app tara a força, encontra o patamar de cada lado (média ± 2 DP, com verificação de CV e deriva), '
                'analisa o EMG em janelas de 0,5 s com 50% de sobreposição (MDF, MNF, RMS, ENM), exclui janelas com '
                'artefato e gera o relatório em PDF.')
    st.stop()

conteudo = arquivo.getvalue()
fmt_arq = (SEP[sep], DEC[dec], int(n_cab))
with st.expander('Prévia do arquivo (confira o número de cada coluna)'):
    try:
        st.dataframe(cu.previa(conteudo, *fmt_arq), hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f'Não foi possível montar a prévia: {e}')
    st.caption(f'Layout configurado: {layout}')

try:
    r = roda(conteudo, arquivo.name, id_vol or 'sem_id', int(col_tempo), colunas, mapa, fmt_arq, asdict(params),
             ss['manual'], tuple(ss['excluir']))
except Exception as e:  # noqa: BLE001
    st.error(f'Não foi possível analisar: {e}')
    st.stop()

if r.avisos:
    with st.expander(f'Avisos de qualidade ({len(r.avisos)})', expanded=True):
        for a in r.avisos:
            st.warning(a)

aba_res, aba_forca, aba_emg, aba_jan, aba_met = st.tabs(
    ['Resultados e relatório', 'Força (patamar)', 'EMG por janela', 'Janelas', 'Parâmetros'])

with aba_res:
    st.markdown('#### Síntese')
    for f in fi.sintese([r]):
        if 'Comparações entre coletas' not in f:
            st.markdown('- ' + f.replace('<b>', '**').replace('</b>', '**'))
    st.dataframe(fi.tabela_padrao([r]))
    st.markdown('#### Fadiga eletromiográfica')
    st.dataframe(fn.formata_tabela(r.df_emg))
    st.dataframe(fn.formata_tabela(r.df_emg_extra))
    if r.df_assimetria is not None and len(r.df_assimetria):
        st.markdown('**Assimetria bilateral (D − E)**')
        st.dataframe(fn.formata_tabela(r.df_assimetria))
    st.markdown('#### Fadiga mecânica')
    st.dataframe(fn.formata_tabela(r.df_forca))

    st.markdown('#### Exportar')
    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', arquivo.name.rsplit('.', 1)[0]) or 'registro'
    c1, c2, c3 = st.columns(3)
    c1.download_button('EMG (CSV)', csv_ptbr(pd.concat([r.df_emg, r.df_emg_extra], axis=1)), f'{stem}_emg.csv', 'text/csv')
    c2.download_button('Força (CSV)', csv_ptbr(r.df_forca), f'{stem}_forca.csv', 'text/csv')
    c3.download_button('Janelas (CSV)', csv_ptbr(r.df_janelas, False), f'{stem}_janelas.csv', 'text/csv')
    meta = {'avaliador': avaliador, 'protocolo': protocolo, 'observacoes': observ}
    if st.button('Gerar relatório em PDF', type='primary'):
        with st.spinner('Montando o relatório…'):
            try:
                ss['pdf'] = (cu.nome_pdf(r, arquivo.name), cu.gera_pdf(r, arquivo.name, meta), 'application/pdf')
            except RuntimeError as e:
                st.warning(f'{e} — entregando em HTML (abra e use Imprimir → Salvar como PDF).')
                ss['pdf'] = (cu.nome_pdf(r, arquivo.name).replace('.pdf', '.html'),
                             cu.monta_html(r, arquivo.name, meta).encode('utf-8'), 'text/html')
    if ss.get('pdf'):
        st.download_button(f'Baixar {ss["pdf"][0]}', ss['pdf'][1], ss['pdf'][0], ss['pdf'][2])
        st.caption('Se mudar arquivo ou parâmetros, gere o relatório novamente.')

with aba_forca:
    mostra_figura(fig_forca(r))
    with st.expander('Ajustar o início do patamar à mão (por lado)'):
        novo = {}
        cols = st.columns(len(r.forca))
        for col_ui, (lado, fr) in zip(cols, r.forca.items()):
            v = col_ui.text_input(f'{fn.nome_exibicao(fr["codigo"], mapa)} — início (s)',
                                  value=f'{ss["manual"][lado]:g}' if lado in ss['manual'] else '',
                                  placeholder=f'auto: {fr["i_ini"] / params.fs:.2f}', key=f'man_{lado}')
            if v.strip():
                try:
                    novo[lado] = float(v.replace(',', '.'))
                except ValueError:
                    col_ui.error('Número inválido')
        if st.button('Aplicar início manual'):
            ss['manual'] = novo
            st.rerun()
    with st.expander('Excluir canais (arquivo/eletrodo com problema)'):
        sel = st.multiselect('Canais a excluir', list(mapa), default=ss['excluir'])
        if st.button('Aplicar exclusão'):
            ss['excluir'] = sel
            st.rerun()

with aba_emg:
    t1, t2, t3, t4 = st.tabs(['MDF', 'RMS', 'ENM', 'EMG filtrado'])
    with t1:
        mostra_figura(fig_tendencia(r, 'MDF_Hz', 'MDF (Hz)'))
    with t2:
        mostra_figura(fig_tendencia(r, 'RMS', f'RMS ({params.unidade_emg})'))
    with t3:
        mostra_figura(fig_tendencia(r, 'ENM', f'ENM ({params.unidade_forca}/{params.unidade_emg})'))
    with t4:
        mostra_figura(fig_emg(r))

with aba_jan:
    st.dataframe(r.df_janelas, hide_index=True)

with aba_met:
    st.table([{'Grupo': g, 'Parâmetro': n, 'Valor': v} for g, n, v in fn.descricao_parametros(params)])
    st.json(fn.versoes())

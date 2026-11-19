"""Núcleo de análise de FADIGA isométrica sustentada (ex.: 30 s a 50% da CIVM) — EMG + células de carga.

Entrada: um arquivo por canal (2 colunas: tempo e sinal), nomeado <ID>_<CÓDIGO><C|I><n>.txt
  ex.: 001_PMDC1.txt -> voluntário 001, canal PMD, Controle, coleta 1.
Os canais são agrupados por (ID, condição, coleta) e alinhados pelo tempo.

Etapas por teste:
  1. Força de cada lado: tara (repouso antes do onset), passa-baixa, onset, pico inicial.
  2. Estabilização POR LADO: faixa = média ± k·DP da força no patamar; o patamar começa no primeiro
     instante (após o pico inicial) em que a média móvel da força permanece na faixa por >= dur_min s.
     A média/DP são reestimados a partir do início encontrado (iterativo). Se não houver estabilização,
     o lado é sinalizado e usa-se um início de reserva (onset + X s).
  3. EMG de cada músculo: filtro (notch + passa-banda), janelas no patamar do lado correspondente,
     exclusão de janelas com artefato, MDF/MNF (Welch) e RMS por janela, regressão linear no tempo.

Sem dependência do Streamlit.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import re
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, welch

APP_VERSION = '1.0.0-coleta-unica-bb-tb'
NOME_LADO = {'Dir': 'direito', 'Esq': 'esquerdo', '': ''}
LADOS_VALIDOS = ('Dir', 'Esq', '')
TIPOS_VALIDOS = ('EMG', 'Força')

# Mapa padrão de códigos de canal -> (nome, lado, tipo). Editável na interface.
CANAIS_PADRAO = {
    'BBD': ('Bíceps braquial', 'Dir', 'EMG'), 'BBE': ('Bíceps braquial', 'Esq', 'EMG'),
    'TBD': ('Tríceps braquial', 'Dir', 'EMG'), 'TBE': ('Tríceps braquial', 'Esq', 'EMG'),
    'CCD': ('Célula de carga', 'Dir', 'Força'), 'CCE': ('Célula de carga', 'Esq', 'Força'),
}
COLETAS_PADRAO = {'1': 'Pré', '2': 'Pós'}
CONDICOES = {'C': 'Controle', 'I': 'Intervenção'}

RE_NOME = re.compile(r'^(?P<id>\d+)[_\s-]*(?P<codigo>[A-Za-z]+)(?P<cond>[CIci])(?P<coleta>\d+)$')


def nome_exibicao(codigo: str, mapa: dict) -> str:
    """'PMD' -> 'Peitoral maior direito (PMD)' (formato que o portal de fusão reconhece)."""
    nome, lado, _ = mapa[codigo]
    adj = NOME_LADO[lado]
    if adj and nome.split()[0].lower().endswith('a'):      # concordância simples: "Célula de carga direita"
        adj = adj[:-1] + 'a'
    return f'{nome} {adj} ({codigo})'.replace('  ', ' ')


# ==============================================================================
# PARÂMETROS
# ==============================================================================
@dataclass(frozen=True)
class Parametros:
    fs: float = 2000.0
    verificar_fs_pelo_tempo: bool = True
    # EMG
    unidade_emg: str = 'µV'
    banda_emg: tuple = (20.0, 450.0)
    ordem_emg: int = 4
    usar_notch: bool = True
    f_notch: float = 60.0
    q_notch: float = 30.0
    # força
    unidade_forca: str = 'kgf'
    fc_forca: float = 20.0
    janela_repouso_s: float = 0.5        # tara: média do trecho de repouso antes do onset
    repouso_min_s: float = 0.25          # repouso mais curto que isso não serve para a tara
    janela_pico_inicial_s: float = 3.0   # overshoot = máximo da força nos primeiros X s após o onset
    onset_pct_patamar: float = 0.10      # onset: força > 10% da mediana do 2º terço do registro
    # estabilização (por lado)
    k_dp: float = 2.0                    # faixa = média ± k·DP
    media_movel_s: float = 0.5
    dur_min_estavel_s: float = 2.0
    inicio_reserva_s: float = 3.0        # se não estabilizar: início = onset + este valor
    patamar_min_s: float = 10.0          # patamar mais curto que isso gera aviso
    cv_max_pct: float = 10.0             # verificação: CV da força no patamar acima disso = não estabilizou
    deriva_max_pct: float = 20.0         # verificação: |inclinação| × duração / média acima disso = não estabilizou
    fim_ignorado_s: float = 0.0          # descarta o fim do registro (ex.: relaxamento)
    janela_comum: bool = False           # True = mesmo início (o mais tardio) para todos os músculos
    # janelas espectrais
    janela_espectral_s: float = 0.5      # janelas de 0,5 s ...
    sobreposicao: float = 0.5            # ... com 50% de sobreposição (0–0,5; 0,25–0,75; 0,5–1,0 ...)
    welch_segmento_s: float = 0.25       # segmento do Welch dentro de cada janela (50% de sobreposição)
    extremos_s: float = 2.0              # 'inicial'/'final' = média das janelas nos primeiros/últimos X s
    # artefatos
    excluir_artefatos: bool = True
    fundo_escala: float = 5000.0         # limite do conversor na unidade do EMG (± valor)
    pico_max_dp: float = 15.0            # pico > X DP robustos (MAD) do EMG no patamar
    rms_max_razao: float = 2.5           # RMS da janela > X × mediana das janelas


def valida_parametros(p: Parametros) -> None:
    nyq = p.fs / 2
    lo, hi = p.banda_emg
    if not (0 < lo < hi < nyq):
        raise ValueError(f'Banda do EMG inválida: é preciso 0 < f_baixa < f_alta < fs/2 ({nyq:g} Hz).')
    if p.usar_notch and not (0 < p.f_notch < nyq):
        raise ValueError('Frequência do notch inválida.')
    if not (0 < p.fc_forca < nyq):
        raise ValueError('Corte do filtro de força inválido.')
    if not (0 <= p.sobreposicao < 0.95):
        raise ValueError('A sobreposição deve estar entre 0 e 0,9.')
    if p.janela_espectral_s * p.fs < 64:
        raise ValueError('Janela espectral curta demais.')
    if not (0 < p.welch_segmento_s <= p.janela_espectral_s):
        raise ValueError('O segmento do Welch deve ser positivo e não maior que a janela.')
    if p.extremos_s <= 0:
        raise ValueError('O trecho usado para os valores inicial/final deve ser positivo.')
    if p.k_dp <= 0 or p.dur_min_estavel_s <= 0 or p.media_movel_s <= 0:
        raise ValueError('k, média móvel e duração mínima de estabilidade devem ser positivos.')


# ==============================================================================
# LEITURA E AGRUPAMENTO
# ==============================================================================
def interpreta_nome(nome_arquivo: str) -> dict | None:
    stem = re.sub(r'\.[A-Za-z0-9]+$', '', nome_arquivo.split('/')[-1].split('\\')[-1]).strip()
    m = RE_NOME.match(stem)
    if not m:
        return None
    return {'id': m['id'].zfill(3), 'codigo': m['codigo'].upper(), 'cond': m['cond'].upper(),
            'coleta': str(int(m['coleta']))}


def le_canal(conteudo: bytes) -> np.ndarray:
    """Lê um arquivo de 2 colunas (tempo, sinal). Aceita tab/;/espaço e decimal com vírgula."""
    texto = conteudo.decode('utf-8-sig', errors='replace')
    linhas = [l for l in texto.splitlines() if l.strip()]
    amostra = linhas[:50]
    usa_virgula_decimal = any(',' in l for l in amostra) and not any('.' in l for l in amostra)
    if usa_virgula_decimal:
        texto = texto.replace(',', '.')
    try:                                   # caminho rápido (C engine): tabulação, o formato do equipamento
        df = pd.read_csv(io.StringIO(texto), sep='\t', header=None)
        if df.shape[1] < 2:
            raise ValueError
    except Exception:  # noqa: BLE001
        df = pd.read_csv(io.StringIO(texto), sep=r'[\t; ]+', engine='python', header=None)
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna(axis=0, how='any')
    if df.shape[1] < 2 or len(df) < 100:
        raise ValueError('esperadas 2 colunas numéricas (tempo e sinal) e pelo menos 100 linhas')
    return df.iloc[:, :2].to_numpy(float)


@dataclass
class Teste:
    id: str
    cond: str        # 'C' | 'I'
    coleta: str      # '1', '2', ...
    canais: dict     # código -> (nome_arquivo, sha256, array Nx2)

    @property
    def chave(self) -> str:
        return f'{self.id}_{self.cond}{self.coleta}'


def agrupa(arquivos: dict[str, bytes]) -> tuple[dict[str, Teste], list[str]]:
    """{nome: bytes} -> ({chave: Teste}, [mensagens sobre arquivos não reconhecidos])."""
    testes, problemas = {}, []
    for nome, conteudo in sorted(arquivos.items()):
        meta = interpreta_nome(nome)
        if meta is None:
            problemas.append(f'{nome}: nome fora do padrão <ID>_<CANAL><C|I><n>.txt — ignorado.')
            continue
        try:
            arr = le_canal(conteudo)
        except Exception as e:  # noqa: BLE001
            problemas.append(f'{nome}: não foi possível ler ({e}).')
            continue
        chave = f"{meta['id']}_{meta['cond']}{meta['coleta']}"
        t = testes.setdefault(chave, Teste(meta['id'], meta['cond'], meta['coleta'], {}))
        if meta['codigo'] in t.canais:
            problemas.append(f'{nome}: canal {meta["codigo"]} repetido no teste {chave} — mantido o primeiro.')
            continue
        t.canais[meta['codigo']] = (nome, hashlib.sha256(conteudo).hexdigest(), arr)
    return testes, problemas


def resolucao(x: np.ndarray) -> float:
    """Menor degrau entre valores distintos (resolução de quantização do canal) — 'impressão digital' do sensor."""
    u = np.unique(np.round(np.asarray(x, float), 6))
    d = np.diff(u)
    d = d[d > 1e-9]
    return float(d.min()) if d.size else np.nan


def verifica_integridade(testes: dict, mapa: dict) -> tuple[dict, list[str]]:
    """Procura arquivos idênticos (mesmo SHA-256) entre canais de um mesmo teste — sinal de que um canal foi
    salvo duas vezes no lugar de outro. Para canais de FORÇA duplicados, tenta descobrir a qual célula o
    arquivo pertence comparando a resolução de quantização com a de cada célula em outras coletas.
    Retorna ({chave_teste: {código: motivo da exclusão}}, mensagens)."""
    exclusoes, msgs = {}, []
    # resolução de referência de cada código, a partir de testes sem duplicata nesse código
    # resolução de cada canal em cada teste (só canais sem duplicata no próprio teste)
    res_teste = {}
    for chave, t in testes.items():
        hashes = [v[1] for v in t.canais.values()]
        for c, (nome, h, arr) in t.canais.items():
            if hashes.count(h) == 1 and mapa.get(c, ('', '', ''))[2] == 'Força':
                res_teste[(chave, c)] = resolucao(arr[:, 1])

    def referencia(t, chave_atual):
        """Resolução de cada código de força no MESMO dia (mesmo voluntário e condição); senão, do voluntário."""
        for filtro in (lambda o: o.id == t.id and o.cond == t.cond, lambda o: o.id == t.id):
            ref = {}
            for (ch, c), v in res_teste.items():
                o = testes[ch]
                if ch != chave_atual and filtro(o):
                    ref.setdefault(c, []).append(v)
            if ref:
                return {c: float(np.median(v)) for c, v in ref.items()}
        return {}
    for chave, t in testes.items():
        por_hash = {}
        for c, (nome, h, arr) in t.canais.items():
            por_hash.setdefault(h, []).append(c)
        for h, cods in por_hash.items():
            if len(cods) < 2:
                continue
            arr = t.canais[cods[0]][2]
            res_arq = resolucao(arr[:, 1])
            ref = referencia(t, chave)
            compat = [c for c in cods if c in ref and np.isfinite(ref[c]) and abs(ref[c] - res_arq) <= 1e-4 * max(1, ref[c])]
            todos_forca = all(mapa.get(c, ('', '', ''))[2] == 'Força' for c in cods)
            if len(compat) == 1:
                dono = compat[0]
                for c in cods:
                    if c != dono:
                        exclusoes.setdefault(chave, {})[c] = (
                            f'arquivo idêntico ao de {dono}; pela resolução ({res_arq:.5f}) ele pertence a {dono} — '
                            f'{c} não foi gravado nesta coleta')
                msgs.append(f'{chave}: {" e ".join(cods)} são o MESMO arquivo. Pela resolução do sinal ({res_arq:.5f}), '
                            f'o arquivo é de {dono}; {", ".join(c for c in cods if c != dono)} foi excluído.')
            elif todos_forca:
                dono = cods[0]
                for c in cods[1:]:
                    exclusoes.setdefault(chave, {})[c] = f'arquivo idêntico ao de {dono} (lado não identificável)'
                msgs.append(f'{chave}: {" e ".join(cods)} são o MESMO arquivo e não foi possível identificar a '
                            f'qual célula pertence; mantido {dono} por convenção — CONFIRA.')
            else:
                for c in cods:
                    exclusoes.setdefault(chave, {})[c] = 'arquivo idêntico ao de outro canal'
                msgs.append(f'{chave}: {" e ".join(cods)} são o MESMO arquivo; todos foram excluídos.')
    # células de carga trocadas de lado entre dias (mesma resolução aparece no outro código)
    por_vol = {}
    for (ch, c), v in res_teste.items():
        t = testes[ch]
        por_vol.setdefault(t.id, {}).setdefault(t.cond, {}).setdefault(c, []).append(v)
    for vid, conds in por_vol.items():
        cs = sorted(conds)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = conds[cs[i]], conds[cs[j]]
                comuns = sorted(set(a) & set(b))
                if len(comuns) == 2:
                    c1, c2 = comuns
                    ra = {c: np.median(a[c]) for c in comuns}
                    rb = {c: np.median(b[c]) for c in comuns}
                    if (abs(ra[c1] - rb[c2]) < 1e-4 and abs(ra[c2] - rb[c1]) < 1e-4 and abs(ra[c1] - ra[c2]) > 1e-4):
                        nome_c = {k: v for k, v in CONDICOES.items()}
                        msgs.append(f'Voluntário {vid}: as células {c1} e {c2} aparecem TROCADAS entre o dia '
                                    f'{nome_c.get(cs[i], cs[i])} e o dia {nome_c.get(cs[j], cs[j])} (resolução de {c1}: '
                                    f'{ra[c1]:.5f} no {nome_c.get(cs[i], cs[i])} e {rb[c1]:.5f} na {nome_c.get(cs[j], cs[j])}; '
                                    f'de {c2}: {ra[c2]:.5f} e {rb[c2]:.5f}). Confira se os cabos/'
                                    f'células foram ligados em lados opostos ou se a calibração foi trocada — as '
                                    f'comparações de força por lado entre os dias dependem disso.')
    # o mesmo arquivo em testes diferentes (ex.: Pós salvo por cima do Pré)
    vistos = {}
    for chave, t in testes.items():
        for c, (nome, h, arr) in t.canais.items():
            if h in vistos and vistos[h][0] != chave:
                msgs.append(f'{nome} é idêntico a {vistos[h][1]} (outro teste): confira se a coleta foi salva duas vezes.')
            vistos.setdefault(h, (chave, nome))
    return exclusoes, msgs


def taras_de_referencia(resultados: list) -> dict:
    """{(id, condição, código): (tara, sessão)} a partir de registros com repouso válido.
    Usado para emprestar a tara a outra coleta do mesmo dia (mesma condição) e do mesmo canal."""
    ref = {}
    for r in resultados:
        cond = r.chave.split('_', 1)[1][0]
        for lado, fr in r.forca.items():
            if fr['repouso_ok']:
                ref.setdefault((r.id, cond, fr['codigo']), (fr['tara'], f'{r.condicao} {r.momento} ({r.chave})'))
    return ref


# ==============================================================================
# FILTROS E UTILITÁRIOS
# ==============================================================================
def filtra_emg(x: np.ndarray, p: Parametros) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    secoes = []
    if p.usar_notch:
        secoes.append(tf2sos(*iirnotch(p.f_notch, p.q_notch, p.fs)))
    secoes.append(butter(p.ordem_emg, list(p.banda_emg), btype='bandpass', fs=p.fs, output='sos'))
    return sosfiltfilt(np.vstack(secoes), x)


def filtra_forca(x: np.ndarray, p: Parametros) -> np.ndarray:
    return sosfiltfilt(butter(4, p.fc_forca, btype='low', fs=p.fs, output='sos'), np.asarray(x, float))


def media_movel(x: np.ndarray, n: int) -> np.ndarray:
    """Média móvel centrada sem efeito de borda (nas bordas usa só as amostras disponíveis)."""
    n = max(1, int(n))
    s = pd.Series(x).rolling(n, center=True, min_periods=1).mean()
    return s.to_numpy()


def regressao(t: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 3 or np.ptp(t) == 0:
        return {'slope': np.nan, 'intercepto': np.nan, 'r2': np.nan, 'p': np.nan, 'n': int(len(t))}
    r = stats.linregress(t, y)
    return {'slope': float(r.slope), 'intercepto': float(r.intercept), 'r2': float(r.rvalue ** 2),
            'p': float(r.pvalue), 'n': int(len(t))}


def p_corrigido(t: np.ndarray, y: np.ndarray, k: float) -> float:
    """p bicaudal da inclinação com erro-padrão × sqrt(k) e gl = n/k − 2 (correção para janelas sobrepostas)."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    n_ef = len(t) / k
    if n_ef < 3 or np.ptp(t) == 0:
        return np.nan
    r = stats.linregress(t, y)
    if not np.isfinite(r.stderr) or r.stderr == 0:
        return np.nan
    tstat = r.slope / (r.stderr * np.sqrt(k))
    return float(2 * stats.t.sf(abs(tstat), n_ef - 2))


def frequencias(w: np.ndarray, fs: float, banda: tuple, seg_s: float = 0.25) -> tuple[float, float]:
    """MDF e MNF de um trecho (Welch, Hann, segmentos de seg_s com 50% de sobreposição), na banda do filtro."""
    nps = int(min(len(w), max(64, round(seg_s * fs))))
    f, pxx = welch(w, fs, nperseg=nps, noverlap=nps // 2)
    sel = (f >= banda[0]) & (f <= banda[1])
    f, pxx = f[sel], pxx[sel]
    tot = pxx.sum()
    if tot <= 0:
        return np.nan, np.nan
    c = np.cumsum(pxx)
    mdf = float(np.interp(tot / 2, c, f))
    mnf = float((f * pxx).sum() / tot)
    return mdf, mnf


# ==============================================================================
# FORÇA: TARA, ONSET, ESTABILIZAÇÃO
# ==============================================================================
def verifica_repouso(x: np.ndarray, p: Parametros) -> dict:
    """Mede o repouso antes da contração. Repouso válido: >= repouso_min_s antes do onset e sem tendência
    (a força não pode já estar subindo). Retorna {'ok', 'tara', 'ruido', 'duracao_s', 'motivo'}."""
    fs = p.fs
    n = len(x)
    f = filtra_forca(x, p)
    nivel_bruto = np.median(f[n // 3: 2 * n // 3])
    n_rep = int(p.janela_repouso_s * fs)
    base_ini = np.min(media_movel(f[: max(2, n_rep)], int(0.05 * fs)))
    limiar0 = base_ini + p.onset_pct_patamar * (nivel_bruto - base_ini)
    acima = np.flatnonzero(f > limiar0)
    i_on0 = int(acima[0]) if acima.size else 0
    fim_rep = max(0, i_on0 - int(0.05 * fs))
    ini_rep = max(0, fim_rep - n_rep)
    trecho = f[ini_rep:fim_rep]
    dur = len(trecho) / fs
    if dur < p.repouso_min_s:
        return {'ok': False, 'tara': float(np.min(f[: max(2, i_on0 + 1)])), 'ruido': np.nan, 'duracao_s': dur,
                'motivo': f'só {dur:.2f} s de repouso antes do onset (mínimo {p.repouso_min_s:g} s)'}
    q = max(1, len(trecho) // 4)
    subida = float(np.mean(trecho[-q:]) - np.mean(trecho[:q]))
    if subida > 0.05 * (nivel_bruto - base_ini):
        return {'ok': False, 'tara': float(np.mean(trecho[:q])), 'ruido': float(np.std(trecho)), 'duracao_s': dur,
                'motivo': f'a força já está subindo antes do onset (+{subida:.2f})'}
    return {'ok': True, 'tara': float(np.mean(trecho)), 'ruido': float(np.std(trecho)), 'duracao_s': dur, 'motivo': ''}


def analisa_forca(x: np.ndarray, p: Parametros, nome: str, avisos: list[str],
                  inicio_manual_s: float | None = None, tara_externa: tuple | None = None) -> dict:
    """tara_externa: (valor, origem) — usada quando este registro não tem repouso válido antes da contração."""
    fs = p.fs
    n = len(x)
    f = filtra_forca(x, p)
    rep = verifica_repouso(x, p)
    if rep['ok']:
        tara, ruido, origem_tara = rep['tara'], rep['ruido'], 'repouso deste registro'
    elif tara_externa is not None:
        tara, ruido, origem_tara = float(tara_externa[0]), np.nan, f'repouso de {tara_externa[1]}'
        avisos.append(f'[{nome}] sem repouso válido antes da contração ({rep["motivo"]}): usada a tara do mesmo '
                      f'canal em {tara_externa[1]} ({tara:.2f}). A tara que seria estimada aqui era {rep["tara"]:.2f}.')
    else:
        tara, ruido, origem_tara = rep['tara'], np.nan, 'menor valor antes do onset (estimativa)'
        avisos.append(f'[{nome}] sem repouso válido antes da contração ({rep["motivo"]}) e sem outra coleta do '
                      f'mesmo canal para emprestar a tara: usado o menor valor antes do onset ({tara:.2f}). '
                      f'A força absoluta pode estar subestimada.')
    f = f - tara
    nivel = np.median(f[n // 3: 2 * n // 3])
    acima = np.flatnonzero(f > p.onset_pct_patamar * nivel)
    i_on = int(acima[0]) if acima.size else 0
    fim = n - int(p.fim_ignorado_s * fs)
    # pico inicial (overshoot): máximo nos primeiros segundos após o onset
    lim_busca = min(fim, i_on + int(p.janela_pico_inicial_s * fs))
    i_pico = i_on + int(np.argmax(f[i_on:lim_busca])) if lim_busca > i_on else i_on

    fm = media_movel(f, int(p.media_movel_s * fs))
    n_min = int(p.dur_min_estavel_s * fs)
    i_est, estabilizou = None, False
    ref_ini = i_on + (fim - i_on) // 2              # 1ª estimativa: 2ª metade do esforço
    for _ in range(4):
        mu, sd = float(f[ref_ini:fim].mean()), float(f[ref_ini:fim].std())
        dentro = (np.abs(fm[:fim] - mu) <= p.k_dp * sd).astype(int)
        # nº de amostras 'dentro' nas próximas n_min amostras
        soma = np.convolve(dentro, np.ones(n_min, int), mode='full')[n_min - 1:]
        validos = np.arange(fim) <= fim - n_min
        cand = np.flatnonzero((soma >= n_min) & validos & (np.arange(fim) >= i_on))
        if not cand.size:
            break
        novo = int(cand[0])
        estabilizou = True
        if novo == i_est:
            break
        i_est, ref_ini = novo, novo
    motivo_nao = f'média ± {p.k_dp:g} DP por {p.dur_min_estavel_s:g} s não atingida'
    if estabilizou:
        trecho = f[i_est:fim]
        cv_est = 100 * trecho.std() / trecho.mean() if trecho.mean() > 0 else np.inf
        incl = regressao(np.arange(len(trecho)) / fs, trecho)['slope']
        deriva = 100 * abs(incl) * (len(trecho) / fs) / trecho.mean() if trecho.mean() > 0 else np.inf
        if cv_est > p.cv_max_pct or deriva > p.deriva_max_pct:
            estabilizou = False
            motivo_nao = (f'faixa atingida em {i_est / fs:.2f} s, mas a força não fica estável depois disso '
                          f'(CV {cv_est:.1f}% / limite {p.cv_max_pct:g}%; deriva {deriva:.1f}% / limite '
                          f'{p.deriva_max_pct:g}%)')
    if inicio_manual_s is not None:
        i_ini, modo = int(round(inicio_manual_s * fs)), 'manual'
    elif estabilizou:
        i_ini, modo = i_est, 'automático (estabilizou)'
    else:
        i_ini, modo = min(fim - 1, i_on + int(p.inicio_reserva_s * fs)), \
            f'reserva (onset + {p.inicio_reserva_s:g} s) — NÃO estabilizou'
        avisos.append(f'[{nome}] a força NÃO estabilizou ({motivo_nao}). '
                      f'Usado início de reserva: onset + {p.inicio_reserva_s:g} s. Confira a figura ou defina à mão.')
    i_ini = int(np.clip(i_ini, 0, fim - 1))
    pat = f[i_ini:fim]
    mu, sd = float(pat.mean()), float(pat.std())
    dur_pat = (fim - i_ini) / fs
    if dur_pat < p.patamar_min_s:
        avisos.append(f'[{nome}] patamar curto: {dur_pat:.1f} s (< {p.patamar_min_s:g} s).')
    t_pat = np.arange(len(pat)) / fs
    reg = regressao(t_pat, pat)
    dentro_pct = 100 * float(np.mean(np.abs(fm[i_ini:fim] - mu) <= p.k_dp * sd)) if len(pat) else np.nan
    n_ext = max(1, int(p.extremos_s * fs))
    f_ini, f_fim = float(pat[:n_ext].mean()), float(pat[-n_ext:].mean())
    residuo = pat - (reg['intercepto'] + reg['slope'] * t_pat) if np.isfinite(reg['slope']) else pat - mu
    return {
        'f': f, 'fm': fm, 'tara': tara, 'origem_tara': origem_tara, 'repouso_ok': rep['ok'],
        'tara_propria': rep['tara'], 'ruido_repouso': ruido, 'i_onset': i_on, 'i_pico': i_pico,
        'i_ini': i_ini, 'i_fim': fim, 'estabilizou': estabilizou, 'modo': modo,
        'i_estab_auto': i_est, 'media': mu, 'dp': sd, 'cv': 100 * sd / mu if mu else np.nan,
        'cv_sem_tendencia': 100 * float(np.std(residuo)) / mu if mu else np.nan,
        'slope': reg['slope'], 'slope_norm': 100 * reg['slope'] / mu if mu else np.nan,
        'pct_na_faixa': dentro_pct, 'duracao_patamar_s': dur_pat,
        'pico_inicial': float(f[i_pico]), 'overshoot_pct': 100 * (float(f[i_pico]) - mu) / mu if mu else np.nan,
        'tempo_estabilizacao_s': (i_ini - i_on) / fs,
        'forca_ini': f_ini, 'forca_fim': f_fim, 'var_forca_pct': 100 * (f_fim - f_ini) / f_ini if f_ini else np.nan,
    }


# ==============================================================================
# EMG: JANELAS, ARTEFATOS, MDF/MNF/RMS
# ==============================================================================
def analisa_emg(x: np.ndarray, i_ini: int, i_fim: int, p: Parametros, forca: np.ndarray | None = None) -> dict:
    """forca: força tarada de referência (mesmo lado) para a ENM por janela = força média ÷ RMS."""
    fs = p.fs
    bruto = np.asarray(x, float)
    xf = filtra_emg(bruto, p)
    n_w = int(round(p.janela_espectral_s * fs))
    passo = max(1, int(round(n_w * (1 - p.sobreposicao))))
    pat = xf[i_ini:i_fim]
    sig = 1.4826 * np.median(np.abs(pat - np.median(pat))) if len(pat) else np.nan
    linhas = []
    for a in range(i_ini, i_fim - n_w + 1, passo):
        w = xf[a:a + n_w]
        wb = bruto[a:a + n_w]
        mdf, mnf = frequencias(w, fs, p.banda_emg, p.welch_segmento_s)
        rms = float(np.sqrt(np.mean(w ** 2)))
        trilho = int(np.sum(np.abs(wb) >= 0.98 * p.fundo_escala)) if p.fundo_escala > 0 else 0
        pico = float(np.abs(w).max() / sig) if sig > 0 else np.nan
        f_w = float(np.mean(forca[a:a + n_w])) if forca is not None else np.nan
        linhas.append({'t_ini_s': a / fs, 't_centro_s': (a + n_w / 2) / fs, 'MDF_Hz': mdf, 'MNF_Hz': mnf,
                       'RMS': rms, 'Forca': f_w, 'ENM': f_w / rms if rms > 0 else np.nan,
                       'amostras_no_limite': trilho, 'pico_em_DP': pico})
    jan = pd.DataFrame(linhas)
    if jan.empty:
        return {'xf': xf, 'janelas': jan, 'sig': sig}
    med_rms = float(np.median(jan['RMS']))
    jan['RMS_razao_mediana'] = jan['RMS'] / med_rms if med_rms > 0 else np.nan
    motivos = []
    for _, r in jan.iterrows():
        m = []
        if r['amostras_no_limite'] > 0:
            m.append('atinge o limite do conversor')
        if r['pico_em_DP'] > p.pico_max_dp:
            m.append(f'pico {r["pico_em_DP"]:.0f} DP')
        if r['RMS_razao_mediana'] > p.rms_max_razao:
            m.append(f'RMS {r["RMS_razao_mediana"]:.1f}× a mediana')
        motivos.append('; '.join(m))
    jan['motivo_artefato'] = motivos
    jan['valida'] = (jan['motivo_artefato'] == '') | (not p.excluir_artefatos)
    return {'xf': xf, 'janelas': jan, 'sig': sig}


def resume_emg(jan: pd.DataFrame, t0_s: float, p: Parametros) -> dict:
    """Métricas de fadiga a partir das janelas válidas. Tempo da regressão relativo ao início do patamar."""
    v = jan[jan['valida']] if len(jan) else jan
    out = {'n_janelas': int(len(jan)), 'n_validas': int(len(v)),
           'pct_excluidas': 100 * (1 - len(v) / len(jan)) if len(jan) else np.nan}
    if len(v) < 3:
        for k in ['MDF_ini', 'MDF_fim', 'MDF_slope', 'MDF_slope_norm', 'MDF_r2', 'MDF_p', 'Queda_MDF',
                  'MNF_ini', 'MNF_fim', 'MNF_slope', 'MNF_slope_norm', 'MNF_r2', 'MNF_p',
                  'RMS_ini', 'RMS_fim', 'RMS_slope', 'RMS_slope_norm', 'RMS_r2', 'RMS_p', 'Var_RMS',
                  'ENM_ini', 'ENM_fim', 'ENM_slope', 'ENM_slope_norm', 'ENM_r2', 'ENM_p', 'Var_ENM', 'Queda_MDF_reg']:
            out[k] = np.nan
        return out
    t = v['t_centro_s'].to_numpy() - t0_s
    t_fim = float(t.max())
    sel_ini = t <= t.min() + p.extremos_s
    sel_fim = t >= t_fim - p.extremos_s
    # p-valor corrigido pela sobreposição: janelas sobrepostas não são independentes; o erro-padrão da
    # inclinação é inflado por sqrt(k) e os graus de liberdade usam n/k janelas (k = 1/(1 − sobreposição)).
    k_ind = 1 / (1 - p.sobreposicao) if p.sobreposicao > 0 else 1.0
    for var, col in [('MDF', 'MDF_Hz'), ('MNF', 'MNF_Hz'), ('RMS', 'RMS'), ('ENM', 'ENM')]:
        y = v[col].to_numpy(dtype=float)
        reg = regressao(t, y)
        p_ind = p_corrigido(t, y, k_ind)
        ini, fim = float(np.nanmean(y[sel_ini])), float(np.nanmean(y[sel_fim]))
        out[f'{var}_ini'], out[f'{var}_fim'] = ini, fim
        out[f'{var}_slope'], out[f'{var}_r2'], out[f'{var}_p'] = reg['slope'], reg['r2'], p_ind
        out[f'{var}_slope_norm'] = 100 * reg['slope'] / reg['intercepto'] if reg['intercepto'] else np.nan
    out['Queda_MDF'] = 100 * (out['MDF_fim'] - out['MDF_ini']) / out['MDF_ini'] if out['MDF_ini'] else np.nan
    # queda pela reta de regressão (usa o patamar inteiro; menos sensível a poucas janelas nos extremos)
    dur = float(t.max() - t.min())
    out['Queda_MDF_reg'] = out['MDF_slope_norm'] * dur if np.isfinite(out['MDF_slope_norm']) else np.nan
    out['Var_RMS'] = 100 * (out['RMS_fim'] - out['RMS_ini']) / out['RMS_ini'] if out['RMS_ini'] else np.nan
    out['Var_ENM'] = 100 * (out['ENM_fim'] - out['ENM_ini']) / out['ENM_ini'] if out['ENM_ini'] else np.nan
    return out


# ==============================================================================
# PIPELINE DE UM TESTE
# ==============================================================================
@dataclass
class Resultado:
    chave: str
    id: str
    condicao: str
    coleta: str
    momento: str
    parametros: dict
    arquivos: dict            # código -> (nome, sha256)
    n_amostras: int
    duracao_s: float
    t: np.ndarray
    forca: dict               # lado ('Dir'/'Esq'/'') -> saída de analisa_forca (+ 'codigo')
    emg: dict                 # código -> {'xf', 'janelas', 'resumo', 'lado', 'i_ini', 'i_fim'}
    df_forca: pd.DataFrame
    df_emg: pd.DataFrame      # tabela de fadiga (compatível com o portal)
    df_emg_extra: pd.DataFrame
    df_janelas: pd.DataFrame  # formato longo, todas as janelas
    atenuacao_notch: pd.Series
    df_assimetria: pd.DataFrame = None
    mapa: dict = None
    avisos: list = field(default_factory=list)


def processa_teste(teste: Teste, mapa: dict, p: Parametros, coletas: dict | None = None,
                   inicio_manual: dict | None = None, excluir: dict | None = None,
                   taras: dict | None = None) -> Resultado:
    """mapa: código -> (nome, lado, tipo). inicio_manual: lado -> início do patamar (s) definido à mão.
    excluir: {código: motivo} canais a descartar (ex.: duplicados). taras: {código: (tara, origem)} emprestadas
    de outra coleta, usadas só quando este registro não tem repouso válido."""
    valida_parametros(p)
    coletas = coletas or COLETAS_PADRAO
    inicio_manual = inicio_manual or {}
    excluir = excluir or {}
    taras = taras or {}
    avisos: list[str] = []
    fs = p.fs
    for c, motivo in excluir.items():
        if c in teste.canais:
            avisos.append(f'Canal {c} excluído: {motivo}.')

    conhecidos = {c: v for c, v in teste.canais.items() if c in mapa and c not in excluir}
    desconhecidos = sorted(set(teste.canais) - set(conhecidos) - set(excluir))
    if desconhecidos:
        avisos.append(f'Canais sem cadastro ignorados: {", ".join(desconhecidos)}.')
    esperados = sorted(set(mapa) - set(teste.canais) - set(excluir))
    if esperados:
        avisos.append(f'Canais cadastrados ausentes neste teste: {", ".join(esperados)}.')
    forca_cod = [c for c in conhecidos if mapa[c][2] == 'Força']
    emg_cod = [c for c in conhecidos if mapa[c][2] == 'EMG']
    if not forca_cod or not emg_cod:
        raise ValueError(f'Teste {teste.chave}: é preciso ao menos 1 canal de força e 1 de EMG.')
    lados_forca = [mapa[c][1] for c in forca_cod]
    if len(set(lados_forca)) != len(lados_forca):
        raise ValueError(f'Teste {teste.chave}: mais de um canal de força no mesmo lado.')

    # alinhamento temporal
    tempos = {c: v[2][:, 0] for c, v in conhecidos.items()}
    n = min(len(tt) for tt in tempos.values())
    if len({len(tt) for tt in tempos.values()}) > 1:
        avisos.append(f'Os canais têm comprimentos diferentes; todos foram cortados em {n} amostras.')
    t_ref = next(iter(tempos.values()))[:n]
    if not all(np.allclose(tt[:n], t_ref, atol=0.5 / fs) for tt in tempos.values()):
        avisos.append('Os vetores de tempo dos canais não coincidem: confira se são da mesma coleta.')
    if p.verificar_fs_pelo_tempo and n > 2:
        dt = float(np.median(np.diff(t_ref)))
        if dt > 0 and abs(1 / dt - fs) / fs > 0.01:
            avisos.append(f'A coluna de tempo sugere {1 / dt:.1f} Hz, mas fs = {fs:g} Hz.')
    t = np.arange(n) / fs

    # força por lado
    forca = {}
    for c in forca_cod:
        lado = mapa[c][1]
        sinal = conhecidos[c][2][:n, 1]
        res = analisa_forca(sinal, p, nome_exibicao(c, mapa), avisos, inicio_manual.get(lado), taras.get(c))
        res['codigo'] = c
        forca[lado] = res
    lados_sem_forca = sorted({mapa[c][1] for c in emg_cod if mapa[c][1] and mapa[c][1] not in forca})
    for lado in lados_sem_forca:
        ref_l = max(forca.values(), key=lambda r: r['i_ini'])
        avisos.append(f'Sem célula de carga válida do lado {NOME_LADO[lado]}: os músculos desse lado usaram o '
                      f'patamar de {nome_exibicao(ref_l["codigo"], mapa)}, e a ENM deles usa essa força.')
    if p.janela_comum:
        i_com = max(r['i_ini'] for r in forca.values())
        for r in forca.values():
            r['i_ini_emg'] = i_com
    else:
        for r in forca.values():
            r['i_ini_emg'] = r['i_ini']

    def forca_do_lado(lado: str) -> dict:
        if lado in forca:
            return forca[lado]
        # músculo sem lado (ou lado sem célula): usa o início mais tardio entre as células
        return max(forca.values(), key=lambda r: r['i_ini_emg'])

    # EMG
    emg, linhas, extras, janelas_longas = {}, [], [], []
    for c in emg_cod:
        nome, lado, _ = mapa[c]
        rf = forca_do_lado(lado)
        i_ini, i_fim = rf['i_ini_emg'], rf['i_fim']
        sinal = conhecidos[c][2][:n, 1]
        r = analisa_emg(sinal, i_ini, i_fim, p, rf['f'][:n] if (lado in forca or len(forca) == 1)
                        else np.sum([fr['f'][:n] for fr in forca.values()], axis=0))
        jan = r['janelas']
        resumo = resume_emg(jan, i_ini / fs, p)
        r.update(resumo=resumo, lado=lado, i_ini=i_ini, i_fim=i_fim)
        emg[c] = r
        rot = nome_exibicao(c, mapa)
        if resumo['n_validas'] < 3:
            avisos.append(f'[{rot}] menos de 3 janelas válidas: métricas de fadiga não calculadas.')
        elif resumo['pct_excluidas'] > 0:
            avisos.append(f'[{rot}] {resumo["n_janelas"] - resumo["n_validas"]} de {resumo["n_janelas"]} janelas '
                          f'excluídas por artefato ({resumo["pct_excluidas"]:.0f}%).')
        if p.fundo_escala > 0 and np.any(np.abs(sinal) >= 0.98 * p.fundo_escala):
            avisos.append(f'[{rot}] o sinal atinge o limite do conversor (±{p.fundo_escala:g} {p.unidade_emg}).')
        ue = p.unidade_emg
        linhas.append({
            'Músculo': rot,
            'MDF inicial (Hz)': resumo['MDF_ini'], 'MDF final (Hz)': resumo['MDF_fim'],
            'MDF slope (Hz/s)': resumo['MDF_slope'], 'Queda MDF (%)': resumo['Queda_MDF'],
            'RMS slope (' + ue + '/s)': resumo['RMS_slope'], 'R² MDF': resumo['MDF_r2'],
            'p-valor MDF': resumo['MDF_p'],
        })
        extras.append({
            'Canal': rot,
            'Freq. média inicial (Hz)': resumo['MNF_ini'], 'Freq. média final (Hz)': resumo['MNF_fim'],
            'Inclinação freq. média (Hz/s)': resumo['MNF_slope'],
            'Freq. mediana normalizada (%/s)': resumo['MDF_slope_norm'],
            'Queda pela regressão (%)': resumo['Queda_MDF_reg'],
            f'RMS inicial ({ue})': resumo['RMS_ini'], f'RMS final ({ue})': resumo['RMS_fim'],
            'Variação RMS (%)': resumo['Var_RMS'], 'R² RMS': resumo['RMS_r2'],
            'Variação ENM (%)': resumo['Var_ENM'],
            'Janelas válidas': f'{resumo["n_validas"]}/{resumo["n_janelas"]}',
            'Patamar usado (s)': f'{fmt(i_ini / fs)}–{fmt(i_fim / fs)}',
        })
        if len(jan):
            jl = jan.copy()
            jl.insert(0, 'Musculo', rot)
            jl.insert(1, 't_rel_patamar_s', jl['t_centro_s'] - i_ini / fs)
            janelas_longas.append(jl)

    uf = p.unidade_forca
    df_forca = pd.DataFrame({
        nome_exibicao(r['codigo'], mapa): {
            f'Tara ({uf})': r['tara'], 'Instante do onset (s)': r['i_onset'] / fs,
            f'Pico inicial ({uf})': r['pico_inicial'], 'Início do patamar (s)': r['i_ini'] / fs,
            'Origem da tara': r['origem_tara'],
            'Critério do início': r['modo'], 'Tempo até estabilizar (s)': r['tempo_estabilizacao_s'],
            'Duração do patamar (s)': r['duracao_patamar_s'],
            f'Força média no patamar ({uf})': r['media'], f'Força inicial ({uf})': r['forca_ini'],
            f'Força final ({uf})': r['forca_fim'], 'Variação da força (%)': r['var_forca_pct'],
            f'DP no patamar ({uf})': r['dp'], 'CV no patamar (%)': r['cv'],
            'CV sem tendência (%)': r['cv_sem_tendencia'], f'Inclinação da força ({uf}/s)': r['slope'],
            'Overshoot inicial (%)': r['overshoot_pct'],
            f'% do tempo em média ± {p.k_dp:g} DP': r['pct_na_faixa'],
        } for r in forca.values()})

    # assimetria bilateral por músculo (pares Dir/Esq com o mesmo nome)
    linhas_as = {}
    por_nome = {}
    for c in emg_cod:
        por_nome.setdefault(mapa[c][0], {})[mapa[c][1]] = emg[c]['resumo']
    for nome, lados in por_nome.items():
        if 'Dir' in lados and 'Esq' in lados:
            d, e = lados['Dir'], lados['Esq']
            linhas_as[nome] = {
                'Queda MDF D (%)': d['Queda_MDF'], 'Queda MDF E (%)': e['Queda_MDF'],
                'Diferença D−E da queda (p.p.)': d['Queda_MDF'] - e['Queda_MDF'],
                'Inclinação D (Hz/s)': d['MDF_slope'], 'Inclinação E (Hz/s)': e['MDF_slope'],
                'Diferença D−E da inclinação (Hz/s)': d['MDF_slope'] - e['MDF_slope'],
                f'RMS D ({p.unidade_emg}/s)': d['RMS_slope'], f'RMS E ({p.unidade_emg}/s)': e['RMS_slope'],
            }
    df_assim = pd.DataFrame(linhas_as).T

    # auditoria espectral do notch
    atenu = {}
    for c in emg_cod:
        bruto = conhecidos[c][2][:n, 1]
        f_, pb = welch(bruto - bruto.mean(), fs, nperseg=min(2048, n))
        _, pf = welch(emg[c]['xf'], fs, nperseg=min(2048, n))
        b = np.abs(f_ - p.f_notch) <= 2
        if b.any() and pb[b].sum() > 0 and pf[b].sum() > 0:
            atenu[nome_exibicao(c, mapa)] = 10 * np.log10(pb[b].sum() / pf[b].sum())

    return Resultado(
        chave=teste.chave, id=teste.id, condicao=CONDICOES.get(teste.cond, teste.cond), coleta=teste.coleta,
        momento=coletas.get(teste.coleta, f'Coleta {teste.coleta}'), parametros=asdict(p),
        arquivos={c: (v[0], v[1]) for c, v in conhecidos.items()}, n_amostras=n, duracao_s=n / fs, t=t,
        forca=forca, emg=emg, df_forca=df_forca, df_emg=pd.DataFrame(linhas).set_index('Músculo'),
        df_emg_extra=pd.DataFrame(extras).set_index('Canal'),
        df_janelas=pd.concat(janelas_longas, ignore_index=True) if janelas_longas else pd.DataFrame(),
        atenuacao_notch=pd.Series(atenu, name=f'Atenuação em {p.f_notch:g} ± 2 Hz (dB)', dtype=float),
        df_assimetria=df_assim, mapa=dict(mapa), avisos=avisos)


# ==============================================================================
# TABELAS CONSOLIDADAS (VÁRIOS TESTES)
# ==============================================================================
def tabela_longa(resultados: list[Resultado], civm: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(EMG: 1 linha por teste × músculo, Força: 1 linha por teste × lado).
    civm: {(id, lado, sessão): CIVM do lado naquela sessão}, sessão = 'C1', 'C2', 'I1', 'I2'... (a CIVM muda a cada dia)."""
    civm = civm or {}
    emg_rows, f_rows = [], []
    for r in resultados:
        base = {'ID_Voluntario': r.id, 'Condicao': r.condicao, 'Momento': r.momento,
                'Sessao': r.chave.split('_', 1)[1]}
        for c, e in r.emg.items():
            s = e['resumo']
            emg_rows.append({**base, 'Musculo': r.df_emg.index[list(r.emg).index(c)], 'Lado': e['lado'],
                             'MDF_Inicial_Hz': s['MDF_ini'], 'MDF_Final_Hz': s['MDF_fim'],
                             'MDF_Slope_Hz_s': s['MDF_slope'], 'MDF_Slope_norm_pct_s': s['MDF_slope_norm'],
                             'Queda_MDF_pct': s['Queda_MDF'], 'Queda_MDF_regressao_pct': s['Queda_MDF_reg'],
                             'MDF_R2': s['MDF_r2'], 'MDF_p_valor': s['MDF_p'],
                             'MNF_Inicial_Hz': s['MNF_ini'], 'MNF_Final_Hz': s['MNF_fim'],
                             'MNF_Slope_Hz_s': s['MNF_slope'],
                             'RMS_Inicial': s['RMS_ini'], 'RMS_Final': s['RMS_fim'],
                             'RMS_Slope_por_s': s['RMS_slope'], 'RMS_Slope_norm_pct_s': s['RMS_slope_norm'],
                             'Var_RMS_pct': s['Var_RMS'], 'RMS_R2': s['RMS_r2'], 'RMS_p_valor': s['RMS_p'],
                             'ENM_Inicial': s['ENM_ini'], 'ENM_Final': s['ENM_fim'], 'ENM_Slope_por_s': s['ENM_slope'],
                             'Var_ENM_pct': s['Var_ENM'], 'Pct_janelas_excluidas': s['pct_excluidas'],
                             'Janelas_validas': s['n_validas'], 'Janelas_total': s['n_janelas'],
                             'Inicio_patamar_s': e['i_ini'] / r.parametros['fs'],
                             'Fim_patamar_s': e['i_fim'] / r.parametros['fs']})
        for lado, fr in r.forca.items():
            alvo = civm.get((r.id, lado, r.chave.split('_', 1)[1]))
            f_rows.append({**base, 'Lado': lado or 'único', 'Canal': fr['codigo'],
                           'Tara': fr['tara'], 'Origem_tara': fr['origem_tara'],
                           'Onset_s': fr['i_onset'] / r.parametros['fs'],
                           'Tempo_ate_estabilizar_s': fr['tempo_estabilizacao_s'],
                           'Overshoot_pct': fr['overshoot_pct'], 'Forca_inicial': fr['forca_ini'],
                           'Forca_final': fr['forca_fim'], 'Var_forca_pct': fr['var_forca_pct'],
                           'CV_sem_tendencia_pct': fr['cv_sem_tendencia'],
                           'Pico_inicial': fr['pico_inicial'], 'Inicio_patamar_s': fr['i_ini'] / r.parametros['fs'],
                           'Estabilizou': fr['estabilizou'], 'Criterio_inicio': fr['modo'],
                           'Duracao_patamar_s': fr['duracao_patamar_s'], 'Forca_media': fr['media'],
                           'Forca_DP': fr['dp'], 'Forca_CV_pct': fr['cv'], 'Forca_slope_por_s': fr['slope'],
                           'Pct_tempo_na_faixa': fr['pct_na_faixa'],
                           'CIVM_informada': alvo,
                           'Pct_CIVM': 100 * fr['media'] / alvo if alvo else np.nan})
    return pd.DataFrame(emg_rows), pd.DataFrame(f_rows)


VARIAVEIS_COMPARACAO = {
    'MDF_Slope_Hz_s': 'Inclinação da MDF (Hz/s)', 'Queda_MDF_pct': 'Queda da MDF (%)',
    'MDF_Slope_norm_pct_s': 'Inclinação normalizada da MDF (%/s)',
    'RMS_Slope_por_s': 'Inclinação do RMS (/s)', 'RMS_Slope_norm_pct_s': 'Inclinação normalizada do RMS (%/s)',
    'Var_RMS_pct': 'Variação do RMS (%)', 'MDF_Inicial_Hz': 'MDF inicial (Hz)',
}
ORDEM_COLETAS = [('Controle', 'Pré'), ('Controle', 'Pós'), ('Intervenção', 'Pré'), ('Intervenção', 'Pós')]


def compara_coletas(df_emg: pd.DataFrame, variavel: str) -> pd.DataFrame:
    """Média ± DP por músculo nas 4 coletas e testes pareados (por voluntário):
    (a) Intervenção-Pós × Controle-Pós; (b) Δ(Pós−Pré) Intervenção × Δ(Pós−Pré) Controle."""
    if df_emg.empty or variavel not in df_emg:
        return pd.DataFrame()
    linhas = []
    for musc, g in df_emg.groupby('Musculo', sort=False):
        if g.duplicated(['ID_Voluntario', 'Condicao', 'Momento']).any():
            linhas.append({'Músculo': musc, 'Aviso': 'duplicatas voluntário×condição×momento — revise'})
            continue
        w = g.pivot_table(index='ID_Voluntario', columns=['Condicao', 'Momento'], values=variavel)
        lin = {'Músculo': musc}
        for cond, mom in ORDEM_COLETAS:
            col = w[(cond, mom)] if (cond, mom) in w else pd.Series(dtype=float)
            lin[f'{cond} {mom}'] = f"{fmt(col.mean())} ± {fmt(col.std())} (n={col.notna().sum()})" \
                if col.notna().any() else '—'

        def pareado(a, b):
            par = pd.concat([a, b], axis=1).dropna()
            if len(par) < 3:
                return len(par), np.nan, np.nan
            pt = stats.ttest_rel(par.iloc[:, 0], par.iloc[:, 1]).pvalue
            try:
                pw = stats.wilcoxon(par.iloc[:, 0], par.iloc[:, 1]).pvalue
            except ValueError:
                pw = np.nan
            return len(par), pt, pw
        tem = lambda k: k in w
        if tem(('Intervenção', 'Pós')) and tem(('Controle', 'Pós')):
            n, pt, pw = pareado(w[('Intervenção', 'Pós')], w[('Controle', 'Pós')])
            lin['I-Pós × C-Pós: n'] = n
            lin['I-Pós × C-Pós: p (t)'] = pt
            lin['I-Pós × C-Pós: p (Wilcoxon)'] = pw
        if all(tem(k) for k in ORDEM_COLETAS):
            d_i = w[('Intervenção', 'Pós')] - w[('Intervenção', 'Pré')]
            d_c = w[('Controle', 'Pós')] - w[('Controle', 'Pré')]
            n, pt, pw = pareado(d_i, d_c)
            lin['ΔI × ΔC: média ΔI−ΔC'] = (d_i - d_c).mean()
            lin['ΔI × ΔC: n'] = n
            lin['ΔI × ΔC: p (t)'] = pt
            lin['ΔI × ΔC: p (Wilcoxon)'] = pw
        linhas.append(lin)
    return pd.DataFrame(linhas)


# ==============================================================================
# FORMATAÇÃO E METADADOS
# ==============================================================================
def fmt(x, casas: int = 2) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return '—'
    if isinstance(x, (bool, np.bool_)):
        return 'sim' if x else 'não'
    if isinstance(x, str):
        return x
    s = f'{float(x):,.{casas}f}'
    return s.replace(',', '§').replace('.', ',').replace('§', '.')


def casas_padrao(linha, coluna) -> int:
    l, c = str(linha), str(coluna)
    if 'p-valor' in c or 'p-valor' in l or c.endswith(': p') or c.endswith('p Holm') or c.startswith('p '):
        return 3
    if 'R²' in c or 'R²' in l or '/s' in c or '/s' in l:
        return 3
    if '(%)' in c or '(%)' in l or '%' in l:
        return 1
    return 2


def formata_tabela(df: pd.DataFrame, casas: int | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for i in df.index:
        for c in df.columns:
            out.loc[i, c] = fmt(df.loc[i, c], casas if casas is not None else casas_padrao(i, c))
    return out


def mapa_para_json(mapa: dict, coletas: dict) -> bytes:
    return json.dumps({'canais': [{'codigo': c, 'nome': n, 'lado': l, 'tipo': t} for c, (n, l, t) in mapa.items()],
                       'coletas': coletas}, ensure_ascii=False, indent=2).encode('utf-8')


def mapa_de_json(conteudo: bytes) -> tuple[dict, dict]:
    try:
        cfg = json.loads(conteudo.decode('utf-8-sig'))
        mapa = {str(c['codigo']).upper(): (str(c['nome']).strip(), str(c.get('lado') or ''), str(c['tipo']))
                for c in cfg['canais']}
        coletas = {str(k): str(v) for k, v in cfg.get('coletas', COLETAS_PADRAO).items()}
        return mapa, coletas
    except Exception as e:  # noqa: BLE001
        raise ValueError(f'Configuração inválida: {e}') from e


def valida_mapa(mapa: dict) -> None:
    for c, (n, l, t) in mapa.items():
        if not re.fullmatch(r'[A-Z]+', c):
            raise ValueError(f'Código "{c}" inválido: use só letras (ex.: VLD).')
        if not n:
            raise ValueError(f'O canal {c} precisa de nome.')
        if l not in LADOS_VALIDOS or t not in TIPOS_VALIDOS:
            raise ValueError(f'Lado/tipo inválido no canal {c}.')
    forcas = [l for (_, l, t) in mapa.values() if t == 'Força']
    if not forcas:
        raise ValueError('Cadastre ao menos um canal de força.')
    if len(forcas) > 2 or len(set(forcas)) != len(forcas):
        raise ValueError('Use 1 canal de força, ou 2 (um Dir e um Esq).')


def descricao_parametros(p: Parametros) -> list[tuple[str, str, str]]:
    notch = f'{p.f_notch:g} Hz, Q = {p.q_notch:g}' if p.usar_notch else 'desativado'
    return [
        ('Aquisição', 'Frequência de amostragem', f'{p.fs:g} Hz'),
        ('EMG', 'Passa-banda Butterworth (fase zero)', f'{p.banda_emg[0]:g}–{p.banda_emg[1]:g} Hz, ordem {p.ordem_emg}'),
        ('EMG', 'Notch (fase zero)', notch),
        ('Força', 'Passa-baixa Butterworth (fase zero)', f'{p.fc_forca:g} Hz, ordem 4'),
        ('Força', 'Tara', f'média de {p.janela_repouso_s:g} s de repouso antes do onset'),
        ('Força', 'Onset', f'força > {p.onset_pct_patamar:.0%} do nível do patamar'),
        ('Patamar', 'Critério de estabilização (por lado)',
         f'média móvel de {p.media_movel_s:g} s dentro de média ± {p.k_dp:g} DP por ≥ {p.dur_min_estavel_s:g} s '
         f'(média/DP reestimados a partir do início encontrado)'),
        ('Patamar', 'Verificação da estabilidade', f'CV ≤ {p.cv_max_pct:g}% e deriva ≤ {p.deriva_max_pct:g}% '
         f'da média ao longo do patamar'),
        ('Patamar', 'Se não estabilizar', f'início = onset + {p.inicio_reserva_s:g} s (sinalizado)'),
        ('Patamar', 'Janela dos músculos', 'comum (início mais tardio)' if p.janela_comum else 'por lado'),
        ('Patamar', 'Fim do registro descartado', f'{p.fim_ignorado_s:g} s'),
        ('Espectro', 'Janelas', f'{p.janela_espectral_s:g} s, sobreposição {p.sobreposicao:.0%} '
         f'(passo de {p.janela_espectral_s * (1 - p.sobreposicao):g} s)'),
        ('Espectro', 'Estimativa espectral', f'Welch (Hann), segmentos de {p.welch_segmento_s:g} s com 50% de '
         f'sobreposição; MDF = 50% da potência na banda do filtro; MNF = média ponderada'),
        ('Espectro', 'Inicial / final', f'média das janelas válidas nos primeiros / últimos {p.extremos_s:g} s do patamar'),
        ('Espectro', 'Inclinação', 'regressão linear por mínimos quadrados vs. tempo no patamar; p-valor com erro-padrão '
         'corrigido pela sobreposição das janelas (n efetivo = n × (1 − sobreposição))'),
        ('Força', 'Tara sem repouso', 'se o registro não tem ≥ ' + f'{p.repouso_min_s:g}' + ' s de repouso estável antes do '
         'onset, usa-se a tara do mesmo canal em outra coleta da mesma condição (mesmo dia)'),
        ('Integridade', 'Arquivos duplicados', 'canais com conteúdo idêntico são detectados; para células de carga, '
         'o dono do arquivo é identificado pela resolução de quantização'),
        ('Artefatos', 'Exclusão de janelas',
         (f'ativa: limite ±{p.fundo_escala:g} {p.unidade_emg}, pico > {p.pico_max_dp:g} DP (MAD), '
          f'RMS > {p.rms_max_razao:g}× a mediana') if p.excluir_artefatos else 'desativada (só sinalizadas)'),
    ]


def versoes() -> dict:
    import matplotlib
    import scipy
    v = {'Python': platform.python_version(), 'NumPy': np.__version__, 'SciPy': scipy.__version__,
         'pandas': pd.__version__, 'Matplotlib': matplotlib.__version__, 'Aplicação': APP_VERSION}
    try:
        import weasyprint
        v['WeasyPrint'] = weasyprint.__version__
    except Exception:
        pass
    return v

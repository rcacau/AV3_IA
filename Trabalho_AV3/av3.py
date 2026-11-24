"""Experimentos AV3. Execute com python av3.py."""
import csv
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median, multimode
from time import perf_counter

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "resultados"


def escrever(nome, campos, linhas):
    with (OUT / nome).open("w", encoding="utf-8", newline="") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(campos.split())
        writer.writerows(linhas)


def esfera(x, y):
    return x*x + y*y


def gaussianas(x, y):
    return np.exp(-x*x-y*y) + 2*np.exp(-((x-1.7)**2+(y-1.7)**2))


def ackley(x, y):
    return -20*np.exp(-0.2*np.sqrt((x*x+y*y)/2)) - np.exp(
        (np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2
    ) + 20 + np.e


def rastrigin(x, y):
    return x*x+y*y-10*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))+20


def mista(x, y):
    return x*np.cos(x)/20 + 2*np.exp(-x*x-(y-1)**2) + 0.01*x*y


PROBLEMAS = [
    ("Esfera", esfera, (-100, -100), (100, 100), 1, (0, 0)),
    ("Gaussianas", gaussianas, (-2, -2), (4, 5), -1, (1.697348, 1.697348)),
    ("Ackley", ackley, (-8, -8), (8, 8), 1, (0, 0)),
    ("Rastrigin", rastrigin, (-5.12, -5.12), (5.12, 5.12), 1, (0, 0)),
    ("Mista", mista, (-10, -10), (10, 10), -1, (0.015, 1.00004)),
]


def busca_2d(problema, metodo, passo, rodadas=100):
    _, funcao, baixo, alto, sinal, _ = problema
    rng = np.random.default_rng(2026)
    atual = (np.tile(baixo, (rodadas, 1)).astype(float) if metodo == "HC"
             else rng.uniform(baixo, alto, (rodadas, 2)))
    custos = sinal*funcao(*atual.T)
    estagnacao = np.zeros(rodadas, dtype=int)
    iteracoes = np.zeros(rodadas, dtype=int)
    for _ in range(1000):
        ativos = np.flatnonzero(estagnacao < 250)
        if ativos.size == 0:
            break
        forma = (ativos.size, 2)
        if metodo == "HC":
            candidatos = atual[ativos] + rng.uniform(-passo, passo, forma)
        elif metodo == "LRS":
            candidatos = atual[ativos] + rng.normal(0, passo, forma)
        else:
            candidatos = rng.uniform(baixo, alto, forma)
        validos = np.all((candidatos >= baixo) & (candidatos <= alto), axis=1)
        valores = sinal*funcao(*candidatos.T)
        melhoria = validos & (valores < custos[ativos])
        indices = ativos[melhoria]
        atual[indices] = candidatos[melhoria]
        custos[indices] = valores[melhoria]
        estagnacao[ativos] += 1
        estagnacao[indices] = 0
        iteracoes[ativos] += 1
    return atual, sinal*custos, iteracoes


def continuo():
    rodadas, grade, resumo = [], [], []
    for problema in PROBLEMAS:
        nome, funcao, _, _, sinal, ponto = problema
        referencia = float(funcao(*ponto))
        for metodo, passos in (
            ("HC", (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)),
            ("LRS", (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 0.99)),
            ("GRS", (None,)),
        ):
            testes = []
            for passo in passos:
                x, f, it = busca_2d(problema, metodo, passo)
                erro = np.maximum(0, sinal*(f-referencia))
                acertos = int(np.sum(erro <= 0.01))
                testes.append((passo, x, f, it, erro, acertos))
                grade.append((nome, metodo, passo, acertos, median(erro)))
            escolhido = next((t for t in testes if t[-1] > 0), None)
            passo, x, f, it, erro, acertos = escolhido or min(
                testes, key=lambda t: median(t[-2])
            )
            freq = Counter(np.round(f, 2))
            frequencia = max(freq.values())
            modas = sorted(float(v) for v, n in freq.items() if n == frequencia)
            melhor = np.argmin(erro)
            resumo.append((nome, metodo, passo, escolhido is not None, referencia,
                           acertos, frequencia, modas, f[melhor], median(erro)))
            for r, (par, valor, i, e) in enumerate(zip(x, f, it, erro), 1):
                rodadas.append((nome, metodo, passo, r, *par, valor, i, e))
    escrever("continuo_rodadas.csv", "funcao metodo passo rodada x1 x2 f iteracoes erro", rodadas)
    escrever("continuo_grade.csv", "funcao metodo passo acertos erro_mediano", grade)
    escrever("continuo_resumo.csv",
             "funcao metodo passo atingiu_tolerancia referencia acertos frequencia modas melhor erro_mediano",
             resumo)
    return resumo


PARES = [(i, j) for i in range(8) for j in range(i+1, 8)]


def aptidao(tabuleiro):
    conflitos = sum(tabuleiro[i] == tabuleiro[j] or
                    abs(tabuleiro[i]-tabuleiro[j]) == j-i for i, j in PARES)
    return 28-conflitos


def tempera(rng):
    atual = [rng.randint(1, 8) for _ in range(8)]
    valor = aptidao(atual)
    melhor, melhor_valor = atual, valor
    temperatura = 2.0
    iteracoes = 0
    while iteracoes < 2500 and melhor_valor < 28:
        vizinho = atual.copy()
        coluna = rng.randrange(8)
        vizinho[coluna] = (atual[coluna]+rng.randrange(7)) % 8 + 1
        novo = aptidao(vizinho)
        delta = novo-valor
        if delta >= 0 or rng.random() < math.exp(delta/temperatura):
            atual, valor = vizinho, novo
        if valor > melhor_valor:
            melhor, melhor_valor = atual, valor
        temperatura = max(0.01, temperatura*0.995)
        iteracoes += 1
    return tuple(melhor), melhor_valor, iteracoes


def rainhas():
    inicio = perf_counter()
    rng = random.Random(2026)
    solucoes = {}
    rodadas = sucessos = iteracoes = 0
    while len(solucoes) < 92:
        tabuleiro, nota, total = tempera(rng)
        rodadas += 1
        iteracoes += total
        if nota == 28:
            sucessos += 1
            solucoes.setdefault(tabuleiro, rodadas)
    escrever("rainhas.csv", "col1 col2 col3 col4 col5 col6 col7 col8 primeira_rodada",
             ((*tabuleiro, rodada) for tabuleiro, rodada in solucoes.items()))
    return rodadas, sucessos, iteracoes, perf_counter()-inicio, next(iter(solucoes))


def torneio(custos, rng, quantidade):
    candidatos = rng.integers(len(custos), size=(quantidade, 3))
    return candidatos[np.arange(quantidade), custos[candidatos].argmin(axis=1)]


def ordem(pai, mae, inicio, fim):
    """Preserva o segmento e preenche circularmente na ordem do segundo pai."""
    tamanho = len(pai)
    filho = np.empty_like(pai)
    filho[inicio:fim] = pai[inicio:fim]
    usados = np.zeros(tamanho, dtype=bool)
    usados[pai[inicio:fim]] = True
    leitura = np.concatenate((mae[fim:], mae[:fim]))
    livre = leitura[~usados[leitura]]
    filho[fim:] = livre[:tamanho-fim]
    filho[:inicio] = livre[tamanho-fim:]
    return filho


def custo_rota(rotas, dist):
    origem = len(dist)-1
    return (dist[origem, rotas[:, 0]] + dist[rotas[:, -1], origem]
            + dist[rotas[:, :-1], rotas[:, 1:]].sum(axis=1))


def ga_rota(dist, alvo, elite, semente, tamanho=60, limite=250):
    rng = np.random.default_rng(semente)
    n = len(dist)-1
    populacao = np.array([rng.permutation(n) for _ in range(tamanho)])
    melhor_custo = float("inf")
    historico = []
    for geracao in range(limite+1):
        custos = custo_rota(populacao, dist)
        indice = custos.argmin()
        if custos[indice] < melhor_custo:
            melhor_custo = custos[indice]
            melhor_rota = populacao[indice].copy()
        historico.append(melhor_custo)
        if melhor_custo <= alvo or geracao == limite:
            return melhor_rota, melhor_custo, geracao, historico
        nova = np.empty_like(populacao)
        nova[:elite] = populacao[np.argsort(custos)[:elite]]
        pais = torneio(custos, rng, tamanho-elite)
        for posicao in range(elite, tamanho, 2):
            a, b = populacao[pais[posicao-elite:posicao-elite+2]]
            inicio, fim = sorted(rng.choice(n+1, 2, replace=False))
            for linha, (pai, mae) in enumerate(((a, b), (b, a)), posicao):
                filho = ordem(pai, mae, inicio, fim)
                if rng.random() < 0.01:
                    i, j = rng.choice(n, 2, replace=False)
                    filho[i], filho[j] = filho[j], filho[i]
                nova[linha] = filho
        populacao = nova


def caixeiro():
    dados = np.loadtxt(BASE / "CaixeiroGruposGA.csv", delimiter=",")
    ids = np.flatnonzero(dados[:, 3] != 0)
    origem = np.flatnonzero(dados[:, 3] == 0)
    assert len(origem) == 1
    grupos = Counter(dados[ids, 3].astype(int))
    assert all(30 < numero < 60 for numero in grupos.values())
    indices = np.r_[ids, origem]
    pontos = dados[indices, :3]
    dist = np.linalg.norm(pontos[:, None]-pontos, axis=2)
    # Calibração independente com sementes diferentes das rodadas reportadas.
    pilotos = [ga_rota(dist, -1, elite, s)[1]
               for s in range(90, 95) for elite in (0, 2)]
    alvo = 1.05*median(pilotos)
    resultados, curvas, resumo = [], [], []
    for elite in (0, 2):
        tempos, custos, acertos = [], [], []
        menor = float("inf")
        for rodada in range(20):
            inicio = perf_counter()
            rota, custo, geracao, historico = ga_rota(dist, alvo, elite, 400+rodada)
            segundos = perf_counter()-inicio
            assert np.array_equal(np.sort(rota), np.arange(len(ids)))
            custos.append(custo)
            tempos.append(segundos)
            if custo <= alvo:
                acertos.append(geracao)
            if custo < menor:
                menor, melhor = custo, rota
            resultados.append((elite, rodada+1, custo, geracao, custo <= alvo, segundos))
            curvas.extend((elite, rodada+1, g, f) for g, f in enumerate(historico))
        circuito = np.r_[len(ids), melhor, len(ids)]
        escrever(f"rota_elite_{elite}.csv", "ordem linha_csv grupo x y z",
                 ((ordem_rota, indices[i]+1, int(dados[indices[i], 3]), *pontos[i])
                  for ordem_rota, i in enumerate(circuito)))
        resumo.append((elite, len(acertos), multimode(acertos), menor,
                       median(custos), mean(tempos), max(Counter(acertos).values(), default=0)))
    escrever("caixeiro_rodadas.csv", "elite rodada custo geracao aceito segundos", resultados)
    escrever("caixeiro_convergencia.csv", "elite rodada geracao custo", curvas)
    escrever("caixeiro_resumo.csv",
             "elite acertos modas melhor mediana tempo_medio frequencia_modal",
             resumo)
    return alvo, resumo


def objetivo_50(x):
    return 500 + np.sum(x*x-10*np.cos(np.pi*x), axis=-1)


def ga_real(rng, tamanho, orcamento):
    populacao = rng.uniform(-5.12, 5.12, (tamanho, 50))
    custos = objetivo_50(populacao)
    trilha = [(tamanho, custos.min())]
    avaliadas = tamanho
    while avaliadas < orcamento:
        quantidade = min(tamanho-1, orcamento-avaliadas)
        pares = (quantidade+1)//2
        a, b = populacao[torneio(custos, rng, 2*pares)].reshape(2, pares, 50)
        u = rng.random(a.shape)
        beta = np.where(u <= 0.5, np.sqrt(2*u), 1/np.sqrt(2*(1-u)))
        centro, desvio = (a+b)/2, beta*(a-b)/2
        filhos = np.stack((centro+desvio, centro-desvio), axis=1).reshape(-1, 50)[:quantidade]
        filhos += rng.normal(0, 0.35, filhos.shape)*(rng.random(filhos.shape) < 1/50)
        np.clip(filhos, -5.12, 5.12, out=filhos)
        indice = custos.argmin()
        populacao = np.vstack((populacao[[indice]], filhos))
        custos = np.r_[custos[indice], objetivo_50(filhos)]
        avaliadas += quantidade
        trilha.append((avaliadas, custos.min()))
    return populacao[custos.argmin()], trilha


def lrs_real(rng, orcamento):
    atual = rng.uniform(-5.12, 5.12, 50)
    valor = objetivo_50(atual)
    trilha = [(1, valor)]
    for i in range(2, orcamento+1):
        candidato = atual.copy()
        indice = rng.integers(50)
        candidato[indice] = min(5.12, max(-5.12, candidato[indice] + rng.normal(0, 0.8)))
        custo = objetivo_50(candidato)
        if custo < valor:
            atual, valor = candidato, custo
        if i % 200 == 0 or i == orcamento:
            trilha.append((i, valor))
    return atual, trilha


def comparacao():
    resumo, rodadas, curvas = [], [], []
    for tamanho in (20, 50, 100, 200, 0):
        custos, tempos = [], []
        nome = "GA-SBX" if tamanho else "LRS"
        orcamento = 20000
        for rodada in range(20):
            rng = np.random.default_rng(800+rodada)
            inicio = perf_counter()
            x, trilha = ga_real(rng, tamanho, orcamento) if tamanho else lrs_real(rng, orcamento)
            tempo = perf_counter()-inicio
            custo = objetivo_50(x)
            custos.append(custo)
            tempos.append(tempo)
            rodadas.append((nome, tamanho, rodada+1, custo, tempo, *x))
            eixo, valores = np.array(trilha).T
            for marco in range(200, orcamento+1, 200):
                indice = np.searchsorted(eixo, marco, side="right")-1
                curvas.append((nome, tamanho, rodada+1, marco, valores[indice]))
        resumo.append((nome, tamanho, min(custos), median(custos), mean(tempos)))
    escrever("comparacao_rodadas.csv", "metodo populacao rodada custo segundos " +
             " ".join(f"x{i}" for i in range(1, 51)), rodadas)
    escrever("comparacao_convergencia.csv", "metodo populacao rodada candidatos custo", curvas)
    escrever("comparacao_resumo.csv", "metodo populacao melhor mediana tempo_medio", resumo)
    return resumo


def main():
    OUT.mkdir(exist_ok=True)
    print("Buscas contínuas", flush=True)
    continuo()
    print("8 rainhas", flush=True)
    r, s, i, t, exemplo = rainhas()
    escrever("rainhas_resumo.csv", "rodadas sucessos iteracoes segundos exemplo",
             [(r, s, i, t, exemplo)])
    print("Caixeiro-viajante", flush=True)
    alvo, _ = caixeiro()
    escrever("caixeiro_alvo.csv", "alvo", [(alvo,)])
    print("Comparação em 50 dimensões", flush=True)
    comparacao()
    print(f"Resultados em {OUT}")


if __name__ == "__main__":
    main()

import os

def obter_dados_das_fontes():
    diretorio_base = os.path.dirname(os.path.abspath(__file__))

    with open(diretorio_base + "\\sentiment labelled sentences\\" + "imdb_labelled.txt", "r") as arquivo_texto:
        dados = arquivo_texto.read().split('\n')

    with open(diretorio_base + "\\sentiment labelled sentences\\" + "amazon_cells_labelled.txt", "r") as arquivo_texto:
        dados += arquivo_texto.read().split('\n')

    with open(diretorio_base + "\\sentiment labelled sentences\\" + "yelp_labelled.txt", "r") as arquivo_texto:
        dados += arquivo_texto.read().split('\n')

    return dados

def tratamento_dos_dados(dados):
    dados_tratados = []
    for dado in dados:
        if len(dado.split("\t")) == 2 and dado.split("\t")[1] != "":
            dados_tratados.append(dado.split("\t"))

    return dados_tratados


def dividir_dados_para_treino_e_validacao(dados):
    quantidade_total = len(dados)
    percentual_para_treino = 0.75
    treino = []
    validacao = []

    for indice in range(0, quantidade_total):
        if indice < quantidade_total * percentual_para_treino:
            treino.append(dados[indice])
        else:
            validacao.append(dados[indice])

    return treino, validacao


def pre_processamento():
    dados = obter_dados_das_fontes()
    dados_tratados = tratamento_dos_dados(dados)

    return dividir_dados_para_treino_e_validacao(dados_tratados)

if __name__ == "__main__":
    pre_processamento()
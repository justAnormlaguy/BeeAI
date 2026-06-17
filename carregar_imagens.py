import os
import cv2
import numpy as np

# ==========================================
# 2. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ==========================================
def carregar_e_processar_imagens(dir_dataset, tamanho=(128, 128)):
    X = []
    y = []
    classes = {'inadequadas': 0, 'adequadas': 1}

    for classe_nome, classe_valor in classes.items():
        caminho_pasta = os.path.join(dir_dataset, classe_nome)
        if not os.path.exists(caminho_pasta):
            continue

        for ficheiro in os.listdir(caminho_pasta):
            caminho_imagem = os.path.join(caminho_pasta, ficheiro)
            imagem = cv2.imread(caminho_imagem)

            if imagem is not None:
                # Redimensionamento
                imagem_redimensionada = cv2.resize(imagem, tamanho)
                # Normalização (0 a 1)
                imagem_normalizada = imagem_redimensionada / 255.0

                X.append(imagem_normalizada)
                y.append(classe_valor)

    return np.array(X), np.array(y)
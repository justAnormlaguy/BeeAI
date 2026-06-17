import os
from pathlib import Path

# PREPARAÇÃO DO DATASET (SYMLINKS)
def organizar_dataset_simbolico(dir_origem, dir_destino):
    """
    Organiza o dataset em duas classes binárias focadas na produção de mel.
    Realiza exclusão ao nível do ficheiro (apenas imagens válidas) e cria symlinks
    em vez de copiar/mover os diretórios inteiros.
    """
    classes_adequadas = ['sunflower', 'dandelion']
    classes_inadequadas = ['rose', 'tulip', 'daisy']

    os.makedirs(os.path.join(dir_destino, 'adequadas'), exist_ok=True)
    os.makedirs(os.path.join(dir_destino, 'inadequadas'), exist_ok=True)

    extensoes_validas = {'.jpg', '.jpeg', '.png'}

    for root, _, files in os.walk(dir_origem):
        for file in files:
            # Exclusão ao nível do ficheiro para ignorar formatos indesejados
            if Path(file).suffix.lower() not in extensoes_validas:
                continue

            nome_pasta = os.path.basename(root)
            src_path = os.path.abspath(os.path.join(root, file))

            if nome_pasta in classes_adequadas:
                dest_path = os.path.join(dir_destino, 'adequadas', file)
            elif nome_pasta in classes_inadequadas:
                dest_path = os.path.join(dir_destino, 'inadequadas', file)
            else:
                continue

            # Criação do symlink se não existir
            if not os.path.exists(dest_path):
                try:
                    os.symlink(src_path, dest_path)
                except OSError as e:
                    print(f"Erro ao criar symlink para {file}: {e}")
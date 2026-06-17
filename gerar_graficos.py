import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# GERAÇÃO DE GRÁFICOS
def gerar_todos_os_graficos(historico, resultados_acuracia, y_teste, y_pred_rf):
    print("\nA gerar os gráficos de avaliação...")

    # Gráfico 1: Curva de Loss da CNN
    plt.figure(figsize=(10, 4))
    plt.plot(historico.history['loss'], label='Loss de Treino')
    plt.plot(historico.history['val_loss'], label='Loss de Validação')
    plt.title('Curva de Perda (Loss) - Rede Neural')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('cnn_loss_curve.png')

    # Gráfico 2: Comparação de Acurácia de todos os modelos
    plt.figure(figsize=(10, 6))
    nomes_modelos = list(resultados_acuracia.keys())
    valores_acc = list(resultados_acuracia.values())
    plt.bar(nomes_modelos, valores_acc, color=['blue', 'cyan', 'green', 'orange', 'purple'])
    plt.title('Comparação de Acurácia dos Modelos de IA')
    plt.ylabel('Acurácia')
    plt.ylim(0, 1)
    for i, v in enumerate(valores_acc):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
    plt.savefig('comparacao_modelos.png')

    # Gráfico 3: Matriz de Confusão (Exemplo com Random Forest)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_teste, y_pred_rf), display_labels=['Inadequadas', 'Adequadas'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão - Random Forest")
    plt.savefig('matriz_confusao_rf.png')

    print("Os gráficos foram guardados como ficheiros PNG no diretório atual.")
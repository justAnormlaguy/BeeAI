from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras import layers, models
from tratar_imagens import organizar_dataset_simbolico
from venv.carregar_Imagens import carregar_e_processar_imagens
from venv.gerar_graficos import gerar_todos_os_graficos


# TREINAMENTO E AVALIAÇÃO
def main():
    dir_origem_kaggle = '../flowers'
    dir_dataset_simbolico = '../dataset_meliferas'

    print("A estruturar diretórios e ficheiros...")
    organizar_dataset_simbolico(dir_origem_kaggle, dir_dataset_simbolico)

    print("A carregar e processar as imagens (isto pode demorar um pouco)...")
    X, y = carregar_e_processar_imagens(dir_dataset_simbolico)

    # Divisão de Treino e Teste (80% treino, 20% teste)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Achatamento (Flatten) para os algoritmos tradicionais
    X_treino_plano = X_treino.reshape(X_treino.shape[0], -1)
    X_teste_plano = X_teste.reshape(X_teste.shape[0], -1)

    resultados_acuracia = {}

    # --- MODELOS DA PARTE 1 ---
    print("\n--- A treinar Modelos da Parte 1 ---")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_treino_plano, y_treino)
    y_pred_knn = knn.predict(X_teste_plano)
    resultados_acuracia['KNN'] = accuracy_score(y_teste, y_pred_knn)
    print(f"Acurácia KNN: {resultados_acuracia['KNN']:.4f}")

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_treino_plano, y_treino)
    y_pred_nb = nb.predict(X_teste_plano)
    resultados_acuracia['Naive Bayes'] = accuracy_score(y_teste, y_pred_nb)
    print(f"Acurácia Naive Bayes: {resultados_acuracia['Naive Bayes']:.4f}")

    # --- MODELOS DA PARTE 2 ---
    print("\n--- A treinar Modelos da Parte 2 ---")

    # SVM
    svm = SVC(kernel='linear')
    svm.fit(X_treino_plano, y_treino)
    y_pred_svm = svm.predict(X_teste_plano)
    resultados_acuracia['SVM'] = accuracy_score(y_teste, y_pred_svm)
    print(f"Acurácia SVM: {resultados_acuracia['SVM']:.4f}")

    # Random Forest (Ensemble)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_treino_plano, y_treino)
    y_pred_rf = rf.predict(X_teste_plano)
    resultados_acuracia['Random Forest'] = accuracy_score(y_teste, y_pred_rf)
    print(f"Acurácia Random Forest: {resultados_acuracia['Random Forest']:.4f}")

    # Rede Neural Convolucional (CNN)
    print("A treinar a Rede Neural (CNN)...")
    modelo_cnn = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Saída binária
    ])

    modelo_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    historico = modelo_cnn.fit(X_treino, y_treino, epochs=10, validation_data=(X_teste, y_teste), verbose=1)

    loss_cnn, acc_cnn = modelo_cnn.evaluate(X_teste, y_teste, verbose=0)
    resultados_acuracia['Rede Neural (CNN)'] = acc_cnn
    print(f"Acurácia CNN: {acc_cnn:.4f}")

    gerar_todos_os_graficos(historico, resultados_acuracia, y_teste, y_pred_rf)

    print("\nPipeline concluído.")

if __name__ == "__main__":
    main()
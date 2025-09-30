

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

PASTA_VALIDACAO = os.path.join(project_root, 'dataset/neurotypical')
CAMINHO_MODELO = os.path.join(project_root, 'lib/models/modelo_tea_classifier.h5')
CLASSE_VERDADEIRA = 'neurotypical' 

NUM_IMAGENS_TESTE = 50

IMG_WIDTH, IMG_HEIGHT = 224, 224

if not os.path.exists(CAMINHO_MODELO):
    print(f"Erro: Modelo '{CAMINHO_MODELO}' não encontrado.")
    exit()
if not os.path.exists(PASTA_VALIDACAO):
    print(f"Erro: Pasta de validação '{PASTA_VALIDACAO}' não encontrada.")
    exit()

print("Carregando modelo...")
model = tf.keras.models.load_model(CAMINHO_MODELO)
print("Modelo carregado!")

arquivos_imagem = [f for f in os.listdir(PASTA_VALIDACAO) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

random.shuffle(arquivos_imagem)
imagens_para_testar = arquivos_imagem[:NUM_IMAGENS_TESTE]

if len(imagens_para_testar) == 0:
    print(f"Nenhuma imagem encontrada em '{PASTA_VALIDACAO}'. Verifique o caminho e os arquivos.")
    exit()

print(f"\nIniciando validação em {len(imagens_para_testar)} imagens da classe '{CLASSE_VERDADEIRA}'...\n")

acertos = 0
total_testado = 0

for nome_imagem in imagens_para_testar:
    total_testado += 1
    caminho_completo = os.path.join(PASTA_VALIDACAO, nome_imagem)

    img = image.load_img(caminho_completo, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch, verbose=0)
    score = prediction[0][0]

    if score < 0.5:
        classe_prevista = 'autism'
    else:
        classe_prevista = 'neurotypical'

    resultado = "INCORRETO"
    if classe_prevista == CLASSE_VERDADEIRA:
        resultado = "CORRETO"
        acertos += 1
    
    print(f"Imagem [{total_testado}/{len(imagens_para_testar)}]: {nome_imagem} -> Previsto: {classe_prevista} (Resultado: {resultado})")

acuracia = (acertos / total_testado) * 100

print("\n--- Relatório Final da Validação ---")
print(f"Pasta Testada: '{PASTA_VALIDACAO}'")
print(f"Classe Verdadeira: '{CLASSE_VERDADEIRA}'")
print(f"Total de Imagens Testadas: {total_testado}")
print(f"Previsões Corretas: {acertos}")
print(f"Previsões Incorretas: {total_testado - acertos}")
print(f"Acurácia nesta classe: {acuracia:.2f}%")
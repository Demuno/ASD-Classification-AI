import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

DIRETORIO_DATASET = os.path.join(project_root, 'dataset')
CAMINHO_SALVAR_MODELO = os.path.join(project_root, 'lib', 'models', 'modelo_tea_classifier.h5')


IMG_WIDTH, IMG_HEIGHT = 224, 224
TAMANHO_IMAGEM = (IMG_WIDTH, IMG_HEIGHT)

TAMANHO_LOTE = 20
NUM_EPOCAS = 45

if not os.path.exists(DIRETORIO_DATASET):
    print(f"Erro: O diretório '{DIRETORIO_DATASET}' não foi encontrado.")
    print("Verifique se o script está na pasta 'src' e a pasta 'dataset' existe no nível superior.")
    exit()

diretorio_modelos = os.path.dirname(CAMINHO_SALVAR_MODELO)
if not os.path.exists(diretorio_modelos):
    os.makedirs(diretorio_modelos)
    print(f"Diretório '{diretorio_modelos}' criado para salvar o modelo.")



datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 
)

train_generator = datagen.flow_from_directory(
    DIRETORIO_DATASET,
    target_size=TAMANHO_IMAGEM,
    batch_size=TAMANHO_LOTE,
    class_mode='binary', 
    subset='training'    
)

validation_generator = datagen.flow_from_directory(
    DIRETORIO_DATASET,
    target_size=TAMANHO_IMAGEM,
    batch_size=TAMANHO_LOTE,
    class_mode='binary',
    subset='validation'  
)


base_model = MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

print("Resumo do Modelo:")
model.summary()

print("\nIniciando o treinamento...")
history = model.fit(
    train_generator,
    epochs=NUM_EPOCAS,
    validation_data=validation_generator
)

model.save(CAMINHO_SALVAR_MODELO)

print(f"\nTreinamento concluído e modelo salvo em '{CAMINHO_SALVAR_MODELO}'")




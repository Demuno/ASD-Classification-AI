import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

path_script = os.path.dirname(os.path.abspath(__file__))
path_model = os.path.join(path_script, '..', 'lib', 'models', 'modelo_tea_classifier.h5')

try:
    print("Carregando o modelo Keras...")
    model = tf.keras.models.load_model(path_model)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None

IMG_WIDTH, IMG_HEIGHT = 224, 224

def prepare_image(image):

    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo não foi carregado. Verifique os logs do servidor.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo está vazio'}), 400

    try:
        image = Image.open(file.stream)

        processed_image = prepare_image(image)
        prediction = model.predict(processed_image)
        probability = float(prediction[0][0])
        prob_sem_tea = probability
        prob_com_tea = 1.0 - prob_sem_tea
    
        response = {
            'probabilidade_com_tea': round(prob_com_tea * 100, 2),
            'probabilidade_sem_tea': round(prob_sem_tea * 100, 2),
            'observacao': 'Este é um resultado de um modelo de IA e não substitui um diagnóstico profissional.'
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro ao processar a imagem: {str(e)}'}), 500

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
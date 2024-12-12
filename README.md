# Projeto de Reconhecimento Facial com TensorFlow

Este projeto demonstra como utilizar o **TensorFlow** e a biblioteca **face_recognition** para detectar e classificar rostos em uma imagem. Especificamente, o código lê uma imagem do elenco de "O Senhor dos Anéis" na pasta -input- e salva uma imagem com as caixas delimitadoras e a classificação com o nome de cada personagem na pasta -output-.

## Estrutura do Projeto

```Sh
face_recognition_project/
├── input/
│   └── elenco.jpg  # Imagem do elenco de "O Senhor dos Anéis"
├── output/
├── images/
│   ├── frodo.jpg
│   ├── sam.jpg
│   ├── merry.jpg
│   ├── gandalf.jpg
│   ├── aragorn.jpg
│   └── legolas.jpg
└── recognize_faces.py
```

## Pré-requisitos

Certifique-se de ter o Python e as seguintes bibliotecas instaladas:

- TensorFlow

- face_recognition

- OpenCV

- NumPy

Você pode instalar essas bibliotecas utilizando o pip:

```Sh
pip install tensorflow face_recognition opencv-python numpy
```

## Descrição do Código

O código principal está no arquivo **recognize_faces.py**:

```python
import face_recognition
import cv2
import numpy as np
from pathlib import Path

# Diretórios de entrada e saída
input_dir = Path('input')
output_dir = Path('output')
images_dir = Path('images')
output_dir.mkdir(parents=True, exist_ok=True)

# Carregar imagens e aprender a reconhecer os rostos
known_face_encodings = []
known_face_names = []

# Adicione as imagens dos personagens na pasta 'images'
character_images = {
    "Frodo": "frodo.jpg",
    "Sam": "sam.jpg",
    "Aragorn": "aragorn.jpg",
    "Legolas": "legolas.jpg",
    "Gandalf": "gandalf.jpg",
    "Merry": "merry.jpg"
}

for name, filename in character_images.items():
    image_path = images_dir / filename
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    else:
        print(f"Não foi possível detectar um rosto em {filename}")

# Ler a imagem do elenco de 'O Senhor dos Anéis'
input_image_path = input_dir / 'elenco.jpg'
image = cv2.imread(str(input_image_path))
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Encontrar todos os rostos e codificações de rostos na imagem
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# Inicializar uma lista de nomes de personagens detectados
face_names = []

# Reconhecer cada rosto na imagem de entrada
for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Desconhecido"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)

# Desenhar caixas delimitadoras e etiquetas com os nomes dos personagens na imagem
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# Salvar a imagem anotada na pasta 'output'
output_image_path = output_dir / 'elenco_annotated.jpg'
cv2.imwrite(str(output_image_path), image)

print(f'Imagem anotada salva em: {output_image_path}')
```

## Explicação do Código

### 1. Importar Bibliotecas:

- **face_recognition**: Biblioteca para reconhecimento facial.

- **cv2**: Biblioteca OpenCV para manipulação de imagens.

- **numpy**: Biblioteca para manipulação de arrays.

- **Path**: Classe da biblioteca pathlib para manipulação de caminhos de arquivos.

### 2. Configurar Diretórios:

- **input_dir**: Diretório onde a imagem do elenco está armazenada.

- **output_dir**: Diretório onde a imagem anotada será salva.

- **images_dir**: Diretório onde as imagens individuais dos personagens estão armazenadas.

### 3. Treinamento com Imagens dos Personagens:

- Carregar as imagens dos personagens e aprender a reconhecer os rostos usando **face_recognition.face_encodings**.

- Adicionar as codificações dos rostos conhecidos (**known_face_encodings**) e os nomes dos personagens (**known_face_names**).

### 4. Detecção de Rostos na Imagem do Elenco:

- Carregar a imagem do elenco e convertê-la para o formato RGB.

- Encontrar todos os rostos na imagem e obter suas codificações.

- Inicializar uma lista para armazenar os nomes dos personagens detectados.

### 5. Reconhecimento Facial:

- Para cada rosto detectado, comparar com os rostos conhecidos e encontrar o melhor resultado de correspondência.

- Adicionar o nome do personagem detectado à lista face_names.

### 6. Desenhar Caixas Delimitadoras:

- Desenhar caixas delimitadoras ao redor de cada rosto detectado e adicionar etiquetas com os nomes dos personagens.

- Usar a função **cv2.rectangle** para desenhar as caixas e **cv2.putText** para adicionar os nomes.

### 7. Salvar a Imagem Anotada:

- Salvar a imagem anotada na pasta **output** com o nome **elenco_annotated.jpg**.

## Como Executar

1. Coloque a imagem do elenco de "**O Senhor dos Anéis**" na pasta **input** com o nome **elenco.jpg**.

![elenco]()

2. Coloque as imagens dos personagens na pasta **images** com os respectivos nomes de arquivo (**frodo.jpg**, **sam.jpg**, etc.).

![images]()

3. Execute o script:

```sh
python recognize_faces.py
```

![retorno]()

4. A imagem com as caixas delimitadoras e classificações será salva na pasta **output** com o nome **elenco_annotated.jpg**.

![elenco_annotated]()

## Contribuição <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="25" height="25" />

Sinta-se à vontade para contribuir com este projeto. Você pode abrir issues para relatar problemas ou fazer pull requests para melhorias.

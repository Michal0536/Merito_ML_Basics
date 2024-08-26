# Importowanie biblioteki TensorFlow i modułów z Keras, które są używane do budowy i trenowania modeli sieci neuronowych
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Wczytywanie danych MNIST z TensorFlow Keras; zestaw danych jest automatycznie podzielony na obrazy treningowe i testowe oraz ich etykiety
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalizacja wartości pikseli obrazów do zakresu 0-1, co poprawia konwergencję podczas trenowania sieci neuronowej
train_images = train_images / 255.0
test_images = test_images / 255.0

# Dodanie wymiaru kanału do obrazów (28x28) -> (28x28x1); potrzebne, bo warstwa Conv2D oczekuje trójwymiarowych danych na wejściu
train_images = train_images[..., None]
test_images = test_images[..., None]

# Budowanie modelu sieci konwolucyjnej za pomocą Keras; używamy modelu sekwencyjnego, co oznacza, że warstwy dodawane są jedna po drugiej
model = models.Sequential([
    # Dodanie pierwszej warstwy konwolucyjnej, która ma 32 filtry, każdy o rozmiarze 3x3, i funkcję aktywacji ReLU
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Dodanie warstwy Max Pooling, która redukuje wymiary obrazu przez wybieranie maksymalnych wartości z okien 2x2
    layers.MaxPooling2D((2, 2)),
    # Kolejna warstwa konwolucyjna z 64 filtrami, każdy również 3x3, z funkcją aktywacji ReLU
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Kolejna warstwa Max Pooling
    layers.MaxPooling2D((2, 2)),
    # Ostatnia warstwa konwolucyjna, również z filtrami 3x3 i funkcją aktywacji ReLU
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Warstwa spłaszczająca (Flatten), która przekształca trójwymiarowe cechy na jednowymiarowy wektor
    layers.Flatten(),
    # Warstwa gęsta (Dense) z 64 jednostkami i funkcją aktywacji ReLU
    layers.Dense(64, activation='relu'),
    # Warstwa wyjściowa z 10 jednostkami (dla 10 cyfr) i funkcją aktywacji softmax, która służy do klasyfikacji wieloklasowej
    layers.Dense(10, activation='softmax')
])

# Kompilacja modelu, definiowanie optymalizatora (Adam), funkcji strat (sparse_categorical_crossentropy) i metryki (accuracy)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu na danych treningowych; trenujemy przez 10 epok i wykorzystujemy 10% danych treningowych jako dane walidacyjne
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# Ewaluacja modelu na danych testowych; wyświetlamy stratę i dokładność
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")  # Wydruk dokładności testowej
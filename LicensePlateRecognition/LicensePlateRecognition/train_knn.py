import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def load_data(data_folder):
    """
    Funkcja do ładowania danych treningowych z folderu.

    Args:
        data_folder (str): Ścieżka do folderu z danymi.

    Returns:
        np.array: Tablica z obrazami.
        np.array: Tablica z etykietami.
    """
    # Lista do przechowywania obrazów
    images = []

    # Lista do przechowywania etykiet
    labels = []

    # Iteracja po wszystkich podfolderach w folderze danych
    for char_name in os.listdir(data_folder):
        # Ścieżka do podfolderu znaku
        char_folder = os.path.join(data_folder, char_name)

        # Iteracja po wszystkich obrazach w podfolderze znaku
        for image_name in os.listdir(char_folder):
            # Ścieżka do obrazu
            image_path = os.path.join(char_folder, image_name)

            # Ładowanie obrazu w skali szarości
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Spłaszczanie obrazu do jednowymiarowej tablicy
            image_flat = image.flatten()

            # Dodawanie spłaszczonego obrazu do listy obrazów
            images.append(image_flat)

            # Dodawanie etykiety (nazwa znaku) do listy etykiet
            labels.append(char_name)

    # Konwersja list na tablice numpy
    images_array = np.array(images)
    labels_array = np.array(labels)

    return images_array, labels_array


def main():
    """
    Główna funkcja programu.

    Ładuje data, trenuje model k-NN, ocenia jego dokładność i zapisuje wytrenowany model do pliku.
    """
    # Ścieżka do folderu z danymi
    data_folder = 'data/chars'

    # Ładowanie danych
    images, labels = load_data(data_folder)

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Tworzenie klasyfikatora k-NN z 3 sąsiadami
    knn = KNeighborsClassifier(n_neighbors=3)

    # Trenowanie modelu k-NN na danych treningowych
    knn.fit(X_train, y_train)

    # Przewidywanie etykiet dla danych testowych
    y_pred = knn.predict(X_test)

    # Obliczanie i wyświetlanie dokładności modelu
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Ścieżka do pliku, w którym zostanie zapisany wytrenowany model
    model_path = 'rest_of_code/knn_model.pkl'

    # Zapisanie wytrenowanego modelu do pliku
    joblib.dump(knn, model_path)

    print(f"Model został zapisany do pliku {model_path}")


if __name__ == "__main__":
    # Uruchamianie głównej funkcji programu
    main()

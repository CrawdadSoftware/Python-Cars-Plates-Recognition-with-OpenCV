import cv2
import numpy as np
import joblib
from reszta_kodu.data import load_image, preprocess_image


def load_model(model_path):
    # Ładowanie wytrenowanego modelu k-NN
    model = joblib.load(model_path)
    return model


def find_license_plate_contour(image):
    # Konwersja obrazu do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarizacja obrazu przy użyciu Otsu
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Dodatkowe operacje morfologiczne w celu oczyszczenia obrazu
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Znajdowanie konturów na przetworzonym obrazie
    contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filtracja konturów na podstawie obszaru
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h  # Obliczanie współczynnika proporcji
            if 2 < aspect_ratio < 6:  # Typowy współczynnik proporcji tablic rejestracyjnych
                return contour  # Zwracanie konturu tablicy rejestracyjnej
    return None  # Zwracanie None, jeśli nie znaleziono odpowiedniego konturu


def extract_license_plate(image, contour):
    # Obliczanie prostokąta ograniczającego dla konturu
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Ustalanie szerokości i wysokości prostokąta ograniczającego
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Definiowanie punktów źródłowych dla transformacji perspektywy
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # Obliczanie macierzy transformacji perspektywy
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Zastosowanie transformacji perspektywy
    warped = cv2.warpPerspective(image, M, (width, height))

    # Obracanie obrazu, jeśli jego wysokość jest większa niż szerokość
    if height > width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped  # Zwracanie prostowanego obrazu tablicy rejestracyjnej


def recognize_characters(plate_image, model):
    # Konwersja obrazu do skali szarości, jeśli jest w kolorze
    if len(plate_image.shape) == 3 and plate_image.shape[2] == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image

    # Binarizacja obrazu przy użyciu Otsu
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Dodatkowe operacje morfologiczne w celu oczyszczenia obrazu
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Znajdowanie konturów na przetworzonym obrazie
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filtracja konturów na podstawie wielkości i proporcji
        if h > 20 and w > 10 and h / w < 4:
            char_image = gray[y:y + h, x:x + w]
            resized_char = cv2.resize(char_image, (20, 20)).flatten()
            chars.append((x, resized_char))

    # Sortowanie konturów znaków według współrzędnej x
    chars = sorted(chars, key=lambda char: char[0])

    plate_text = ""
    for _, char_image in chars:
        char_image = np.array([char_image])
        predicted_char = model.predict(char_image)[0]
        plate_text += predicted_char  # Dodawanie rozpoznanych znaków do wyniku

    return plate_text  # Zwracanie rozpoznanego tekstu z tablicy rejestracyjnej


def process_image(image_path):
    model = load_model('rest_of_code/knn_model.pkl')
    image = load_image(image_path)

    contour = find_license_plate_contour(image)

    if contour is not None:
        plate_image = extract_license_plate(image, contour)
        plate_text = recognize_characters(plate_image, model)
        return plate_text  # Zwracanie rozpoznanego tekstu tablicy rejestracyjnej
    return ""  # Zwracanie pustego tekstu, jeśli nie znaleziono tablicy rejestracyjnej

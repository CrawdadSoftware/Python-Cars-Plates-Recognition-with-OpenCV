import cv2


def load_image(image_path):
    """
    Funkcja do ładowania obrazu z podanej ścieżki.

    Args:
        image_path (str): Ścieżka do pliku obrazu.

    Returns:
        image: Obraz załadowany za pomocą OpenCV.
    """
    # Ładowanie obrazu z podanej ścieżki
    image = cv2.imread(image_path)

    # Sprawdzenie, czy obraz został poprawnie załadowany
    if image is None:
        print(f"Nie można załadować obrazu: {image_path}")
    else:
        print(f"Obraz załadowany pomyślnie: {image_path}")

    # Zwracanie załadowanego obrazu
    return image


def preprocess_image(image):
    """
    Funkcja do przetwarzania obrazu.

    Args:
        image: Obraz załadowany za pomocą OpenCV.

    Returns:
        thresh: Obraz binarny po przetworzeniu.
    """
    # Konwersja obrazu do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Informacja o konwersji obrazu do skali szarości
    print("Obraz przekonwertowany do skali szarości.")

    # Binarizacja obrazu przy użyciu metody Otsu
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Informacja o binaryzacji obrazu
    print("Obraz zbinarny przy użyciu metody Otsu.")

    # Zwracanie przetworzonego obrazu
    return thresh


# Przykładowe wywołanie funkcji, aby pokazać, jak działają
if __name__ == "__main__":
    # Ścieżka do przykładowego obrazu
    example_image_path = 'path/to/example_image.jpg'

    # Ładowanie przykładowego obrazu
    example_image = load_image(example_image_path)

    # Przetwarzanie przykładowego obrazu, jeśli został poprawnie załadowany
    if example_image is not None:
        processed_image = preprocess_image(example_image)

        # Wyświetlanie oryginalnego i przetworzonego obrazu
        cv2.imshow('Oryginalny obraz', example_image)
        cv2.imshow('Przetworzony obraz', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

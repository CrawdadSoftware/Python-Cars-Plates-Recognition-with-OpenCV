import sys
import os
import json
from rest_of_code.processing import process_image


def main(image_folder, output_file):
    """
    Główna funkcja programu.

    Args:
        image_folder (str): Ścieżka do folderu z obrazami.
        output_file (str): Ścieżka do pliku wynikowego JSON.

    Tworzy słownik mapujący nazwy plików obrazów na rozpoznane teksty tablic rejestracyjnych
    i zapisuje go do pliku JSON.
    """
    # Inicjalizacja pustego słownika do przechowywania wyników
    result = {}

    # Iteracja po wszystkich plikach w folderze obrazów
    for image_name in os.listdir(image_folder):
        # Sprawdzenie, czy plik jest obrazem JPEG
        if image_name.endswith('.jpg'):
            # Pełna ścieżka do pliku obrazu
            image_path = os.path.join(image_folder, image_name)

            # Przetwarzanie obrazu w celu rozpoznania tablicy rejestracyjnej
            plate_text = process_image(image_path)

            # Dodanie wyniku do słownika
            result[image_name] = plate_text

    # Zapisanie wyników do pliku JSON
    with open(output_file, 'w') as f:
        json.dump(result, f)

    # Informacja o zakończeniu procesu zapisu
    print(f"Wyniki zostały zapisane do pliku {output_file}")


if __name__ == "__main__":
    """
    Punkt wejścia programu.

    Sprawdza argumenty wiersza poleceń i uruchamia główną funkcję programu.
    """
    # Sprawdzenie, czy podano odpowiednią liczbę argumentów
    if len(sys.argv) != 3:
        # Wyświetlenie informacji o użyciu skryptu
        print("Usage: python Crawdad_Software.py <image_folder> <output_file>")

        # Zakończenie programu z kodem błędu
        sys.exit(1)

    # Pobranie ścieżki do folderu z obrazami z argumentów wiersza poleceń
    image_folder = sys.argv[1]

    # Pobranie ścieżki do pliku wynikowego z argumentów wiersza poleceń
    output_file = sys.argv[2]

    # Uruchomienie głównej funkcji programu
    main(image_folder, output_file)

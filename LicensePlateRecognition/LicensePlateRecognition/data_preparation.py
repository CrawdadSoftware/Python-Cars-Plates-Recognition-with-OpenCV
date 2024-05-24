import cv2
import os

def save_contour_images(image, contours, output_folder, image_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    idx = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20 and w > 20:  # Filter out small contours
            char_image = image[y:y + h, x:x + w]
            char_image = cv2.resize(char_image, (20, 20))
            char_folder = os.path.join(output_folder, image_name.split('.')[0])
            if not os.path.exists(char_folder):
                os.makedirs(char_folder)
            cv2.imwrite(os.path.join(char_folder, f"char_{idx}.png"), char_image)
            idx += 1

def main(image_folder, output_folder):
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            save_contour_images(thresh, contours, output_folder, image_name)


if __name__ == "__main__":
    main('data/train_1/train_1', 'data/chars')

import os
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm

dataset_path = r"C:\Users\user\Desktop\projects\Volleyball_detector\full_dataset_for_ball_models_and_kaggle"
image_dir = r'C:\Users\user\Desktop\projects\Volleyball_detector\full_dataset_for_ball_models_and_kaggle\images'
label_dir = r'C:\Users\user\Desktop\projects\Volleyball_detector\full_dataset_for_ball_models_and_kaggle\labels'

def filter_annotations(label_dir, target_class='2'):
    """
    Удаляет строки с определенным классом из всех аннотаций в директории.

    Args:
        label_dir (str): Путь к директории с аннотациями (с подпапками 'train' и 'val').
        target_class (str): Класс, который нужно удалить из аннотаций.

    Modifies:
        Перезаписывает файлы аннотаций без удаленного класса.
    """
    for split in ['train', 'val']:
        split_dir = os.path.join(label_dir, split)
        for filename in os.listdir(split_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(split_dir, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                filtered_lines = [line for line in lines if not line.startswith(target_class + ' ')]
                with open(file_path, 'w') as file:
                    file.writelines(filtered_lines)

def replace_class_in_annotations(label_dir, old_class='1', new_class='0'):
    """
    Заменяет все вхождения одного класса на другой в файлах аннотаций.

    Args:
        label_dir (str): Путь к директории с аннотациями (с подпапками 'train' и 'val').
        old_class (str): Класс, который нужно заменить.
        new_class (str): Новый класс, который будет подставлен.

    Modifies:
        Перезаписывает файлы аннотаций с обновленным классом.
    """
    for split in ['train', 'val']:
        split_dir = os.path.join(label_dir, split)
        for filename in os.listdir(split_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(split_dir, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                replaced_lines = [line.replace(f'{old_class} ', f'{new_class} ', 1) for line in lines]
                with open(file_path, 'w') as file:
                    file.writelines(replaced_lines)

def remove_orphan_images(label_dir, image_dir):
    """
    Удаляет изображения, для которых соответствующие аннотации пустые.

    Args:
        label_dir (str): Путь к папке с аннотациями.
        image_dir (str): Путь к папке с изображениями.

    Modifies:
        Удаляет ненужные изображения из папок train и val.
    """
    for split in ['train', 'val']:
        label_split_dir = os.path.join(label_dir, split)
        image_split_dir = os.path.join(image_dir, split)
        for filename in os.listdir(label_split_dir):
            if filename.endswith('.txt'):
                label_path = os.path.join(label_split_dir, filename)
                if os.path.getsize(label_path) == 0:
                    base_name = os.path.splitext(filename)[0]
                    for img_file in os.listdir(image_split_dir):
                        if img_file.startswith(base_name):
                            os.remove(os.path.join(image_split_dir, img_file))
                            print(f"Удалено изображение: {img_file}")

def visualize_annotations(label_dir, image_dir):
    """
    Визуализирует аннотации на изображениях, рисуя прямоугольники вокруг объектов.

    Args:
        label_dir (str): Путь к папке с аннотациями.
        image_dir (str): Путь к папке с изображениями.

    Shows:
        Отображает изображение с наложенными прямоугольниками для каждого объекта.
    """
    for split in ['train', 'val']:
        label_split_dir = os.path.join(label_dir, split)
        image_split_dir = os.path.join(image_dir, split)
        for filename in os.listdir(label_split_dir):
            if filename.endswith('.txt'):
                label_path = os.path.join(label_split_dir, filename)
                image_name = filename.replace('.txt', '.jpg')
                image_path = os.path.join(image_split_dir, image_name)
                if not os.path.exists(image_path):
                    continue
                
                image = cv2.imread(image_path)
                height, width, _ = image.shape

                with open(label_path, 'r') as file:
                    lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, w, h = map(float, parts)
                    x1 = int((x_center - w / 2) * width)
                    y1 = int((y_center - h / 2) * height)
                    x2 = int((x_center + w / 2) * width)
                    y2 = int((y_center + h / 2) * height)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                cv2.imshow('Annotated Image', image)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return
        cv2.destroyAllWindows()



def resize_images(image_dir, target_size=(640, 640)):
    """
    Масштабирует все изображения в указанной папке до заданного размера.

    Args:
        image_dir (str): Путь к папке с изображениями.
        target_size (tuple): Размер (ширина, высота), до которого нужно изменить изображения.

    Modifies:
        Перезаписывает файлы изображений в новом размере.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    
    for image_file in tqdm(image_files, desc=f"Resizing images in {image_dir}"):
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Ошибка чтения файла: {image_path}")
            continue
        
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(image_path, img_resized)


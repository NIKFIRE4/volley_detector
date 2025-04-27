import os
import shutil

def merge_datasets(source_dirs, target_dir):
    # Создаем целевые директории
    for split in ["train", "val"]:
        os.makedirs(os.path.join(target_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "labels", split), exist_ok=True)

    counter = 1  # Счётчик для именования файлов

    for split in ["train", "val"]:
        img_target = os.path.join(target_dir, "images", split)
        lbl_target = os.path.join(target_dir, "labels", split)

        for source in source_dirs:
            img_source = os.path.join(source, "images", split)
            lbl_source = os.path.join(source, "labels", split)

            # Проверяем, существуют ли директории
            if not os.path.exists(img_source) or not os.path.exists(lbl_source):
                print(f"Пропуск: {source} не содержит {split}")
                continue

            for filename in os.listdir(img_source):
                base, ext = os.path.splitext(filename)
                new_filename = f"{counter:06d}{ext}"  # Пример: 000001.jpg
                new_labelname = f"{counter:06d}.txt"

                # Копируем изображение
                shutil.copy2(os.path.join(img_source, filename), os.path.join(img_target, new_filename))

                # Копируем соответствующую аннотацию, если она есть
                label_file = f"{base}.txt"
                if os.path.exists(os.path.join(lbl_source, label_file)):
                    shutil.copy2(
                        os.path.join(lbl_source, label_file),
                        os.path.join(lbl_target, new_labelname)
                    )
                else:
                    print(f"⚠️ Внимание: Для изображения {filename} не найдена аннотация.")

                counter += 1

    print(f"✅ Объединение завершено. Всего файлов: {counter - 1}")

# Путь к исходным датасетам
source_dirs = [r"full_dataset_for_ball_models", r"append_datasets\dataset_kaggle"]

# Путь к целевой директории
merge_datasets(source_dirs, "full_dataset_for_ball_models_and_kaggle2")

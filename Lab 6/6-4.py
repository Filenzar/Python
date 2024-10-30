import cv2
import numpy as np

# Функция для создания преобразований изображения
def generate_transformations(img):
    transformations = []
    transformations.append(img)  # Оригинальное изображение
    # Повороты
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1)
        transformations.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
    # Отражения
    transformations.append(cv2.flip(img, 1))  # По вертикали
    transformations.append(cv2.flip(img, 0))  # По горизонтали
    return transformations

# 1. Загрузка изображений
img_scene = cv2.imread('Halloween/lab7.png')  # Изображение, где будем искать призраков
ghost_images = ['Halloween/candy_ghost.png', 'Halloween/pampkin_ghost.png', 'Halloween/scary_ghost.png']  # Список изображений призраков

# Преобразование в серый цвет для SIFT
img_scene_gray = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# Создаем объект SIFT
sift = cv2.SIFT_create()

# 2. Обнаружение ключевых точек и вычисление дескрипторов для сцены
keypoints_scene, descriptors_scene = sift.detectAndCompute(img_scene_gray, None)

# 3. Цикл по всем изображениям призраков
for ghost_image_path in ghost_images:
    img_ghost = cv2.imread(ghost_image_path)
    transformations = generate_transformations(img_ghost)

    for img_ghost_transformed in transformations:
        img_ghost_gray = cv2.cvtColor(img_ghost_transformed, cv2.COLOR_BGR2GRAY)

        # Обнаружение ключевых точек и вычисление дескрипторов для призрака
        keypoints_ghost, descriptors_ghost = sift.detectAndCompute(img_ghost_gray, None)

        # Сопоставление ключевых точек с использованием FLANN
        index_params = dict(algorithm=1, trees=5)  # KDTREE
        search_params = dict(checks=50)  # Опции поиска

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_ghost, descriptors_scene, k=2)

        # Отбор хороших совпадений
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Отбор по критерию Левенштейна
                good_matches.append(m)

        # 4. Нахождение гомографии
        if len(good_matches) > 4:  # Нужно минимум 4 совпадения для нахождения гомографии
            src_pts = np.float32([keypoints_ghost[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Найти гомографию с помощью RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

            # Применить гомографию к углам изображения призрака
            h, w = img_ghost_gray.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Выделить призрака на изображении
            img_scene = cv2.polylines(img_scene, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

# Показать финальные результаты
cv2.imshow('Detected Ghosts', img_scene)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import random

# Загрузка изображения
img_path = 'Vanya.jpg'  # Укажите путь к вашему изображению
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразование из BGR в RGB

# Функция для генерации аугментированных изображений
def image_augmentation(image, target_size=(200, 200)):
    # Поворот на случайный угол
    angle = random.uniform(-30, 30)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Отражение по вертикали или горизонтали
    if random.choice([True, False]):
        rotated = cv2.flip(rotated, 0)  # Вертикальное отражение
    if random.choice([True, False]):
        rotated = cv2.flip(rotated, 1)  # Горизонтальное отражение

    # Вырезание части изображения
    h, w = rotated.shape[:2]
    x = random.randint(0, w - target_size[1])
    y = random.randint(0, h - target_size[0])
    cropped = rotated[y:y + target_size[0], x:x + target_size[1]]

    # Размытие
    if random.choice([True, False]):
        cropped = cv2.GaussianBlur(cropped, (5, 5), 0)

    # Изменение размера
    augmented_image = cv2.resize(cropped, target_size)

    return augmented_image

# Настройка фигуры
fig, ax_image = plt.subplots()
plt.subplots_adjust(bottom=0.25)
img_display = ax_image.imshow(img)

# Состояния фильтров
apply_gaussian = False
apply_bw = False
apply_canny = False
current_image = img.copy()  # Хранение текущего изображения

# Слайдеры для RGB каналов
ax_r = plt.axes([0.15, 0.05, 0.65, 0.03])
ax_g = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_b = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_gaus = plt.axes([0.05, 0.7, 0.15, 0.05])
ax_bw = plt.axes([0.05, 0.6, 0.15, 0.05])
ax_res = plt.axes([0.05, 0.3, 0.15, 0.05])
ax_gen = plt.axes([0.05, 0.5, 0.15, 0.05])
ax_canny = plt.axes([0.05, 0.4, 0.15, 0.05])

# Инициализация слайдеров
s_r = Slider(ax_r, label="R channel", valmin=0, valmax=255, valinit=128)
s_g = Slider(ax_g, label="G channel", valmin=0, valmax=255, valinit=128)
s_b = Slider(ax_b, label="B channel", valmin=0, valmax=255, valinit=128)

# Кнопки
b_gf = Button(ax_gaus, 'Gaussian Blur')
b_bw = Button(ax_bw, 'Black & White')
b_res = Button(ax_res, 'Reset')
b_gen = Button(ax_gen, 'Augmented')
b_canny = Button(ax_canny, 'Canny Edges')

# Обновление изображения
def update(val):
    modified_img = current_image.copy()
    r = s_r.val
    g = s_g.val
    b = s_b.val

    # Применение слайдеров
    modified_img[..., 0] = np.clip(modified_img[..., 0] + r - 128, 0, 255)
    modified_img[..., 1] = np.clip(modified_img[..., 1] + g - 128, 0, 255)
    modified_img[..., 2] = np.clip(modified_img[..., 2] + b - 128, 0, 255)

    # Применение фильтров
    if apply_gaussian:
        modified_img = cv2.GaussianBlur(modified_img, (15, 15), 0)
    if apply_bw:
        bw_img = cv2.cvtColor(modified_img, cv2.COLOR_RGB2GRAY)
        modified_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2RGB)
    if apply_canny:
        edges = cv2.Canny(modified_img, 100, 200)
        modified_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        count_contours(modified_img)  # Подсчет контуров

    img_display.set_data(modified_img)
    plt.draw()

# Генерация аугментированного изображения
def generate_augmented_image(event):
    global current_image
    # Сохраняем текущее состояние
    r = s_r.val
    g = s_g.val
    b = s_b.val
    gaussian_state = apply_gaussian
    bw_state = apply_bw
    canny_state = apply_canny

    # Генерируем аугментированное изображение
    current_image = image_augmentation(img)
    
    # Применяем сохраненные настройки
    modified_img = current_image.copy()
    modified_img[..., 0] = np.clip(modified_img[..., 0] + r - 128, 0, 255)
    modified_img[..., 1] = np.clip(modified_img[..., 1] + g - 128, 0, 255)
    modified_img[..., 2] = np.clip(modified_img[..., 2] + b - 128, 0, 255)

    if gaussian_state:
        modified_img = cv2.GaussianBlur(modified_img, (15, 15), 0)
    if bw_state:
        bw_img = cv2.cvtColor(modified_img, cv2.COLOR_RGB2GRAY)
        modified_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2RGB)
    if canny_state:
        edges = cv2.Canny(modified_img, 100, 200)
        modified_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    img_display.set_data(modified_img)
    plt.draw()

# Применение фильтров
def toggle_gaussian(event):
    global apply_gaussian
    apply_gaussian = not apply_gaussian
    update(None)

def toggle_black_white(event):
    global apply_bw
    apply_bw = not apply_bw
    update(None)

def toggle_canny(event):
    global apply_canny
    apply_canny = not apply_canny
    update(None)

def reset(event):
    s_r.set_val(128)
    s_g.set_val(128)
    s_b.set_val(128)
    global apply_gaussian, apply_bw, apply_canny, current_image
    apply_gaussian = False
    apply_bw = False
    apply_canny = False
    current_image = img.copy()  # Сбрасываем текущее изображение
    img_display.set_data(current_image)
    plt.draw()

def count_contours(image):
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Нахождение контуров
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Отображение контуров на изображении
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)  # Красные контуры
    img_display.set_data(contour_image)
    
    # Подсчет контуров
    print(f'Количество найденных контуров: {len(contours)}')

# Подключение событий
s_r.on_changed(update)
s_g.on_changed(update)
s_b.on_changed(update)
b_gf.on_clicked(toggle_gaussian)
b_bw.on_clicked(toggle_black_white)
b_canny.on_clicked(toggle_canny)
b_res.on_clicked(reset)
b_gen.on_clicked(generate_augmented_image)

plt.show()

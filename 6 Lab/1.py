import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, Button 
import random  

# Загрузка изображения
img_path = 'Vanya.jpg'  # Укажите путь к вашему изображению
img = cv2.imread(img_path)  # Чтение изображения с указанного пути
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразование из BGR в RGB

# Установим размер выходного изображения
output_size = (224, 224)  # Например, 224x224

# Настройка фигуры
fig, ax_image = plt.subplots()  # ax_image ось, fig фигура (окно, где будет изображение)
plt.subplots_adjust(bottom=0.35)  # Настройка отступов для размещения слайдеров и кнопок
img_display = ax_image.imshow(img)  # Отображение изображения в оси ax_image
ax_image.axis('on')  # Вкл/выкл оси

# Слайдеры для RGB каналов
ax_r = plt.axes([0.15, 0.05, 0.65, 0.03])  # Ось для слайдера красного канала
ax_g = plt.axes([0.15, 0.10, 0.65, 0.03])  # Ось для слайдера зеленого канала
ax_b = plt.axes([0.15, 0.15, 0.65, 0.03])  # Ось для слайдера синего канала

# Оси для кнопок и других фильтров
ax_gaus = plt.axes([0.05, 0.6, 0.15, 0.05])  # Ось для кнопки Гауссова размытия
ax_bw = plt.axes([0.05, 0.5, 0.15, 0.05])  # Ось для кнопки черно-белого изображения
ax_res = plt.axes([0.05, 0.4, 0.15, 0.05])  # Ось для кнопки сброса
ax_update = plt.axes([0.05, 0.3, 0.15, 0.05])  # Ось для кнопки обновления изображения
ax_contours = plt.axes([0.05, 0.7, 0.15, 0.05])  # Ось для кнопки поиска контуров
ax_sharpen = plt.axes([0.05, 0.2, 0.15, 0.05])  # Ось для кнопки повышения резкости

# Инициализация слайдеров (ползунков) для RGB каналов
s_r = Slider(ax_r, label="R channel", valmin=0, valmax=255, valinit=128)  # Слайдер для красного канала
s_g = Slider(ax_g, label="G channel", valmin=0, valmax=255, valinit=128)  # Слайдер для зеленого канала
s_b = Slider(ax_b, label="B channel", valmin=0, valmax=255, valinit=128)  # Слайдер для синего канала

# Кнопки для фильтров и операций
b_gf = Button(ax_gaus, 'Gaussian Blur')  # Кнопка для применения Гауссова размытия
b_bw = Button(ax_bw, 'Black & White')  # Кнопка для перевода изображения в черно-белое
b_res = Button(ax_res, 'Reset')  # Кнопка для сброса значений
b_update = Button(ax_update, 'Random Image')  # Кнопка для обновления изображения
b_contours = Button(ax_contours, 'Find Contours')  # Кнопка для поиска контуров
b_sharpen = Button(ax_sharpen, 'Sharpen Image')  # Кнопка для повышения резкости

# Функция для повышения резкости изображения
def sharpen_image(img):
    """Применение эффекта резкости (Sharpen) с использованием разницы между изображением и его размытой версией."""
    
    # Если изображение в градациях серого, преобразуем в RGB
    if img.ndim == 2:  # Если изображение в градациях серого (один канал)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Преобразуем в RGB
    
    # Применяем Гауссово размытие
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    # Добавляем разницу между исходным изображением и размытым
    sharpened_img = cv2.addWeighted(img, 2.5, blurred_img, -1.5, 0)
    
    return sharpened_img

# Обновление изображения с использованием значений слайдеров
def update(val):
    r = s_r.val
    g = s_g.val
    b = s_b.val
    modified_img = img.copy()
    modified_img[..., 0] = np.clip(modified_img[..., 0] + r - 128, 0, 255)
    modified_img[..., 1] = np.clip(modified_img[..., 1] + g - 128, 0, 255)
    modified_img[..., 2] = np.clip(modified_img[..., 2] + b - 128, 0, 255)
    img_display.set_data(modified_img)  # Обновляем отображаемое изображение
    plt.draw()  # Перерисовываем изображение

# Применение Гауссова размытия
def apply_gaussian_blur(event):
    blurred_img = cv2.GaussianBlur(img, (15, 15), 0)  # Применяем размытие
    img_display.set_data(blurred_img)  # Обновляем изображение
    plt.draw()

# Применение черно-белого фильтра
def apply_black_white(event):
    bw_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Преобразуем изображение в черно-белое
    # Для отображения черно-белого изображения в цветном формате
    bw_img_colored = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2RGB)
    img_display.set_data(bw_img_colored)  # Обновляем изображение
    plt.draw()

# Сброс значений слайдеров и изображения
def reset(event):
    s_r.set_val(128)  # Сбрасываем слайдер для красного канала
    s_g.set_val(128)  # Сбрасываем слайдер для зеленого канала
    s_b.set_val(128)  # Сбрасываем слайдер для синего канала
    img_display.set_data(img)  # Восстанавливаем исходное изображение
    plt.draw()

# Обновление изображения при нажатии на кнопку
def update_image(event):
    """Функция для обновления изображения при нажатии на кнопку."""
    try:
        img_display.set_data(next(gen))  # Загружаем изображение
        plt.draw()
    except StopIteration:
        print("Генерация изображений завершена.")

# Поиск и отображение контуров
def find_contours(event):
    """Функция для поиска и отображения контуров."""
    img_with_contours, contours = find_and_draw_contours(img)  # Ищем и рисуем контуры
    img_display.set_data(img_with_contours)  # Обновляем изображение
    contour_count = count_contours(contours)  # Подсчитываем количество контуров
    print(f"Найдено контуров: {contour_count}")
    plt.draw()

# Применение фильтра резкости
def apply_sharpen(event):
    """Применение фильтра резкости при нажатии на кнопку."""
    sharpened_img = sharpen_image(img)  # Применяем повышение резкости
    img_display.set_data(sharpened_img)  # Обновляем изображение
    plt.draw()

# Генератор случайных изменений изображения
def random_augmentation(img):
    # Случайно выбрать операцию
    operations = ['rotate', 'flip_vertical', 'flip_horizontal', 'crop', 'blur']
    operation = random.choice(operations)

    if operation == 'rotate':
        # Поворот на случайный угол
        angle = random.randint(-70, 70)
        return rotate_image(img, angle)
    elif operation == 'flip_vertical':
        # Отражение по вертикали
        return cv2.flip(img, 0)  # 0 - вертикальное отражение
    elif operation == 'flip_horizontal':
        # Отражение по горизонтали
        return cv2.flip(img, 1)  # 1 - горизонтальное отражение
    elif operation == 'crop':
        # Вырезание части изображения
        return random_crop(img)
    elif operation == 'blur':
        # Применение размытия
        return cv2.GaussianBlur(img, (15, 15), 0)

# Применение случайных изменений
def rotate_image(img, angle):
    """Функция для поворота изображения на заданный угол."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

# Случайная обрезка изображения
def random_crop(img):
    """Функция для случайной обрезки изображения."""
    h, w = img.shape[:2]
    top = random.randint(0, h // 2)
    left = random.randint(0, w // 2)
    bottom = random.randint(h // 2, h)
    right = random.randint(w // 2, w)
    cropped_img = img[top:bottom, left:right]
    return cropped_img

# Генератор изображений для аугментаций
def image_generator(img):
    while True:
        augmented_img = random_augmentation(img)
        yield augmented_img


def find_and_draw_contours(img):
    """Находит контуры на изображении и рисует их."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #Параметр cv2.RETR_EXTERNAL указывает, что нужно искать только внешние контуры объектов (в отличие от вложенных контуров).
    #Параметр cv2.CHAIN_APPROX_SIMPLE позволяет использовать упрощенное представление контуров, сохраняя только важные точки (например, углы).
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 0, 0), 1)
    return img_with_contours, contours

def count_contours(contours):
    """Подсчитывает количество найденных контуров."""
    return len(contours)

# Генератор для изображений с аугментациями
gen = image_generator(img)  # Генератор для создания новых изображений с аугментациями

# Привязка функций к событиям слайдеров и кнопок
s_r.on_changed(update)
s_g.on_changed(update)
s_b.on_changed(update)
b_gf.on_clicked(apply_gaussian_blur)
b_bw.on_clicked(apply_black_white)
b_res.on_clicked(reset)
b_update.on_clicked(update_image)
b_contours.on_clicked(find_contours)
b_sharpen.on_clicked(apply_sharpen)

# Показ графического интерфейса
plt.show()

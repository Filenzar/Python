import sys
import numpy as np

def read_file(file_path):
    return np.loadtxt(file_path, dtype=int)

def mix_data(real_data, synthetic_data, probability):
    # Генерация случайных чисел и выбор данных
    random_values = np.random.rand(len(real_data)) # Рандом чисел от 0 до 1
    mixed_data = np.where(random_values < probability, synthetic_data, real_data) #1-Проверка, 2-True, 3-False
    return mixed_data

def mix_data_third(real_data, synthetic_data, probability):
    # Перемешивание обоих массивов
    shuffled_real = np.random.permutation(real_data)
    shuffled_synthetic = np.random.permutation(synthetic_data)
    
    # Генерация случайных чисел для выбора
    random_values = np.random.rand(len(real_data))
    mixed_data = np.where(random_values < probability, shuffled_synthetic, shuffled_real)
    return mixed_data

def main(real_file, synthetic_file, probability):
    real_data = read_file(real_file)
    synthetic_data = read_file(synthetic_file)

    if real_data.shape != synthetic_data.shape:
        raise ValueError("Файлы должны содержать одинаковое количество чисел.")

    # Способ 1
    mixed_data_1 = mix_data(real_data, synthetic_data, probability)
    print("Способ 1:", mixed_data_1)

    # Способ 2 
    indices = np.arange(real_data.shape[0]) # [0,1,2,3,4,...]
    np.random.shuffle(indices)
    random_values2 = np.random.rand(len(real_data))
    mixed_data_2 = np.where(random_values2 < probability, synthetic_data[indices], real_data[indices])
    print("Способ 2:", mixed_data_2)

    # Способ 3
    mixed_data_3 = mix_data_third(real_data, synthetic_data, probability)
    print("Способ 3:", mixed_data_3)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Использование: python random_select.py <file_1> <file_2> <P>")
        sys.exit(1)

    file_1 = sys.argv[1]
    file_2 = sys.argv[2]
    probability = float(sys.argv[3])

    if not (0 <= probability <= 1):
        raise ValueError("Вероятность должна быть в диапазоне от 0 до 1.")

    main(file_1, file_2, probability)

#python random_select.py file_1.txt file_2.txt 0.3

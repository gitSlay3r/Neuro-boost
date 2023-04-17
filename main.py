import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import os
import csv
import statistics


def find_files(directory, extension):
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for file in files:
            if file.endswith(extension):
                path = os.path.join(root, file)
                if f'result{extension}' in path:
                    yield path


directory = '/Users/' ## директория для поиска файолв
extension = '.log' ## расширение файла для поиска

columns = ['MBERRLIM', 'TOLLIN', 'TOLVARNEWT', 'CHECKP', 'MBERRCTRL', 'Time']
df = pd.DataFrame(columns=columns)

count = 0
for file_path in find_files(directory, extension):
    # print(file_path)
    with open(file_path, 'r') as f:
        # print(file_path)
        contents = f.read()
        try:
            keyword1_values = [value for value in contents.split('\n') if 'MBERRLIM' in value][0].split(" ")[-1]
            keyword2_values = [value for value in contents.split('\n') if 'TOLLIN' in value][0].split(" ")[-1]
            keyword3_values = [value for value in contents.split('\n') if 'TOLVARNEWT' in value][0].split(" ")[-1]
            keyword4_values = [value for value in contents.split('\n') if 'CHECKP' in value][0].split(" ")[-1]
            keyword5_values = [value for value in contents.split('\n') if 'MBERRCTRL' in value][0].split(" ")[-1]
            keyword6_values = [value for value in contents.split('\n') if 'Полное время ЦП' in value][0].split(" ")[-1][
                              :-1]
            time_col = keyword6_values.split(".")
            extracted_values = {
                'MBERRLIM': float(keyword1_values),
                'TOLLIN': float(keyword2_values),
                'TOLVARNEWT': float(keyword3_values),
                'CHECKP': float(keyword4_values),
                'MBERRCTRL': float(keyword5_values),
                'CPU_Time': keyword6_values,
                'Time': 3600*int(time_col[0]) + 60*int(time_col[1]) + int(time_col[2])
            }
            df.loc[count] = [extracted_values[column] for column in columns]
            count += 1
        except IndexError:
            pass


df = df.reset_index(drop=True)
# print(count)
# print(df)


def from_RSM_to_numpy(f):
    result_list = f.split('\\r\\n')
    lst = []
    for i in result_list:
        for j in i.split(' '):
            try:
                lst.append(float(j))
            except ValueError:
                pass
    return lst


def max_value(lst1, lst2):
    max = 0
    min_len = min(len(lst1), len(lst2))
    for i in range(min_len):
        x = abs(lst1[i] - lst2[i]) / lst1[i] if lst1[i] != 0 else 0
        if x > max:
            max = x
    return max

extension = '.RSM'

column = ['Max_error']
df1 = pd.DataFrame(columns=column)

count = 0
iter_files = (find_files(directory, extension))
iter_files = list(iter_files)

true_path = iter_files[0]
with open(true_path, 'r') as f:
    true_result = f.read()
true_list = from_RSM_to_numpy(true_result)

for file_path in iter_files[1:]:
    # print(file_path)
    with open(file_path, 'r') as f:
        # print(file_path)
        model_result = f.read()
    model_list = from_RSM_to_numpy(model_result)
    error = max_value(true_list, model_list)
    df1.loc[count] = error
    count += 1


df1 = df1.reset_index(drop=True)
# print(count)
# print(df1)
print(a)
df_combined = pd.concat([df, df1], axis=1)

## поиск и вытаскивание данных из сетки grid
filename = 'poro'
extension = '.GRDECL'
filename2 = 'perm'


def read_array(directory, filename):
    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            type(file)
            if file.lower().startswith(filename):
                # yield path
                array = []
                print(path)
                with open(f'{path}', 'r') as f:
                    array = []
                    line = f.readline()
                    length = len(filename)
                    while line[:length] != filename.upper():
                        line = f.readline()
                    for i, line in enumerate(f.readlines()):
                        if line[:2] == '--':
                            continue
                        splitted = line.strip().split(' ')
                        for num in splitted:
                            if '-9999.25' not in num:
                                if num.find('*') != -1:
                                    count, value = num.split('*')
                                    count, value = int(count), float(value)
                                    array += [value] * count
                                elif num == '/' or num.strip() == '':
                                    continue
                                else:
                                    array.append(float(num))
                    return array


# find_files_filename(directory, extension, filename)
data = read_array(directory, filename)
data2 = read_array(directory, filename2)

data = np.array(data)
data2 = np.array(data2)

print(data)
print(data2)
# -возможное сжатие сетки-
#%%
print(max(data))
data2 = 0.7572e-7 * pow((data * 100), 5.894) ## в случае отсутствия permx производится расчет по этой формуле
#%%
# if data2 is None:
#     data2 = 0.7572e-7 * pow((data * 100), 5.894)
print(type(data2))
print(data2)
mean = np.mean(data)
std = np.std(data)
median = statistics.median(data)
max_value = max(data)
# min_value = min(data)
quan80 = np.quantile(data, 0.8)
# data_norm = (data - mean) / std

mean2 = np.mean(data2)
std2 = np.std(data2)
median2 = statistics.median(data2)
max_value2 = max(data2)
# min_value2 = min(data2)
quan80_2 = np.quantile(data2, 0.8)
print(quan80_2)
# data_norm2 = (data2 - mean2) / std2

## статистика
columns = ['mean_poro', 'std_poro', 'median_poro', 'max_poro', 'quan80_poro',
            'mean_perm', 'std_perm', 'median_perm', 'max_perm',  'quan80_perm']

df_combined['mean_poro'] = np.array(mean) * len(df_combined['Time'])
df_combined['std_poro'] = np.array(std) * len(df_combined['Time'])
df_combined['median_poro'] = np.array(median) * len(df_combined['Time'])
df_combined['max_poro'] = np.array(max_value) * len(df_combined['Time'])
df_combined['quan80_poro'] = np.array(quan80) * len(df_combined['Time'])

df_combined['mean_perm'] = np.array(mean2) * len(df_combined['Time'])
df_combined['std_perm'] = np.array(std2) * len(df_combined['Time'])
df_combined['median_perm'] = np.array(median2) * len(df_combined['Time'])
df_combined['max_perm'] = np.array(max_value2) * len(df_combined['Time'])
df_combined['quan80_poro'] = np.array(quan80_2) * len(df_combined['Time'])

df_combined.to_csv('Data.csv', index=False) ## датасет для модели

import os
#
# path = 'F:/Code/DIC-dataset/Deep_DIC_dataset/gt3_1/'
#
# f = os.listdir(path)
# sorted_f = sorted(f, key=lambda x: int(x.split('_')[2].split('.')[0]))
# n = 0
# for i in sorted_f:
#     oldname = path+sorted_f[n]
#     newname = path + 'train_image_' + str(n+1) + '.mat'
#     os.rename(oldname, newname)
#     n += 1
# print('ok')
#
# images_path = 'F:/Code/DIC-dataset/Deep_DIC_dataset_images/imgs3/'
# f = os.listdir(images_path)
# sorted_f = sorted(f, key=lambda x: int(x.split('_')[2].split('.')[0]))
# n_l = 0
# n_r = 0
# n = 0
# for i in sorted_f:
#     if sorted_f[n].endswith('1.png'):
#         oldname = images_path+sorted_f[n]
#         newname = images_path + 'train_image_' + str(n_l+1) + '_1.png'
#         os.rename(oldname, newname)
#         n_l += 1
#     elif sorted_f[n].endswith('2.png'):
#         oldname = images_path+sorted_f[n]
#         newname = images_path + 'train_image_' + str(n_r+1) + '_2.png'
#         os.rename(oldname, newname)
#         n_r += 1
#     n += 1
# print('ok')

dataset_path = ''
filename = "validation_dataset.txt"
mynumbers = []
with open(filename) as f:
    for line in f:
        item = line.strip().split('\n')
        for subitem in item:
            print(subitem)
            mynumbers.append(subitem)
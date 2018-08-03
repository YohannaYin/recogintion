#coding:utf-8
import os
# 创建train.list文件
dic = {}
train_path = '/home/hong/align/train_baseline/crop_images_DB/'
train_list = open('/home/hong/align/train_baseline/train100.list', 'w')
train_list.close()
count = 0
for i in os.listdir(train_path):
    if i != 'train.txt' and i != 'data_train.txt':
        count =count + 1
        if count<100:
            for f in os.listdir(os.path.join(train_path, i)):
                with open('/home/hong/align/train_baseline/train100.list', 'a') as l:
                    l.write(train_path + i+'/' +f+ "\t" + str(count) +"\n")

# print(count)
#创建test.list文件
# test_path = '/home/kesci/work/align/test/crop_images_DB/'
# test_list = open('/home/kesci/work/test.list', 'w')
# test_list.close()
# for i in os.listdir(test_path):
#     if i != 'train.txt' and i != 'data_train.txt':
#         with open('/home/kesci/work/test.list', 'a') as l:
#             l.write(test_path + i + "\n")

# #创建val.list文件
# val_path = '/home/hong/align/validate/crop_images_DB/'
# val_list = open('/home/hong/align/validate/val.list', 'w')
# val_list.close()
# count = 0
# for i in os.listdir(val_path):
#     count = count + 1
#     if count<3:
#         if i != 'train.txt' and i != 'data_train.txt' and i != 'data_validate.txt' and i != 'validate.txt' and i != '.DS_Store':
#             for f in os.listdir(os.path.join(val_path, i)):
#                 with open('/home/hong/align/validate/val1.list', 'a') as l:
#                     l.write(val_path + i + '/' + f + "\t" + str(count) + "\n")
#     if i != 'train.txt' and i != 'data_train.txt' and i!='data_validate.txt' and i!='validate.txt' and i!='.DS_Store':
#         for f in os.listdir(os.path.join(val_path, i)):
#             with open('/home/hong/align/validate/val.list', 'a') as l:
#                 l.write(val_path + i +'/'+f + "\t" + str(count) + "\n")
# recogintion
face-recognition
首先load.py是用来加载图片的
load_metadata_train用来加载训练集图片
align.py 人脸关键点检测包
model.py resnet网络,返回res5层(2048维),最后一个pool层(128维),fc层(用的softmax分类器)
data_list.py:myreader类,包含train_reader和val_reader,用于训练时训练数据和验证数据的读取
facial_landmarks.py:用于预处理数据,将图片先剪裁成128*128大小,并对齐(侧脸无法检测到68个关键点,直接resize),将图片存到另一个crop_DB包中
params_pass_46.tar.gz:训练过程的产生参数,用于inference阶段
shape_predictor_68_face_landmarks.dat: 68个人脸关键点
paddle_face_baseline.py:训练模型
infer.py:测试模型,可以直接粘贴到jupter notebook上运行

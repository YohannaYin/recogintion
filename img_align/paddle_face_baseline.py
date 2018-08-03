#coding:utf-8
import sys
import os
import paddle as paddle
from model import *
from data_list import *
from utils import *
import gzip
imageSize = 64
crop_size = 64
DATA_DIM = 1 * imageSize * imageSize
CLASS_DIM = 1037
BATCH_SIZE = 50
model_name = "baseline"


paddle.init(use_gpu=False, trainer_count=8)

# **********************获取参数***************************************
def get_parameters(parameters_path=None, cost=None):
    if not parameters_path:
        # 使用cost创建parameters
        if not cost:
            raise NameError('请输入cost参数')
        else:
            # 根据损失函数创建参数
            parameters = paddle.parameters.create(cost)
            print "cost"
            return parameters
    else:
        print(parameters_path)
        # 使用之前训练好的参数
        try:
            # 使用训练好的参数,.tar.gz格式的
            with gzip.open(parameters_path,'r') as f:
            #使用直接存放参数格式的 with open(parameters_path, 'r') as f:
                parameters = paddle.parameters.Parameters.from_tar(f)
            print "使用parameters"
            return parameters
        except Exception as e:
            raise NameError("你的参数文件错误,具体问题是:%s" % e)

# ***********************获取训练器***************************************
    # datadim 数据大小
def get_trainer(type_size, parameters_path):
    # 获得图片对于的信息标签
    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    label = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(type_size))

    em,fea, fc = resnet_baseline(image, CLASS_DIM)

    # 获取全连接层,也就是分类器
    out = fc

    # 获得损失函数
    cost = paddle.layer.classification_cost(input=out, label=label)

    # 获得参数
    if not parameters_path:
        parameters = paddle.parameters.create(cost)
    else:
        parameters = get_parameters(parameters_path=parameters_path)

    '''
    定义优化方法
    learning_rate 迭代的速度
    momentum 跟前面动量优化的比例
    regularzation 正则化,防止过拟合
    '''
    optimizer = paddle.optimizer.Momentum(momentum=0.9,
                                          regularization=paddle.optimizer.L2Regularization(rate=0.0002 * BATCH_SIZE),
                                          learning_rate=0.1 / BATCH_SIZE,
                                          learning_rate_schedule="pass_manual",
                                          learning_rate_args="20:1, 40:0.1, 60:0.01")

    '''
    创建训练器
    cost 分类器
    parameters 训练参数,可以通过创建,也可以使用之前训练好的参数
    update_equation 优化方法
    '''
    trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=optimizer)
    return trainer

def start_trainer(trainer, num_passes):
    save_parameters_name = 'model'
    myReader = MyReader(imageSize=imageSize, center_crop_size=crop_size)
    # 获得数据
    # reader = myReader.train_reader(train_list="/home/hong/align/train_baseline/train1.list")
    trainer_reader = myReader.train_reader(train_list="/home/hong/align/train_baseline/train.list")

    reader = paddle.batch(reader=paddle.reader.shuffle(reader=trainer_reader, buf_size=2048), batch_size=BATCH_SIZE)
    validate_reader = myReader.val_reader(val_list="/home/hong/align/validate/val1.list")
    # 保证保存模型的目录是存在的
    father_path = save_parameters_name[:save_parameters_name.rfind("/")]
    if not os.path.exists(father_path):
        os.makedirs(father_path)

    # 指定每条数据和padd.layer.data的对应关系
    feeding = {"image": 0, "label": 1}

    # 定义训练事件
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            # print(event)
            if event.batch_id % 10 == 0:
                print "\nPass %d, Batch %d, Cost %f, Error %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics['classification_error_evaluator'])
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # 测试准确率
            result = trainer.test(reader=paddle.batch(
                                                          reader=paddle.reader.shuffle(reader=trainer_reader,
                                                                                       buf_size=2048),
                                                          batch_size=BATCH_SIZE),feeding=feeding)
            print "\nTest with Pass %d, Classification_Error %s" % (
                event.pass_id, result.metrics['classification_error_evaluator'])
            with open('./record','w') as f:
                f.write(str(event.pass_id)+str(result.metrics['classification_error_evaluator']))
            with gzip.open('/home/hong/PycharmProjects/mypaddlepaddle/baseline/baseline/params_pass_%d.tar.gz' % (event.pass_id), 'w') as f:
                trainer.save_parameter_to_tar(f)
    trainer.train(reader=reader,num_passes=num_passes,event_handler=event_handler,feeding=feeding)



if __name__ == '__main__':
    trainer = get_trainer(type_size=1037,
                          parameters_path=None)
    start_trainer(trainer=trainer,num_passes=100)
    # main()
    # val(0,100)
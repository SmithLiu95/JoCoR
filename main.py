# -*- coding:utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
import argparse, sys
import datetime
from algorithm.jocor import JoCoR




parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[symmetric, smallCluster]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--co_lambda', type=float, default=0.1)
parser.add_argument('--model_type', type=str, help='[B,R]', default='B')



args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device= "cuda"

# Hyper Parameters
batch_size = 64
learning_rate = args.lr


from transform import build_transforms
# load dataset
if args.dataset=='CARS':
    input_channel=3
    init_epoch=0
    args.top_bn = False
    from data.metric_data import MetricDataSet
    if args.noise_type=='symmetric':
        prefix=''
    else:
        prefix='split9'
    img_source='/home/liuchang/PRISMv2/CARS_{}{}noised_train.csv'.format(args.noise_rate,prefix)
    train_dataset = MetricDataSet(img_source, transform=build_transforms(True),
                     mode="RGB",leng=-1,train=True)
    img_source='/home/liuchang/PRISMv2/CARS_test.csv'
    test_dataset = MetricDataSet(img_source, transform=build_transforms(False),
                    mode="RGB",leng=-1,train=False)
    img_source='/home/liuchang/PRISMv2/CARS_val.csv'
    val_dataset = MetricDataSet(img_source, transform=build_transforms(False),
                    mode="RGB",leng=-1,train=False)
    num_classes=train_dataset.nb_classes
if args.dataset == 'mnist':
    input_channel = 1
    num_classes = 10
    init_epoch = 0
    filter_outlier = True
    args.epoch_decay_start = 80
    args.model_type = "mlp"
    # args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                          download=True,
                          train=True,
                          transform=transforms.ToTensor(),
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate
                          )

    test_dataset = MNIST(root='./data/',
                         download=True,
                         train=False,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_type,
                         noise_rate=args.noise_rate
                         )

if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    init_epoch = 20
    args.epoch_decay_start = 80
    filter_outlier = True
    args.model_type = "cnn"
    # args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/',
                            download=True,
                            train=True,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

    test_dataset = CIFAR10(root='./data/',
                           download=True,
                           train=False,
                           transform=transforms.ToTensor(),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate
                           )

if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    init_epoch = 5
    args.epoch_decay_start = 100
    # args.n_epoch = 200
    filter_outlier = False
    args.model_type = "cnn"


    train_dataset = CIFAR100(root='./data/',
                             download=True,
                             train=True,
                             transform=transforms.ToTensor(),
                             noise_type=args.noise_type,
                             noise_rate=args.noise_rate
                             )

    test_dataset = CIFAR100(root='./data/',
                            download=True,
                            train=False,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate



def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    from ret_benchmark.data.collate_batch import collate_fn

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               collate_fn=collate_fn,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')


    # from model.bninception import bninception


    model = JoCoR(args, train_dataset, device, input_channel, num_classes)



    save_dir = args.result_dir +'/' +args.dataset+'/%s/' % args.model_type

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate)+ '_co' + str(args.co_lambda)+ '_lr' + str(args.lr)

    txtfile = save_dir + "/" + model_str + ".txt"
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))

    with open(txtfile, "a") as myfile:
        myfile.write(str(args)+'\n')
        myfile.write('iter train_acc1 train_acc2 val_p11 val_map1 val_p12 val_map2 test_acc1 test_map1 test_acc2 test_map2\n')

    # evaluate models with random weights
    val_p11,val_map1, val_p12, val_map2 = model.evaluate(val_loader)

    epoch = 0

    print(
        'Epoch [%d/%d] Val Accuracy on the %s validation images: Model1 P@1= %.4f MAP@R= %.4f Model2 P@1= %.4f MAP@R= %.4f' % (
            epoch + 1, args.n_epoch, len(test_dataset), val_p11,val_map1, val_p12, val_map2))



    best_val_p1=0
    best_test_p1=0
    best_test_map=0
    best_epoch=-1
    patience=10
    # training
    for epoch in range(1, args.n_epoch):
        # train models
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = model.train(train_loader, epoch)

        # evaluate models
        val_p11,val_map1, val_p12, val_map2 = model.evaluate(val_loader)

        # save results
        print(
            'Epoch [%d/%d] Train Accuracy: Model 1 Acc= %.4f %% Model2 Acc= %.4f %%. Val Accuracy on the %s validation images: Model1 P@1= %.4f MAP@R= %.4f Model2 P@1= %.4f MAP@R= %.4f' % (epoch + 1,  args.n_epoch, train_acc1, train_acc2, len(test_dataset), val_p11,val_map1, val_p12, val_map2))

        if best_val_p1<val_p11 or best_val_p1<val_p12:
            test_p11,test_map1, test_p12, test_map2 = model.evaluate(test_loader)
            if best_val_p1<val_p11:
                best_val_p1=val_p11
                best_test_p1=test_p11
                best_test_map=test_map1
                best_epoch=epoch
            if best_val_p1<val_p12:
                best_val_p1=val_p12
                best_test_p1=test_p12
                best_test_map=test_map2
                best_epoch=epoch
            patience=10
            print("[new best] epochs: %d, test_p1= %.4f, test_map= %.4f" % (best_epoch + 1, best_test_p1, best_test_map))
        else:
            test_p11,test_map1, test_p12, test_map2 = 0,0,0,0
            patience-=1
            if patience==0:
                break

        
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(val_p11) + " " + str(val_map1)+ " " +str(val_p12) + " " + str(val_map2) +' '  + str(test_p11) + " " + str(test_map1)+ " " +str(test_p12) + " " + str(test_map2) + "\n")

    print("best epochs: %d, test_p1= %.4f, test_map= %.4f" % (best_epoch + 1, best_test_p1, best_test_map))



    with open(txtfile, "a") as myfile:
        myfile.write("best epochs: %d, val_p1= %.4f, test_p1= %.4f, test_map= %.4f" % (best_epoch + 1, best_val_p1, best_test_p1, best_test_map))



if __name__ == '__main__':
    main()

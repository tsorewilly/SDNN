import torch
# import visdom
import numpy as np
import argparse
import os

from engine.trainer import train
from config.set_params import params as sp
from modeling.model import HARmodel
from utils.build_dataset import build_dataloader
from tensorboardX import SummaryWriter

import torch
# import visdom
import pandas as pd

from utils.preprocessing import HARdataset

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def main():
    """Driver file for training HAR model."""

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--checkpoint", default=None, type=str, )
    args = parser.parse_args()

    # params = sp().params

    model = HARmodel(params["input_dim"], params["num_classes"])

    if params["use_cuda"]:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"],
                                weight_decay=params["weight_decay"])

    params["start_epoch"] = 1

    # If checkpoint path is given, load checkpoint data
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))

            checkpoint = torch.load(args.checkpoint)
            params["start_epoch"] = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.checkpoint, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(params["resume"]))

    train_loader, val_loader, length = build_dataloader(params["root"], params)
    print(params["root"])
    # logger = visdom.Visdom()
    writer = SummaryWriter(log_dir=params["Checkpoint"], flush_secs=60)

    train(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion,
          optimizer=optimizer, params=params, writer=writer, length=length)  # logger=logger,


def build_testset(params):
    df = pd.read_csv(params["test"], low_memory=False)
    # parts = ["glove"]
    parts = ["muscle1", "muscle2", "muscle3", "muscle4", "glove"]
    variables = ["fIAV_{}", "fLogD_{}", "fMAV_{}", "fMAX_{}", "fNZM_{}", "fRMS_{}", "fSSC_{}",
                 "fVAR_{}", "fWA_{}", "fWL_{}", "fZC_{}", "fARC_{}", "fFME_{}", "fFMD_{}"]
    # AR_{}", "fWA_{}", "fWL_{}", "fZC_{}", "fARC_{}", "fFME_{}", "fFMD_{}", "EM_{}"]
    var_list = []
    for part in parts:
        for var in variables:
            var_list.append(list(df[var.format(part)]))
    #var_list.append(list(df["EM_muscle1"]))
    #var_list.append(list(df["EM_muscle2"]))
    var_list.append(list(df["EM_muscle3"]))
    var_list.append(list(df["EM_muscle4"]))
    var_list.append(list(df["EM_glove"]))
    # var_list.append(list(df["EM_slave"]))

    var_list = torch.tensor(var_list)
    # print(df.classe)    #取出测试数据的分类标签列
    return var_list


def testmain():
    """Driver file to run inference on test data."""
    className = ["A", "B", "C"]

    # params = sp().params

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--checkpoint",
        default='G:\Documents\My Postdoc Work\Intelligent Robot Navigation\DNN\Human-Activity-Recognition-master\{c}\model_200.pth'.format(c=params["Checkpoint"]),  # ,d = params["newcheckpoint"]
        type=str,
    )
    args = parser.parse_args()
    # assert type(args.checkpoint) is str, "G:\Documents\My Postdoc Work\Intelligent Robot Navigation\DNN\Human-Activity-Recognition-master\Checkpoints\model_275.pth"

    model = HARmodel(params["input_dim"], params["num_classes"])

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}'".format(args.checkpoint))

    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        return

    dataset = HARdataset(params["root"])
    # print(dataset.length)
    mean, std = dataset.mean, dataset.std
    # print(mean.size(), std.size()) #不同样本各个特征的平均值和标准差

    # logger = visdom.Visdom()

    testset = build_testset(params)
    print(testset.size())
    testset = (testset - mean) / std
    # 减均值除以标准差，即标准分数，给出了一组数据中各数值的相对位置，
    # 没有改变一个数据在该组数据中的位置，也没有改变该组数据的分布形状，只是数据变为平均数为0，标准差为1.

    results = []
    for i in range(testset.size(1)):
        test_data = testset[:, i].view(1, 1, -1)
        # print(test_data.size()) # 1*1*56
        output = model(test_data)
        results.append(int(output.max(1)[1]))
    # print(results)
    results = [className[i] for i in results]
    print("Prediction results:")
    print(results)

    print(len(results))
    # with open(r'G:\Documents\My Postdoc Work\Intelligent Robot Navigation\DNN\Human-Activity-Recognition-master\data\test{}.csv'.format(params["a"])) as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     column = [row['classe'] for row in reader]
    df = pd.read_csv(params["test"], low_memory=False)
    column = list(df['classe'])
    # EM = list(df['fIAV_muscle4'])
    print(column)
    # print(EM)
    i = correctA = correctB = correctC = A = B = C = AB = AC = BA = BC = CA = CB = 0

    while i < len(column):
        if column[i] == 'A':
            A += 1
        if column[i] == 'B':
            B += 1
        if column[i] == 'C':
            C += 1

        if results[i] == column[i]:
            if column[i] == 'A':
                correctA += 1
            elif column[i] == 'B':
                correctB += 1
            else:
                correctC += 1
        else:
            if column[i] == 'A':
                if results[i] == 'B':
                    AB += 1
                if results[i] == 'C':
                    AC += 1
            elif column[i] == 'B':
                if results[i] == 'A':
                    BA += 1
                if results[i] == 'C':
                    BC += 1
            else:
                if results[i] == 'A':
                    CA += 1
                if results[i] == 'B':
                    CB += 1

        i += 1
    print(A, B, C, correctA, correctB, correctC, AB, AC, BA, BC, CA, CB)
    print('AA = %d' % correctA, 'AB = %d' % AB, 'AC = %d' % AC, 'BB = %d' % correctB, 'BA = %d' % BA, 'BC = %d' % BC,
          'CC = %d' % correctC, 'CA = %d' % CA, 'CB = %d' % CB)
    meanAccuraccy = (correctA / A + correctB / B + correctC / C) / 3
    print(meanAccuraccy)

    params["test_acc"] = meanAccuraccy


if __name__ == "__main__":
    a = 1
    params = sp().params
    f = open("./output/" + "record.txt", "w")

    for i in range(a, a + 10):
        params["root"] = "data/data_pig_train" + str(i) + ".csv"
        params["test"] = "data/data_pig_test" + str(i) + ".csv"
        params["Checkpoint"] = "Checkpoints" + str(i)
        params["a"] = i
        print(params["root"])
        main()
        test_sum = 0
        for _ in range(10):
            testmain()
            test_sum += params["test_acc"]
        test_acc = test_sum / 10
        params["test_acc"] = str(test_acc)
        print(params["test_acc"])

        f.writelines("train:" + params["val_acc"] + " " + "test:" + params["test_acc"])
        f.write('\n')
    f.close()
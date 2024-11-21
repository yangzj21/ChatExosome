#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2024/7/25
# __author__: 'Alex Lu'
import copy
import os
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime
from timm.utils import accuracy, AverageMeter
import logging


def train_predict(model, device, train_loader, valid_loader, test_loader, **kwargs):
    best_model_state_dict, evaluated_acc = train(model, device, train_loader, valid_loader, **kwargs)
    model.load_state_dict(best_model_state_dict)
    predicted_acc = predict(model, device, test_loader, **kwargs)

    model_save_path = kwargs.get("model_save_path", "./outputs/checked")
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
        model_name_prefix = kwargs.get("model_name_prefix", "SXT")

        model_file_name = f'{model_name_prefix}_{datetime.now().strftime("%Y%m%d%H%M%S")}_' \
                          f'{int(evaluated_acc * 1000)}_{int(predicted_acc * 1000)}.pth'
        model_file = os.path.join(model_save_path, model_file_name)
        logging.info(f"model_file:{model_file}")
        torch.save(best_model_state_dict, model_file)


def train(model, device, train_loader, valid_loader, **kwargs):
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    min_epochs = kwargs.get("min_epochs", 10)
    total_epochs = 50  # 定义一个足够大的训练轮次
    patience_count = 0  # 记录验证集准确率不再提高的轮次
    patience = 10  # 定义耐心值，即验证集准确率不再提高的轮次
    best_accuracy = 0.0
    best_loss = 0.0
    best_model_state_dict = None
    best_epoch = 0
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    time0 = datetime.now()
    model.to(device)
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        pred_list = []
        true_list = []
        for datas, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch"):
            datas, labels = datas.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(datas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * datas.size(0)
            _, predicted = torch.max(outputs, 1)
            pred_list += predicted.tolist()
            true_list += labels.tolist()

        all_accuracy = accuracy_score(true_list, pred_list)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss.append(epoch_loss)
        train_acc.append(all_accuracy)

        logging.info(f"Epoch:{epoch + 1}/{total_epochs}. Training Avg Loss: {epoch_loss:.6f}")

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        pred_list = []
        true_list = []
        with torch.no_grad():
            for datas, labels in tqdm(valid_loader, desc="Validating", unit="batch"):
                datas, labels = datas.to(device), labels.to(device)
                outputs = model(datas)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pred_list += predicted.tolist()
                true_list += labels.tolist()
        accuracy = correct / total
        test_acc.append(accuracy)
        epoch_loss = running_loss / len(valid_loader.dataset)
        test_loss.append(epoch_loss)
        # 如果当前模型性能更好，则保存模型
        if (accuracy > best_accuracy) or (accuracy == best_accuracy and best_loss > epoch_loss + 0.0002):
            best_accuracy = accuracy
            best_loss = epoch_loss
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
            patience_count = 0  # 重置耐心计数
        else:
            patience_count += 1

        logging.info(f'Validating Accuracy: {correct / total:.6f}. (={correct}/{total}). '
                     f'Validating patience_count: {patience_count}')
        # 如果耐心计数达到耐心值，则跳出训练
        if epoch > min_epochs:
            if (patience_count >= patience and best_loss < 0.2) or (epoch >= min(total_epochs // 2, 50)):
                logging.info("Early stopping!")
                break

    time1 = datetime.now()
    logging.info(f'In train, best_accuracy is {best_accuracy}, in epoch {best_epoch}, avg loss is {best_loss:.6f}. '
                f'time cost is {(time1 - time0).seconds} seconds.')
    torch.save({'train_loss':train_loss,'test_loss':test_loss,'train_acc':train_acc,'test_acc':test_acc}, f'result_{time0}')
    return best_model_state_dict, best_accuracy


def predict(model, device, test_loader, **kwargs):
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    pred_list = []
    true_list = []
    test_loss = []
    test_acc = []
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()

    correct = 0
    total = 0

    time1 = datetime.now()
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        for datas, labels in tqdm(test_loader, desc="Test", unit="batch"):
            datas, labels = datas.to(device), labels.to(device)
            outputs = model(datas)
            _, predicted = torch.max(outputs, 1)


            batch_size = datas.size(0)
            total += batch_size
            correct += (predicted == labels).sum().item()
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_meter.update(acc1.item(), n=batch_size)
            acc5_meter.update(acc5.item(), n=batch_size)

            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), n=batch_size)

            pred_list += predicted.tolist()
            true_list += labels.tolist()
            
    predicted_accuracy = correct / total

    time2 = datetime.now()
    logging.info(f'In predict, accuracy is : {predicted_accuracy:.6f}. ={correct}/{total}. loss is {loss_meter.avg:.6f}. '
                f'time cost is {(time2 - time1).microseconds} microseconds.')
    test_acc.append(accuracy)
    epoch_loss = running_loss / len(test_loader.dataset)
    test_loss.append(epoch_loss)
    # precision = precision_score(true_list, pred_list, average='weighted', zero_division=1)
    # recall = recall_score(true_list, pred_list, average='weighted', zero_division=1)
    # f1_score = 2 * (precision * recall) / (precision + recall)

    precision, recall, f1_score, _ = precision_recall_fscore_support(true_list, pred_list, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(true_list, pred_list)
    logging.info(f'precision: {precision:.3f}. recall:{recall:.3f}. F1_score:{f1_score:.3f}. confusion_matrix:{conf_matrix}'
                 f'acc1:{acc1_meter.avg/100:.3f}, acc5:{acc5_meter.avg/100:.3f}.')

    # cm = confusion_matrix(true_list, pred_list)
    # cr = classification_report(true_list, pred_list, digits=3, zero_division=0)

    return predicted_accuracy

'''
    predict_prob_and_label: return predict label & its prob
'''
def predict_prob_and_label(model, device, test_loader, **kwargs):
    model.to(device)

    pred_list = []
    pred_prob_list = []

    with torch.no_grad():
        model.eval()
        for datas, labels in tqdm(test_loader, desc="Test", unit="batch"):
            datas, labels = datas.to(device), labels.to(device)
            outputs = model(datas)
            pred_prod, predicted = torch.max(outputs, 1)

            pred_list += predicted.tolist()
            pred_prob_list += pred_prod.tolist()
            
    return pred_prob_list, pred_list

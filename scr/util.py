import numpy as np
from sklearn.preprocessing import scale,StandardScaler
import torch
import torch.utils.data as Data
import math
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

def gpu_available():
    use_gpu = torch.cuda.is_available()
    return use_gpu

def get_data(train_path,test_path,validate_path):
    # load data and label
    train_all = np.load(train_path)
    test_all = np.load(test_path)
    validate_all = np.load(validate_path)

    train_data = train_all[:,0:1250]
    train_label = train_all[:,1250]

    test_data = test_all[:,0:1250]
    test_label = test_all[:,1250]

    validate_data = validate_all[:,0:1250]
    validate_label = validate_all[:,1250]
    return [train_data,train_label,test_data,test_label,validate_data,validate_label]

def z_score(train_data,test_data,validate_data):
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    validate_data = scaler.transform(validate_data)
    return [train_data,test_data,validate_data]

def dataTypeTransfer(train_data,test_data,validate_data):
    train_data = torch.from_numpy(train_data)
    train_data = torch.unsqueeze(train_data, dim=1).type(torch.FloatTensor)

    test_data = torch.from_numpy(test_data)
    test_data = torch.unsqueeze(test_data, dim=1).type(torch.FloatTensor)

    validate_data = torch.from_numpy(validate_data)
    validate_data = torch.unsqueeze(validate_data,dim=1).type(torch.FloatTensor)
    return [train_data,test_data,validate_data]

class makeDataset(Data.Dataset):
    def __init__(self,train_data,train_label):
        self.x_data = train_data
        label = train_label
        self.len = train_data.shape[0]
        self.y_data = torch.from_numpy(label).type(torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def train(Epoch,model,criterion,optimizer,
            train_loader,validate_loader,device,scheduler=True):
    if scheduler == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                       verbose=True, patience=10)
    running_loss = 0.0
    model.train()
    optimizer.zero_grad()
    train_acc = []
    for epoch in range(Epoch):
        print(1)
        for batch_idx, data in enumerate(train_loader):
            inputs, target = data
            if device == True:
                inputs = inputs.cuda()
                target = target.cuda()
            output = model(inputs)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, Epoch, running_loss))
                correct = 0
                total = 0
                validate_loss = 0.0
                with torch.no_grad():
                    for data in train_loader:
                        inputs, target = data
                        if device == True:
                            inputs = inputs.cuda()
                            target = target.cuda()
                        output = model(inputs)
                        _,predicted = torch.max(output.data, dim=1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                acc = 100*correct/total
                train_acc.append(acc)
                print('Accuracy on test set: %d %% [%d  /  %d]' % (acc, correct, total))

                running_loss = 0.0
                correct = 0
                total = 0
                validate_loss = 0.0
                with torch.no_grad():
                    for data in validate_loader:
                        inputs, target = data
                        if device == True:
                            inputs = inputs.cuda()
                            target = target.cuda()
                        output = model(inputs)
                        loss = criterion(output, target)
                        _,predicted = torch.max(output.data, dim=1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        validate_loss += loss.item()
                acc = 100*correct/total
                print(('Accuracy on validate set: %d %% [%d  /  %d]' % (acc, correct, total)))
                print('[%d, %5d] Val loss: %.3f' % (epoch + 1, Epoch, validate_loss))
                if scheduler == True:
                    scheduler.step(validate_loss)
    # np.save('/content/drive/My Drive/'+'EDB_Trans4', train_acc)
    return model
def evaluation(model,test_loader):
    model.eval()
    target_pred=[]
    target_true=[]
    y_score_roc = np.empty(shape=[0, 5])
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            output = model(inputs)
   
            _, predicted = torch.max(output, dim=1)
            # output_softmax = F.softmax(output, dim=1).cpu()
            target_pred += predicted.data.tolist()
            target_true += target.data.tolist()
            y_score_roc = np.concatenate((y_score_roc, output.cpu()), axis=0)
    return target_pred, target_true, y_score_roc

def get_results(target_true,target_pred):
    Acc = accuracy_score(target_true, target_pred)
    report = classification_report(target_true, target_pred,digits=5)
    # confusion matrix
    Conf_Mat = confusion_matrix(target_true, target_pred)  
    Acc_N = Conf_Mat[0][0] / np.sum(Conf_Mat[0])
    Acc_S = Conf_Mat[1][1] / np.sum(Conf_Mat[1])
    Acc_V = Conf_Mat[2][2] / np.sum(Conf_Mat[2])
    Acc_F = Conf_Mat[3][3] / np.sum(Conf_Mat[3])
    Acc_Q = Conf_Mat[4][4] / np.sum(Conf_Mat[4])
    TN = Conf_Mat[0][0]
    FN = Conf_Mat[1][0]
    TP = Conf_Mat[1][1]
    FP = Conf_Mat[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # F!:
    F1 = 2*((TPR*PPV)/(TPR+PPV))


    print('PRINT RESULTS REPORT')
    print('--------------------------------------')
    print('--------------------------------------')
    print('Confusion Matrix:')
    print('True Positive = %.4f'  % (TP))
    print('True Negative = %.4f' % (TN))
    print('False Positive =%.4f' % (FP))
    print('False Negative =%.2f' %  (FN))

    print('--------------------------------------')
    print('ACCURACY:')
    print('\nAccuracy=%.2f%%' % (Acc * 100))
    print('Accuracy_N=%.2f%%' % (Acc_N * 100))
    print('Accuracy_S=%.2f%%' % (Acc_S * 100))
    print('Accuracy_V=%.2f%%' % (Acc_V * 100))
    print('Accuracy_F=%.2f%%' % (Acc_F * 100))
    print('Accuracy_Q=%.2f%%' % (Acc_Q * 100))

    print('--------------------------------------')
    print('Other Evaluation Criteria:')
    print('Recall = %.4f'  % (TPR))
    print('Precision = %.4f' % (PPV))
    print('Specification =%.4f' % (TNR))
    print('F1 =%.4f' %  (F1))
    print('--------------------------------------')
    print('REPORT:')
    print(report)
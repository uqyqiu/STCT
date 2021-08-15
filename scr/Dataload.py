from scr.util import get_data, z_score,dataTypeTransfer,\
    makeDataset
import torch.utils.data as Data

def makeDataloader(train_data,train_label,batch_size=128, num_workers=0):
    dataset = makeDataset(train_data,train_label)
    return Data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

def DataLoad(train_path, test_path, validate_path):
    data_label = get_data(train_path, test_path, validate_path)
    data_standard = z_score(data_label[0],data_label[2],data_label[4])
    data_torch  = dataTypeTransfer(data_standard[0], data_standard[1], data_standard[2])
    # Dataset = makeDataset()
    train_loader = makeDataloader(data_torch[0], data_label[1],num_workers=2)
    test_loader = makeDataloader(data_torch[1], data_label[3])
    validate_loader = makeDataloader(data_torch[2], data_label[5])
    return [train_loader,test_loader,validate_loader]

import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from timm.utils import AverageMeter
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
import model_func
from model_func import *


# loading the data
class GetData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def getData(self):
        self.data = pd.read_csv(self.data_path)
        self.data = self.data[["open", "close", "high", "low", "adj close", 'Volume']]
        self.close_min = self.data['close'].min()
        self.close_max = self.data["close"].max()
        self.data = self.data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
        return self.data

    # n is the number of date period, 30 in here
    def process_data(self, n):
        if self.data is None:
            self.getData()
        feature = [
            self.data.iloc[i: i + n].values.tolist()
            for i in range(len(self.data) - n + 2)
            if i + n < len(self.data) - 2
        ]
        label = [
            self.data.close.values[i + n + 2]
            for i in range(len(self.data) - n + 2)
            if i + n < len(self.data) - 2
        ]

        # separate train and test
        train_x = feature[:-500]
        test_x = feature[-500:]
        train_y = label[:-500]
        test_y = label[-500:]

        return train_x, test_x, train_y, test_y


# training model combining train and test
def train_model(model, criterion, device, optimizer, train_loader,
                test_dataLoader, ema,use_ema, last_pccs, model_save, GD, epoch):

    # initial parameter to save result
    loss_meter_train = AverageMeter()
    loss_meter_test = AverageMeter()

    # training
    for batch_idx, (train, target) in enumerate(train_loader):
        model.train()
        train, target = train.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(train)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if use_ema:
            ema.update()
        torch.cuda.synchronize()
        loss_meter_train.update(loss.item(), target.size(0))
    ave_train_loss = loss_meter_train.avg

    # test
    real_list = []
    pred_list = []
    with torch.no_grad():
        if use_ema:
            ema.apply_shadow()
        for test, real in test_dataLoader:
            model.eval()
            test, real = test.to(device, non_blocking=True), real.to(device, non_blocking=True)
            test_output = model(test)
            loss = criterion(test_output, real)
            loss_meter_test.update(loss.item(), real.size(0))

            test_output, real = test_output.to('cpu'), real.to('cpu')
            test_output = test_output.numpy()
            for i in test_output:
                pred_list.append(i)
            real = real.numpy()
            for i in real:
                real_list.append(i)

        if use_ema:
            ema.restore()
        ave_test_loss = loss_meter_test.avg

        # r square section
        pccs = r2_score(pred_list, real_list)
        pccs = round(pccs, 3)

        # result save section
        if pccs > last_pccs and pccs > 0.5:
            model_name = str(pccs)+'.pth'
            model_path = os.path.join(model_save, model_name)
            torch.save(model.state_dict(), model_path)

            pred = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in pred_list]
            real = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in real_list]

            plot_img(real, pred, model_save, pccs)
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    lr = round(lr, 6)

    # show
    print('Epoch: {}| Train Loss: {}| Test Loss: {}| LR: {}'.format(epoch,
                                                                    round(ave_train_loss, 8),
                                                                    round(ave_test_loss, 8),
                                                                    lr))

    return ave_train_loss, ave_test_loss, pccs


# make graph and save
def plot_img(real, pred, fig_save, pccs):
    # plt.figure(figsize=(18, 9))
    plt.plot(range(len(pred)), pred, color='r')
    plt.plot(range(len(real)), real, color='b')
    plt.plot(pred, 'r-', label=u'Predicted data')
    # save fig
    plt.plot(real, 'b-', label=u'Real data')
    plt.legend(["Predict", "Real"], loc="upper right")
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close', fontsize=18)
    title = 'R square value ' + str(pccs)
    plt.title(title)
    plt.savefig(fig_save + "/" + str(pccs) + ".png")
    plt.cla()


if __name__ == '__main__':
    # model parameter
    model_lr = 0.01
    csv_dir = 'Data/BHP.AX.csv'
    stock_name = 'BHP'
    INPUT_SIZE = 6
    BATCH_SIZE = 32
    EPOCHS = 150
    days_num = 30
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # selecting model
    Model = 'CNN_GRU'
    if Model == 'LSTM':
        model = model_func.LSTM(n=INPUT_SIZE)
    elif Model == 'GRU':
        model = model_func.GRU(n=INPUT_SIZE)
    elif Model == 'CNN_GRU':
        model = model_func.CNNGRUModel(n=INPUT_SIZE)

    model.to(DEVICE)

    use_ema = False
    Best_PCCS = 0

    total_train_loss = []
    total_test_loss = []

    # make folder with running time
    localtime = time.asctime(time.localtime(time.time())).split()
    str_time = str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3][0:2])
    # checkpoint folder
    file_dir = 'checkpoints/' + Model + stock_name + str(str_time)

    # ema separate the save folder
    if use_ema:
        file_dir = file_dir + '_' + 'ema'

    # final folder for save checkpoint
    if os.path.exists(file_dir):
        print('true')
        num = len(os.listdir('checkpoints/'))
        file_dir = file_dir + '_' + str(num)
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)

    writer = SummaryWriter(str(file_dir) + '/logs')

    # data processing and loading
    GD = GetData(data_path=csv_dir)
    x_train, x_test, y_train, y_test = GD.process_data(days_num)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_data = TensorDataset(x_train, y_train)
    train_dataLoader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_data = TensorDataset(x_test, y_test)
    test_dataLoader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # loss function
    criterion = nn.MSELoss()
    criterion.to(DEVICE)

    # optimizer
    optimizer = sgd_optimizer(model, model_lr, 0.9, 1e-4)

    # learning rate cosine annealing
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-3)

    # EMA establishment
    ema = model_func.EMA(model, 0.999)
    ema.register()

    last_pccs = 0
    is_set_lr = False

    for epoch in range(EPOCHS):
        train_loss, test_loss, pccs = train_model(model=model,
                                                  criterion=criterion,
                                                  device=DEVICE,
                                                  optimizer=optimizer,
                                                  train_loader=train_dataLoader,
                                                  test_dataLoader=test_dataLoader,
                                                  ema=ema,
                                                  use_ema=use_ema,
                                                  last_pccs=last_pccs,
                                                  model_save=file_dir,
                                                  GD=GD,
                                                  epoch=epoch)
        last_pccs = pccs
        total_train_loss.append(train_loss)
        total_test_loss.append(test_loss)
        # Change learning rate at last 50 epochs
        if epoch < EPOCHS-50:
            cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-3
                    is_set_lr = True

        # tensorboard
        writer.add_scalar(tag='Train loss', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag='Test loss', scalar_value=test_loss, global_step=epoch)
        writer.add_scalar(tag='R-square value', scalar_value=pccs, global_step=epoch)

    # save the loss for train and test, but it is not vary clear, backup plan for tensorboard failure
    plt.plot(range(len(total_train_loss)), total_train_loss, color='r')
    plt.plot(range(len(total_test_loss)), total_test_loss, color='b')
    plt.plot(total_train_loss, 'r-', label=u'Train loss')
    # save fig
    plt.plot(total_test_loss, 'b-', label=u'Test loss')
    plt.legend(["Train", "Test"], loc="upper right")
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    title = Model + ' loss'
    plt.title(title)
    plt.savefig(file_dir + "/" + str(title) + ".png")
    plt.cla()

writer.close()
import snnflow
import torch
from tqdm import tqdm
import torch.nn.functional as F
from snnflow.IO.Dataset import MNIST as dataset
from ANN_particle_fillter import ExampleDataset


# 创建训练数据集
root = './snnflow/Datasets/MNIST'
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)

# 设置运行时间与批处理规模
run_time = 50
bat_size = 100

# 创建DataLoader迭代器
dataset_1 = ExampleDataset()
train_loader = snnflow.Dataloader(train_set, batch_size=bat_size, shuffle=True, drop_last=False)
test_loader = snnflow.Dataloader(test_set, batch_size=bat_size, shuffle=False)


class TestNet(snnflow.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # coding
        self.input = snnflow.Encoder(num=6, coding_method='poisson')

        # neuron group
        self.layer1 = snnflow.NeuronGroup(64,  neuron_model='clif')
        self.layer2 = snnflow.NeuronGroup(128, neuron_model='clif')
        self.layer3 = snnflow.NeuronGroup(64,  neuron_model='clif')

        # decoding
        self.output = snnflow.Decoder(num=1, dec_target=self.layer3, coding_method='spike_counts')

        # Connection
        self.connection1 = snnflow.Connection(self.input,  self.layer1, link_type='full')
        self.connection2 = snnflow.Connection(self.layer1, self.layer2, link_type='full')
        self.connection3 = snnflow.Connection(self.layer2, self.layer3, link_type='full')

        # Minitor
        self.mon_V = snnflow.StateMonitor(self.layer1, 'V')
        self.mon_O = snnflow.StateMonitor(self.layer1, 'O')

        # Learner
        self.learner = snnflow.Learner(trainable=self, algorithm='STCA')
        self.learner.set_optimizer('Adam', 0.001)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        simulator = snnflow.Torch_Backend(device)
        simulator.dt = 0.1

        self.set_simulator(simulator)

# 网络实例化
Net = TestNet()

eval_losses = []
eval_acces = []
losses = []
acces = []
num_correct = 0
num_sample = 0
device = 'cuda'

for epoch in range(100):

    # 训练阶段
    print("Start training")
    train_loss = 0
    train_acc = 0
    pbar = tqdm(total=len(train_loader))
    for i, item in enumerate(train_loader):
        # 前向传播
        data, label = item
        Net.input(data)
        Net.output(label)
        Net.run(run_time)
        output = Net.output.predict
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        label = torch.tensor(label, device=device)
        batch_loss = F.cross_entropy(output, label)

        # 反向传播
        Net.learner.optim_zero_grad()
        batch_loss.backward(retain_graph=False)
        Net.learner.optim_step()

        # 记录误差
        train_loss += batch_loss.item()
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label).sum().item()  # 记录标签正确的个数
        acc = num_correct / data.shape[0]
        train_acc += acc

        pbar.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
        pbar.update()
    pbar.close()
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

    # 测试阶段
    eval_loss = 0
    eval_acc = 0
    print("Start testing")
    pbarTest = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            data, label = item
            Net.input(data)
            Net.run(run_time)
            output = Net.output.predict
            output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
            label = torch.tensor(label, device=device)
            batch_loss = F.cross_entropy(output, label)
            eval_loss += batch_loss.item()

            _, pred = output.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / data.shape[0]
            eval_acc += acc
            pbarTest.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
            pbarTest.update()
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
    pbarTest.close()
    print('epoch:{},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch,eval_loss / len(test_loader), eval_acc / len(test_loader)))
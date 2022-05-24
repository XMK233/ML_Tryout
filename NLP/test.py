import numpy as np      # tang.npz的压缩格式处理
import os       # 打开文件
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchnet import meter
import tqdm

def get_data():
    if os.path.exists(data_path):
        datas = np.load(data_path, allow_pickle=True)      # 加载数据
        data = datas['data']  # numpy.ndarray
        word2ix = datas['word2ix'].item()   # dic
        ix2word = datas['ix2word'].item()  # dic
        return data, word2ix, ix2word

data_path = '../originalDataset/tang.npz'
data, word2ix, ix2word = get_data()
data = torch.from_numpy(data)
dataloader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True, num_workers=1)  # shuffle=True随机打乱

## 看看总共循环多少次: 5758

def print_origin_poem(i):
    '''
    data里面每一行就是一首诗，总共57580首。
    本函数可以打印出data[i]对应的诗。
    '''
    poem_txt = [ix2word[int(i)] for i in data[i]]
    print("".join(poem_txt[poem_txt.index("<START>") + 1:-1]))
    print()
    print("".join(poem_txt[:]))

print_origin_poem(45423)

def generate(model, start_words, ix2word, word2ix):     # 给定几个词，根据这几个词生成一首完整的诗歌
    txt = []
    for word in start_words: # 机器学习
        txt.append(word) ## txt: [机,器,学,习]
    input = Variable(torch.Tensor([word2ix['<START>']]).view(1,1).long())      # tensor([8291.]) → tensor([[8291.]]) → tensor([[8291]])
    hidden = None
    num = len(txt)
    # print(input)
    for i in range(48):      # 最大生成长度
        output, hidden = model(input, hidden)
        if i < num:
            w = txt[i]
            input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0]
            w = ix2word[top_index.item()]
            txt.append(w)
            input = Variable(input.data.new([top_index])).view(1, 1)
        if w == '<EOP>':
            break
    return ''.join(txt)

def gen_acrostic(model, start_words, ix2word, word2ix):
    result = []
    txt = []
    for word in start_words:
        txt.append(word)
    input = Variable(
        torch.Tensor([word2ix['<START>']]).view(1, 1).long())  # tensor([8291.]) → tensor([[8291.]]) → tensor([[8291]])
    input = input.cuda()
    hidden = None

    num = len(txt)
    index = 0
    pre_word = '<START>'
    for i in range(48):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0]
        w = ix2word[top_index.item()]

        if (pre_word in {'。', '!', '<START>'}):
            if index == num:
                break
            else:
                w = txt[index]
                index += 1
                input = Variable(input.data.new([word2ix[w]])).view(1,1)
        else:
            input = Variable(input.data.new([word2ix[w]])).view(1,1)
        result.append(w)
        pre_word = w
    return ''.join(result)


class Trsfmr_decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, nheads=8, nlayers=6):
        super(Trsfmr_decoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                       nhead=nheads)  # , dim_feedforward=hidden_dim
        self.transformer_encoder = nn.TransformerEncoder(self.encode_layer, num_layers=nlayers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nheads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=nlayers)

        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        embeds_ = self.embeddings(input)
        embeds = self.embeddings(input)
        memory = self.transformer_encoder(embeds_)
        output = self.transformer_decoder(embeds, memory)
        # print(output.shape)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, None


def train():
    modle = Trsfmr_decoder(len(word2ix), 128, 256)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modle.parameters(), lr=1e-3)  # 学习率1e-3
    loss_meter = meter.AverageValueMeter()

    period = []
    loss2 = []
    for epoch in range(8):  # 最大迭代次数为8
        loss_meter.reset()
        for i, data in tqdm.tqdm(enumerate(dataloader)):  # data: torch.Size([128, 125]), dtype=torch.int32
            print(data)
            data = data.long().transpose(0, 1).contiguous()  # long为默认tensor类型，并转置, [125, 10]
            input, target = Variable(data[:-1, :]), Variable(data[1:, :])
            optimizer.zero_grad()
            output, _ = modle(input)
            loss = criterion(output, target.view(-1))  # torch.Size([15872, 8293]), torch.Size([15872])
            loss.backward()
            optimizer.step()

            loss_meter.add(
                loss.item())  # loss:tensor(3.3510, device='cuda:0', grad_fn=<NllLossBackward>)loss.data:tensor(3.0183, device='cuda:0')

            period.append(i + epoch * len(dataloader))
            loss2.append(loss_meter.value()[0])

            # print(data)
            if (1 + i) % 2 == 0:  # 每575个batch可视化一次
                print(str(i) + ':' + generate(modle, '床前明月光', ix2word, word2ix))

            if i == 5: ## 只跑5个batch
                break

        torch.save(modle.state_dict(), 'model_poet_2.pth')

        break  ## 我们暂时只跑1个epoch

    # plt.plot(period, loss2)
    # plt.show()

if __name__ == '__main__':
    data_path = '../originalDataset/tang.npz'
    data, word2ix, ix2word = get_data()
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True, num_workers=1)  # shuffle=True随机打乱
    train()
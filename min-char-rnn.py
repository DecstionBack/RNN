"""
From: https://gist.github.com/karpathy/d4dee566867f8291f086
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
#读入数据，应该是纯文本格式的
data = open('input.txt', 'r').read() # should be simple plain text file

#data是一系列的字符串，使用set(data)会按照字符级别将其划分成不同的字符，并且加入结合中（去重，建立词汇表），然后使用list将其组成一个有顺序的词典
#data是str，set(data)是字符级别的集合，不会很大
chars = list(set(data))

#data_size 数据量  vocab_size:词汇表大小，也就是词典长度
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))

#将字符按照在list中的顺序变成词典，key值是字符，value值是在list中的位置，char_to_ix的作用是根据字符可以查找在list中的位置
#enumerate(chars)为(0,'a'),(1,'b')的形式
char_to_ix = { ch:i for i,ch in enumerate(chars) }
#与char_to_ix正好相反，根据位置来查找对应的字符，这里使用dict来查找会比使用list查找的效率高
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#网络结构的参数，hidden_size代表隐藏层的神经元个数H，seq_length代表时间步数，这里设置为25，代表一次传入25个字符， learning_rate是SGD的步长
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# Ht = tanh(Wxh * x + Whh * Ht-1 + bh)
# y = Why * Ht + by
#开始时都是初始化为服从正态分布的小数
#程序中使用的字符向量表示是one-hot表示方法，所以维度也就是vocab_size，所以Wxh的维度为hidden_size * vocab_size
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden 后面乘以0.01和除以sqrt(10000)是一样的，其实应该是除以sqrt(n)
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

#训练过程
def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers. 在词向量矩阵中根据标号就可以取出来相应的词向量
  hprev is Hx1 array of initial hidden state 对于每一个样本是H*1的，对于minibatch而言应该是N*H的矩阵   x:N*D  Wxh:D*H Whh:H*H Why:H*V b:H*1 h:N*h
  returns the loss, gradients on model parameters, and last hidden state  返回HT
  """

  #xs:字符的one-hot向量表示
  xs, hs, ys, ps = {}, {}, {}, {}
  #为什么hs的最后一个变量是hprev？当t=0时需要hs[t-1]这时候需要最后一个元素有值（并且是h的初始值），所以这里将最后一个设为hprev, 并且hs[-1]经过一轮计算后也被新的值覆盖
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  #inputs是长度为seq_length的输入串，是字符的下标列表，（这里没有mini-batch,一次只是传入一个字符串）
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation  one-hot表示方法，所以是V*1
    xs[t][inputs[t]] = 1  #对应的字符位置为1，其余为0
    #hs是隐藏状态H，每一个时间步更新一行，H为T*H矩阵，hs[t]得到的是一个向量
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    #ys是输出结果，这里是一个T*V的矩阵， ys[t]是一个向量
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    #ps是对得到的ys进行softmax得到的结果
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    #计算每一个的损失Li，根据交叉熵可以看出来下面这样写比较简单
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

  #上面前向传播一轮已经完成，后面开始计算反向
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  #dhnext是一个中间变量一直往前传播的，所以这里是向量
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    #每一个都是Lt对参数求导
    #下面两句话等价于dy = softmax(y) - y，求导过程推导一下就出来了
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here

    #下面的推导一下公式就出来了
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)

  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients 将梯度限制在-5到5之间
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1] #最后一个hs[len(inputs)-1]就是前面要返回的对最后的h求导结果

#测试过程
def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  #seed_ix是inputs[0]，即输入第一个字符后面sample出来一系列字符
  x[seed_ix] = 1
  #ixes代表选取的结果，结果是一个n*V的矩阵，其中每一个向量是一个one-hot向量代表选取的字符
  ixes = []
  #n代表选取多少个字符
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    #p.ravel将p展成1-D向量，p=ndarray代表词汇表中每个词语选中的概率，np.random.choice代表根据概率p随机选择，如果不指定概率p则是按照均匀分布来选择
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

#p是指针位置，n是迭代次数
n, p = 0, 0
#这儿m开头的几个变量是做什么的？只是在更新的时候起作用，觉得这儿甚至可以省略，直接操作后面zip()中的变量即可
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0     这儿的smooth_loss应该结合下面的loss更新规则一起看，应该是与Adagrad相关的
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0:

    #numpy中写列向量需要这样写，如果写成np.zeors(hidden_size)，则默认是一行
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data

  #data是文本展成字符串得到的，inputs是从p开始去取出来seq_length长度的字符作为输入得到的下标列表
  #targets是从p+1开始的长度为seq_length-1的字符下标，这里是作为inputs的答案，每一个字符下一个应该输出的答案是下一个字符
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  #每迭代100轮就观察一下RNN训练得到的结果怎么样
  if n % 100 == 0:
    #sample时需要将上一状态的hprev带着，但是不会对隐藏状态h进行修改
    sample_ix = sample(hprev, inputs[0], 200)

    #将sample得到的下标转换成具体的字符输出
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print ('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  #zip起到的作用是param从第一个list中选，dparam从第二个list中选，mem从第三个中选，m开头的几个变量其实完全可以不用的
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 

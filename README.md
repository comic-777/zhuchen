# 第二周

本周主要学习了cycle gan的相关知识并在自己的电脑上运行了cycle gan的代码。



## 学习

该模型效果

![xiaoguo](.\picture\xiaoguo.jpg)

这个模型可以在缺少配对数据集的情况下，完成不同域之间的转换

不需要准备相配对的图片，即在数据集不需要这样的配对图片

![image-20230119171902733](.\picture\image-20230119171902733.png)



### 网络架构

输入为任意两组数据，不需要相关联。

需要两对G、D网络 从真实图像——（生成器G）——生成图像——（逆生成器F）——还原图像，对比真实图像与还原图像

![image-20230119202150222](.\picture\image-20230119202150222.png)

**通过cycle-consistency loss ，循环一致性损失**

**循环一致性损失函数 核心**

1. **使得transfer进去的图像仍保留原始图像的信息，防止“思密达”现象**
2. **间接实现了pix2pix的paired image translation功能**
3. **防止模式崩溃，总生成相同图像**

**来避免生成类似X域但完全不属于X域的图**

**来自X域的图像必须保证在经过G生成器后得到的Y~可以通过DY判别器的检验，同时Y~必须在经过F生成器后可以回到X域**

**同理，来自Y域的图像也必须有相同的保证**



把生成图和原始图算L1范数，逐元素作差取绝对值期望求和

<img src=".\picture\image-20230119211827265.png" alt="image-20230119211827265" style="zoom:67%;" />



整体损失函数由两个生成器判别器损失函数加上循环一致性损失函数组成，λ用来调节损失强度

![image-20230119211936298](.\picture\image-20230119211936298.png)

缺点和讨论

不擅长形状，三维信息，没有理解高级语言，不理解先验知识

## 代码运行

### 下载代码

在github项目地址（https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix）获取cyclegan的代码，

<img src=".\picture\image-20230119155528986.png" alt="image-20230119155528986" style="zoom: 33%;" />

不知道为什么这个项目不能直接下载zip压缩包，我选择Open with Visual Studio ，在VS里下载了项目代码，移动文件夹到相应位置，也可以达到相同的效果。



### 创建代码运行环境

利用anaconda创建新环境，该项目的环境要求在requirements的文本文件中，如图所示

```
torch>=1.4.0
torchvision>=0.5.0
dominate>=2.4.0
visdom>=0.1.8.8
```

也可以利用`conda env create -f environment.yml`. 命令创建新环境，完成后，在anaconda环境文件夹（envs）内可以看到该环境已经创建。

### 使用Pycharm打开项目

选择新创建的环境作为该项目的python解释器，在其中可以看到当前使用的各个库的版本，可以看到下载的版本均符合requirements里的环境要求。

<img src=".\picture\image-20230119155948541.png" alt="image-20230119155948541" style="zoom:67%;" />

利用`conda env create -f environment.yml`. 命令创建的环境pytorch版本为1.8.1，在使用GPU加速计算时会出现会出现capability sm_86 is not compatible的问题，同时根据输出可以看到 The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75当前pytorch只能支持上面几种架构。这样会导致计算中无法调用GPU并且计算会出现问题，导致不仅计算速度慢，最后还无法得到正确结果，出来的模型完全用不了

##### **这个是我使用pix2pix模型训练出来的结果，可以看到fake_B一片黑，没有任何效果。**

<img src=".\picture\image-20230119162113405.png" alt="image-20230119162113405" style="zoom:33%;" />



##### 解决方法

升级pytorch版本，在终端中输入`pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`下载适配CUDA的版本后，再运行代码就可以正常运行。

##### 注意

1.不能使用最新版本的pytorch加CUDA，不适配

2.原因是pytorch版本过低不支持新架构的GPU，因此更改GPU驱动时没有用的（我在一开始下载了数个版本的驱动加上cudnn，一直没有解决这个问题，最后更改pytorch版本解决问题）

<img src=".\picture\image-20230119163409484.png" alt="image-20230119163409484" style="zoom: 67%;" />



### 准备数据集

项目提供的下载数据集方法`bash ./datasets/download_cyclegan_dataset.sh maps`

在文件夹中打开该文件，可以得到url地址，打开后可以自己选择想要下载的数据集

<img src=".\picture\image-20230119164808351.png" alt="image-20230119164808351" style="zoom:50%;" />

下载完数据集，将其解压放到项目文件夹中的datasets下



### 训练模型

运行`python train.py --dataroot ./datasets/apple2orange --name apple2orange_cyclegan --model cycle_gan`即可正常开始训练。

### 可视化训练过程

使用visdom可视化训练过程。进入到库所在的文件夹中，打开文件server.py文件，将`download_scripts()`注释，防止每次打开**visdom**时都自动更新，导致无法使用。

<img src=".\picture\image-20230119173619509.png" alt="image-20230119173619509" style="zoom:50%;" />


# 实时声音克隆

该仓库是[从说话人确认到多说话人文本语音合成的迁移学习](https://arxiv.org/pdf/1806.04558.pdf))的实现，带有实时工作的语音编码器。这是我的[硕士论文](https://matheo.uliege.be/handle/2268.2/6801)。

SV2TTS是一个深度学习框架，分三个阶段。在第一阶段，人们从几秒钟的音频中创建一个声音的数字表示。在第二和第三阶段，该表示被用作参考，以生成给定任意文本的语音。

**视频演示**(点击图片):

[![Toolbox demo](https://i.imgur.com/8lFUlgz.png)](https://www.youtube.com/watch?v=-O_hYhToKoA)

### **实现的论文**  

|                          链接                          | 特定的工具           | 标题                                               | 源代码                                                  |
| :----------------------------------------------------: | -------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| [**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS**           | **将学习从说话人验证转移到多说话人文本到语音合成** | 本仓库                                                  |
|   [1802.08435](https://arxiv.org/pdf/1802.08435.pdf)   | WaveRNN (语音编码器) | 高效的神经音频合成                                 | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|   [1703.10135](https://arxiv.org/pdf/1703.10135.pdf)   | Tacotron(合成器)     | Tacotron:走向端到端语音合成                        | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|   [1710.10467](https://arxiv.org/pdf/1710.10467.pdf)   | GE2E(编码器)         | 说话人确认的广义端到端损失                         | 本仓库                                                  |

## 注意

像深度学习中的其他事情一样，这个项目很快就过时了。许多其他开源库或SaaS应用程序(通常是付费的)会给你比这个库更好的音频质量。如果你关心你要克隆的声音的保真度及其表现力，这里有一些替代声音克隆解决方案的个人建议:

+ 请查看 [CoquiTTS](https://github.com/coqui-ai/tts) 它是一个更先进的开源存储库，具有更好的语音克隆质量和更多功能。
+ 查看 [paperswithcode](https://paperswithcode.com/task/speech-synthesis/) 了解语音合成领域的其他存储库和最新研究。
+ 查看 [Resemble.ai](https://www.resemble.ai/) (免责声明:我在那里工作)了解最先进的语音克隆技术。



## 安装

#### 1.安装依赖

1. Windows和Linux都支持。为了训练和推理速度，建议使用GPU，但这不是强制性的。

2. 推荐Python 3.7。Python 3.5或更高版本应该可以工作，但是您可能必须调整依赖项的版本。我建议使用“venv”建立一个虚拟环境，但这是可选的。

3. 安装 [ffmpeg](https://ffmpeg.org/download.html#get-packages)。这是阅读音频文件所必需的。

4. 安装[PyTorch](https://pytorch.org/get-started/locally/)。选择最新的稳定版本，你的操作系统，你的软件包管理器(默认为pip ),如果你有GPU，最后选择任何一个推荐的CUDA版本，否则选择CPU。运行给定的命令。

5. 用```pip install -r requirements.txt```安装其余的依赖

   :exclamation:对于windows,安装[Microsoft Visual C++ 14.0](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)(勾选前3个即可).

### 2.(可选)下载预训练模型

预训练模型现在可以自动下载。如果这对你不起作用，你可以手动下载它们[here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models)。

### 3. (可选)测试配置

在下载任何数据集之前，您可以先使用以下工具测试您的配置:

```python demo_cli.py```

如果所有测试都通过了，你就可以运行了。

### 4. (可选) 下载数据集

对于单独使用工具箱，我只推荐下载 [`LibriSpeech/train-clean-100`](https://www.openslr.org/resources/12/train-clean-100.tar.gz).将内容提取为`<datasets_root>/LibriSpeech/train-clean-100`,其中`<datasets_root>`是您选择的目录。工具箱中支持其他数据集，查看[这里](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets)。你可以不下载任何数据集，但是你需要你自己的数据作为音频文件，或者你必须用工具箱来记录它。

### 5. 启用工具箱

您可以尝试使用工具箱:

`python demo_toolbox.py -d <datasets_root>`  
或者  
`python demo_toolbox.py`  

这取决于您是否下载了任何数据集。

如果您运行的是X-server，或者出现`Aborted(core dumped)`错误，请参见[此问题](https://github . com/coren tinj/Real-Time-Voice-Cloning/issues/11 # issue comment-504733590)。
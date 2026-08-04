任务：我们现在要利用GLM5.1完成立项任务验证，

首先要验证的一个任务就是：将GLM MoE的专家均匀的，无重复的放置在8张卡上，应该是就是expert parallelism=8, 然后在跑Livecodebench的一条数据的时候，模拟一个专家从NPU HBM (64GB *8的NPU显存) offload到DDRAM（1.5T的CPU内存），以及DDRAM offload到 SSD(/data目录实际存放的介质，7T的存储空间介质)，分别的时延以及各类测算延迟的相关指标；以及这个expert从SSD 加载至DDRAM的时延，DDRAM加载至NPU的时延，以及SSD一次性直接加载至NPU HBM的时延；然后只用一条livecodebench实验的目的只是为了模拟这一块的过程；
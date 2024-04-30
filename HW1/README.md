## 1.requirment.txt
pip install numpy, pandas, matplotlib
## 2.执行方式
**参数设置**：在main.py函数中，可以执行单步的函数调用，即通过在命令行中执行"python main.py --data_path ... --dataset ... ..."这样的方式来获得结果。设置默认参数为ratio_train=0.7、ratio_val=0.15、ratio_test=0.15，同时为进行BoxCox变化，增加 ’--alpha‘ 参数，并设置默认参数大小为0.5。其余保持不变。同时，可以在main.py函数中同时执行data_visualize函数以得到相应的图，增加 ’--t‘ 参数，并设置默认参数大小为10，在执行data_visualize时将该参数传入。

**执行方式**：res.txt保存执行获得的结果，run.py中进行多次执行，而无需每次执行都需要修改main.py的参数。在run.py中初始化三个list————data_dirs、model_dirs、transform_dirs来重复执行。每一个list参照main.py中的dict设置，此处不再赘述。直接执行"python run.py"即可得到对应的结果。
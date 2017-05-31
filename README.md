# MNIST linear regression

## Folder description
1. `ds`: MNIST data set
2. `exp_result`: training results; stores `.npy` files of weight *W* and *b*
3. `log`:  log files record loss in different epochs and iterations
4. `test_result`: testing results; stores *error rate* on test data set
5. `loss_plot`: plotting the loss for different numbe of layers

## Basic experiment settings for now
- total training data points: 60000
- total testing data points: 10000
- epoch: 10 or 1000
- minibatch size: 150 or 20
- precision: specified by `_g32w32`, `_g32w5` and `_g5w5` at the end of filename
- models: specified by `_2l` or nothing as for 2 layers or 1 layer
- experiment id: specified by the last number to represent duplicate exp with same hyperparameters

## Result
### Test 1:
#### Basic experiment settings for now
- total training data points: 60000
- total testing data points: 10000
- epoch: 10 or 1000
- minibatch size: 150 or 20
- precision: specified by `_g32w32`, `_g32w5` and `_g5w5` at the end of filename
- models: specified by `_2l` or nothing as for 2 layers or 1 layer
- experiment id: specified by the last number to represent duplicate exp with same hyperparameters

#### 1 layer:
- `_g5w5` precision: 8.04% error rate
- `_g32w5` precision: 8.24% error rate
- `_g32w32` precision: 7.94% error rate
<!-- <img src="./loss_plot/loss_1l.png" alt="loss plot 1 layer" style="width: 50px;"/> -->

#### 2 layers:
- `_g5w5` precision: 2.73% error rate
- `_g32w5` precision: 2.91% error rate
- `_g32w32` precision: 2.43% error rate
- `e1000mb20_g32w32` precision: 2.19% error rate
<!-- <img src="./loss_plot/loss_2l.png" alt="loss plot 2 layer" style="width: 50px;"/> -->
<!-- ![loss plot 2 layer](./loss_plot/loss_2l.png =100x) -->

#### Feedback Alignment(FA):
- `_g32w32_2l` precision: 2.75% error rate
<!-- <img src="./loss_plot/loss_fa.png" alt="loss plot fa" style="width: 50px;"/> -->
<!-- ![loss plot fa](./loss_plot/loss_fa.png =100x) -->

#### Random Gradients(RG):
1. Fixed Random Gradients with Sign Concordant:
- Vary coefficient scales of gradients of `W`:

|No.    |gradient scale | gW   | gb   | learning rate |accuracy|loss |
|:---:  |:-------------:|:---: |:---: |:---:          |:---:   |:---:|
|`_rg_1`|uniform [0,1)  |0.01  |1     |1e-3           |19.5%   |1.34 |
|`_rg_2`|               |1     |      |               |4.07%   |0.18 |
|`_rg_3`|               |0.1   |      |               |8.38%   |0.34 |

> learning rate decay by 10 at iteration of No.`2000`, `3000` and `3500`

- Without learning rate decay:

|No.    |gradient scale | gW   | gb   | learning rate |accuracy|loss |
|:---:  |:-------------:|:---: |:---: |:---:          |:---:   |:---:|
|`_rg_4`|uniform [0,1)  | 1    |1     | 1e-3          |3.35%   |0.14 |

<!-- <img src="./loss_plot/loss_rg.png" alt="loss plot rg" style="width: 50px;"/> -->

### Test 2:
#### Basic Experiment Setting:
- total training data points: 60000
- total testing data points: 10000
- training data points: 54000
- validation data points: 6000
- epoch: 100
- minibatch size: 120
- experiment id: specified by the last number to represent duplicate exp with same hyperparameters

#### Five comparable experiment settings:
1. SGD 
2. SGD with Momentum and Decay (SGD_md)
3. Batch Manhattan with Momentum and Decay (BM_md)
4. Batch Manhattan Scaled by a Random Matrix with Momentum and Decay (BM_Scale_md)
5. Batch Manhattan with Adam (BM_Adam)

#### Result:
<img src="./loss_plot/test2/lr_1e-3/test2_loss.png" alt="loss plot test2" style="width: 50px;"/>
<img src="./loss_plot/test2/lr_1e-3/test2_vld_error.png" alt="vld plot test2" style="width: 50px;"/>

|Training Method   | Error Rate on Testset |
|:-------  |:-------------:|
|`SGD`|8.76% |
|`SGD_md`|  10.02%             |
|`BM_md`|         4.71%      |
|`BM_Scale_md`| 3.61%|
|`BM_Adam`| 2.52%|

## Instruction for re-running
For example in file `linear_reg_low.py`, in function `main()`, there are two processes `trainProc()` and `testProc()`. Comment one and run `$python linear_reg_low.py` for either training and testing.

Note: When rerunning, files in folder `exp-result` will be overwritten, however, files in `log` or `test-result` will be appended if the parameters are the same.




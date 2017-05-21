# MNIST linear regression

### Folder description
1. `ds`: MNIST data set
2. `exp-result`: training results; stores `.npy` files of weight *W* and *b*
3. `log`:  log files record loss in different epochs and iterations
4. `test-result`: testing results; stores *error rate* on test data set

### Basic experiment settings for now
- total training data points: 60000
- epoch: 10
- minibatch size: 150
- precision: specified by `_low` and `_full` at the end of filename

### Result
- low precision: 8.24% error rate
- full precision: 7.94% error rate

### Instruction for re-running
For example in file `linear_reg_low.py`, in function `main()`, there are two processes `trainProc()` and `testProc()`. Comment one and run `$python linear_reg_low.py` for either training and testing.

Note: When rerunning, files in folder `exp-result` will be overwritten, however, files in `log` or `test-result` will be appended if the parameters are the same.




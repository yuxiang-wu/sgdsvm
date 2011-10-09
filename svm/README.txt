



1. COMPILING SVMSGD AND SVMASGD

Compiling under Unix is achieved using the traditional command "make".  
The compilation requires the libz library. This library usually comes
preinstalled on most Linux distributions, and is otherwise available
from from http://www.zlib.org.  

Compilation switches can be conveniently specified as follows:

$ make clean; make OPT='-DLOSS=HingeLoss'

The available loss functions can be looked up in file "loss.h".
The following compilation switches can be used:

    -DLOSS=<lossfunctionclass>    Select the loss function (LogLoss).
    -DBIAS=<0_or_1>               Select whether the model has bias (yes).
    -DREGULARIZED_BIAS=<0_or_1>   Select whethet the bias is regularized (no).

Compiling under Windows is possible using Cygwin, using MSYS, or using the
MSVC project files provided in the subdirectory "win" of the sgd distribution.
Make sure to read the instructions as you need to compile zlib adequately.
You then need to copy the executable files in this directory.



2. USAGE

Synopsis:

    svmsgd [options] trainfile [testfile]
    svmasgd [options] trainfile [testfile]

Programs "svmsgd" and "svmasgd" compute a L2 regularized linear model using
respectively the SGD and ASGD algorithms. Both programs perform a number of
predefined training epochs over the training set. Each epoch is followed by a
performance evaluation pass over the training set and optionally over a
validation set. The training set performance is useful to monitor the progress
of the optimization. The validation set performance is useful to estimate the
generalization performance. The recommended procedure is to monitor the
validation performance and stop the algorithm when the validation metrics no
longer improve. In the limit of large number of examples, program "svmasgd"
should reach this point after one or two epochs only.

Both programs accept the same options:

    -lambda x       : Regularization parameter (default: 1e-05.)
    -epochs n       : Number of training epochs (default: 5)
    -dontnormalize  : Do not normalize the L2 norm of patterns.
    -maxtrain n     : Restrict training set to the first n examples.

The programs assume that the training data file already contains randomly
shuffled examples. In addition, unless option -dontnormalize is specified,
every input vector is scaled to unit norm when it is loaded.

Several kinds of data files are supported:

  * Text files in svmlight format (suffix ".txt"),
  * Dedicated binary files (suffix ".bin"),
  * Gzipped versions of the above (suffix ".txt.gz" or ".bin.gz").



3. PREPROCESSING

Please follow the instructions in file "data/README.txt" to populate the
directories "data/rcv1" and "data/pascal". Three preprocessing programs can
then be used to generate training and validation data files.
These programs take no argument. 

 * Program "prep_alpha" preprocesses the alpha dataset: loading the original
   training data file, applying a random permutation, and producing suitable
   training and validation files named "alpha.train.bin.gz" and
   "alpha.test.bin.gz".

 * Program "prep_webspam" preprocesses the webspam dataset: loading the
   original training data file, applying a random permutation, and producing
   suitable training and validation files named "webspam.train.bin.gz" and
   "webspam.test.bin.gz".  You probably need 16GB of RAM for this dataset.

 * Program "prep_rcv1" preprocesses the RCV1-V2 dataset. The task consists of
   identifying documents belonging to the CCAT category. In order to obtain a
   larger training set, the official training and testing set are swapped: the
   four official test files become the training set, and the official training
   file becomes the validation set.  Program "prep_rcv1" therefore recomputes
   the TF-IDF features in order to base the IDF coefficients on our new
   training set. As usual the data is randomly shuffled before producing the
   data files "rcv1.train.bin.gz" and "rcv1.test.bin.gz".



4. EXAMPLE: RCV1-V2, HINGE LOSS

Preparation

    $ ./prep_rcv1
    $ make clean && make OPT=-DLOSS=HingeLoss


Using stochastic gradient descent (svmsgd):

    $ ./svmsgd -lambda 1e-4 rcv1.train.bin.gz rcv1.test.bin.gz
    # Running: ./svmsgd -lambda 0.0001 -epochs 5
    # Compiled with:  -DLOSS=HingeLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file rcv1.train.bin.gz
    # Read 370541+410724=781265 examples.
    # Reading file rcv1.test.bin.gz
    # Read 10786+12363=23149 examples.
    # Number of features 47153.
    # Using eta0=0.5
    --------- Epoch 1.
    Training on [0, 781264].
    wNorm=1141.36 wBias=0.0690558
    Total training time 0.2 secs.
    train: Testing on [0, 781264].
    train: Loss=0.170795958407 Cost=0.227863744764 Misclassification=5.686%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187133783125 Cost=0.244201569483 Misclassification=6.061%.
    --------- Epoch 2.
    Training on [0, 781264].
    wNorm=1141 wBias=0.0709104
    Total training time 0.39 secs.
    train: Testing on [0, 781264].
    train: Loss=0.170593793946 Cost=0.227643679517 Misclassification=5.671%.
    test:  Testing on [0, 23148].
    test:  Loss=0.18709754618 Cost=0.244147431751 Misclassification=6.043%.
    --------- Epoch 3.
    Training on [0, 781264].
    wNorm=1140.74 wBias=0.0715381
    Total training time 0.58 secs.
    train: Testing on [0, 781264].
    train: Loss=0.17054568518 Cost=0.227582476075 Misclassification=5.668%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187105426667 Cost=0.244142217562 Misclassification=6.035%.
    --------- Epoch 4.
    Training on [0, 781264].
    wNorm=1140.62 wBias=0.0715058
    Total training time 0.77 secs.
    train: Testing on [0, 781264].
    train: Loss=0.17051783095 Cost=0.227548613852 Misclassification=5.67%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187110187534 Cost=0.244140970436 Misclassification=6.039%.
    --------- Epoch 5.
    Training on [0, 781264].
    wNorm=1140.58 wBias=0.0715744
    Total training time 0.97 secs.
    train: Testing on [0, 781264].
    train: Loss=0.170506173396 Cost=0.227534977776 Misclassification=5.67%.
    test:  Testing on [0, 23148].
    test:  Loss=0.187108951994 Cost=0.244137756374 Misclassification=6.026%.


Using averaged stochastic gradient descent (svmasgd):

    $ ./svmasgd -lambda 1e-4 rcv1.train.bin.gz rcv1.test.bin.gz
    # Running: ./svmasgd -lambda 0.0001 -epochs 2
    # Compiled with:  -DLOSS=HingeLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file rcv1.train.bin.gz
    # Read 370541+410724=781265 examples.
    # Reading file rcv1.test.bin.gz
    # Read 10786+12363=23149 examples.
    # Number of features 47153.
    # Using eta0=0.5
    --------- Epoch 1.
    Training on [0, 781264].
    wNorm=1151.98 aNorm=1154.22 wBias=0.0660923 aBias=0.0724829
    Total training time 0.31 secs.
    train: Testing on [0, 781264].
    train: Loss=0.170190851016 Cost=0.227789813418 Misclassification=5.682%.
    test:  Testing on [0, 23148].
    test:  Loss=0.186527332121 Cost=0.244126294523 Misclassification=6.074%.
    --------- Epoch 2.
    Training on [0, 781264].
    wNorm=1145.12 aNorm=1148.65 wBias=0.0685983 aBias=0.0716673
    Total training time 0.62 secs.
    train: Testing on [0, 781264].
    train: Loss=0.170209090871 Cost=0.22746499297 Misclassification=5.675%.
    test:  Testing on [0, 23148].
    test:  Loss=0.186586913647 Cost=0.243842815747 Misclassification=6.052%.


The same experiment has been run using well known SvmLight and SvmPerf
software packages (Joachims, 1999, 2006). The experiments above use a
regularization coefficient lambda=1e-4.  Although this specific value copies
the settings described by Joachims (2006), the svm_light and svm_perf command
line arguments specify the regularization coefficient using different
calculations. The equivalent commands are:

    $ svm_light -c .0127998  train.dat svmlight.model
    training time: 23642 seconds
    test error: 6.0219%
    primal:0.227488

    $ svm_perf -c 100 train.dat svmperf.model
    training time: 66 seconds.
    test error: 6.0348%
    primal: 0.2278 

And using the LibLinear's dual coordinate ascent method (Hsieh, 2008)

    $ ./liblinear-1.8/train -B 1 -s 3 -c 0.0127998 rcv1.train.txt model
    training time: 2.50 seconds
    test error: 6.0219%




4. EXAMPLE: RCV1-V2, LOG LOSS

Preparation

    $ make clean && make OPT='-DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=1'


Using stochastic gradient descent (svmsgd):

    $ ./svmsgd -lambda 5e-7 -epochs 12 rcv1.train.bin.gz rcv1.test.bin.gz 
    # Running: ./svmsgd -lambda 5e-07 -epochs 20
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=1
    # Reading file rcv1.train.bin.gz
    # Read 370541+410724=781265 examples.
    # Reading file rcv1.test.bin.gz
    # Read 10786+12363=23149 examples.
    # Number of features 47153.
    # Using eta0=16
    --------- Epoch 1.
    Training on [0, 781264].
    wNorm=63685.2 wBias=0.791322
    Total training time 0.33 secs.
    train: Testing on [0, 781264].
    train: Loss=0.132214521777 Cost=0.148135817914 Misclassification=4.87%.
    test:  Testing on [0, 23148].
    test:  Loss=0.157003879634 Cost=0.172925175771 Misclassification=5.659%.
    --------- Epoch 2.
    ....
    --------- Epoch 12.
    Training on [0, 781264].
    wNorm=46076.9 wBias=0.829737
    Total training time 3.9 secs.
    train: Testing on [0, 781264].
    train: Loss=0.116796685579 Cost=0.128315899577 Misclassification=4.243%.
    test:  Testing on [0, 23148].
    test:  Loss=0.141970092667 Cost=0.153489306665 Misclassification=5.141%.
    --------- Epoch 15.
    Training on [0, 781264].
    wNorm=45857.4 wBias=0.836092
    Total training time 4.88 secs.
    train: Testing on [0, 781264].
    train: Loss=0.116684764422 Cost=0.128149104329 Misclassification=4.234%.
    test:  Testing on [0, 23148].
    test:  Loss=0.141844303433 Cost=0.153308643341 Misclassification=5.149%.


Using averaged stochastic gradient descent (svmasgd):

    $ ./svmasgd -lambda 5e-7 -epochs 8 rcv1.train.bin.gz rcv1.test.bin.gz 
    # Running: ./svmasgd -lambda 5e-07 -epochs 20
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=1
    # Reading file rcv1.train.bin.gz
    # Read 370541+410724=781265 examples.
    # Reading file rcv1.test.bin.gz
    # Read 10786+12363=23149 examples.
    # Number of features 47153.
    Using eta0=16
    --------- Epoch 1.
    Training on [0, 781264].
    wNorm=80052.8 aNorm=84590.9 wBias=0.804059 aBias=1.15136
    Total training time 0.53 secs.
    train: Testing on [0, 781264].
    train: Loss=0.124878176518 Cost=0.144891373784 Misclassification=4.371%.
    test:  Testing on [0, 23148].
    test:  Loss=0.152986679405 Cost=0.17299987667 Misclassification=5.184%.
    --------- Epoch 2.
    ...
    --------- Epoch 8.
    Training on [0, 781264].
    wNorm=49068.8 aNorm=53944.2 wBias=0.777135 aBias=0.927685
    Total training time 4.46 secs.
    train: Testing on [0, 781264].
    train: Loss=0.115982336689 Cost=0.12824954665 Misclassification=4.226%.
    test:  Testing on [0, 23148].
    test:  Loss=0.141690611036 Cost=0.153957820997 Misclassification=5.132%.


The same experiment has been run using the liblinear package.

 - Using the tron optimizer:

    $ ./liblinear-1.8/train -B 1 -s 0 -c 2.55994 rcv1.train.txt model
    training time: 33.40 seconds
    test error: 5.137%

 - Using the dual coordinate ascent optimizer:

    $ ./liblinear-1.8/train -B 1 -s 7 -c 2.55994 rcv1.train.txt model
    training time: 15.18 seconds
    test error: 5.128%


5. EXAMPLE: ALPHA, LOG LOSS

Preparation

    $ ./prep_alpha
    $ make clean && make

Using stochastic gradient descent:

    $ ./svmsgd -lambda 1e-6 -epochs 100 alpha.train.bin.gz alpha.test.bin.gz
    # Running: ./svmsgd -lambda 1e-06 -epochs 100
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file alpha.train.bin.gz
    # Read 124859+125141=250000 examples.
    # Reading file alpha.test.bin.gz
    # Read 124654+125346=250000 examples.
    # Number of features 501.
    # Using eta0=0.5
    --------- Epoch 1.
    ...
    --------- Epoch 100.
    Training on [0, 249999].
    wNorm=12614.8 wBias=-0.0389792
    Total training time 51.69 secs.
    train: Testing on [0, 249999].
    train: Loss=0.465435876783 Cost=0.471743261888 Misclassification=21.72%.
    test:  Testing on [0, 249999].
    test:  Loss=0.467762934377 Cost=0.474070319482 Misclassification=21.89%.


Using averaged stochastic gradient descent:

    $ ./svmasgd -lambda 1e-6 -epochs 10 alpha.train.bin.gz alpha.test.bin.gz
    # Running: ./svmasgd -lambda 1e-06 -epochs 100
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file alpha.train.bin.gz
    # Read 124859+125141=250000 examples.
    # Reading file alpha.test.bin.gz
    # Read 124654+125346=250000 examples.
    # Number of features 501.
    # Using eta0=0.5
    --------- Epoch 1.
    ...
    --------- Epoch 10.
    Training on [0, 249999].
    wNorm=13113.4 aNorm=12387.2 wBias=-0.0236621 aBias=-0.0634538
    Total training time 8.08 secs.
    train: Testing on [0, 249999].
    train: Loss=0.464875154208 Cost=0.471431871494 Misclassification=21.69%.
    test:  Testing on [0, 249999].
    test:  Loss=0.46715672888 Cost=0.473713446165 Misclassification=21.85%.


6. EXAMPLE: WEBSPAM, LOG LOSS

This dataset has a very high number of features.
We recommend having 16GB of ram for this.

Preparation. 

    $ ./prep_webspam
    $ make clean && make

Using stochastic gradient descent:

    $ ./svmsgd -lambda 1e-7 -epochs 10 webspam.train.bin.gz webspam.test.bin.gz
    # Running: ./svmsgd -lambda 1e-07 -epochs 10
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file webspam.train.bin.gz
    # Read 151610+98390=250000 examples.
    # Reading file webspam.test.bin.gz
    # Read 60579+39421=100000 examples.
    # Number of features 16609144.
    # Using eta0=8
    --------- Epoch 1.
    Training on [0, 249999].
    wNorm=116852 wBias=-1.66619
    Total training time 8.11 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0293753128812 Cost=0.0352179377339 Misclassification=1.006%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0341769076299 Cost=0.0400195324826 Misclassification=1.109%.
    --------- Epoch 2.
    ...
    --------- Epoch 10.
    Training on [0, 249999].
    wNorm=173809 wBias=-0.871056
    Total training time 76.38 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0135216161482 Cost=0.0222120638234 Misclassification=0.2664%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0207285095923 Cost=0.0294189572676 Misclassification=0.51%.


Using averaged stochastic gradient descent:

    $ ./svmasgd -lambda 1e-7 -epochs 10 webspam.train.bin.gz webspam.test.bin.gz
    # Running: ./svmasgd -lambda 1e-07 -epochs 10
    # Compiled with:  -DLOSS=LogLoss -DBIAS=1 -DREGULARIZED_BIAS=0
    # Reading file webspam.train.bin.gz
    # Read 151610+98390=250000 examples.
    # Reading file webspam.test.bin.gz
    # Read 60579+39421=100000 examples.
    # Number of features 16609144.
    # Using eta0=8
    --------- Epoch 1.
    Training on [0, 249999].
    wNorm=120486 aNorm=88375.5 wBias=-1.68667 aBias=-1.54283
    Total training time 18.3 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0247099559994 Cost=0.0307342739681 Misclassification=0.8292%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0287062066371 Cost=0.0347305246059 Misclassification=0.947%.
    --------- Epoch 2.
    ...
    --------- Epoch 10.
    Training on [0, 249999].
    wNorm=183711 aNorm=161889 wBias=-0.893128 aBias=-1.02782
    Total training time 193.32 secs.
    train: Testing on [0, 249999].
    train: Loss=0.0138575580013 Cost=0.0230430876888 Misclassification=0.3884%.
    test:  Testing on [0, 99999].
    test:  Loss=0.0204913948969 Cost=0.0296769245844 Misclassification=0.584%.



1. COMPILING CRFSGD AND CRFASGD

Compiling under Unix is achieved using the traditional command "make".  
The compilation requires the libz library. This library usually comes
preinstalled on most Linux distributions, and is otherwise available
from from http://www.zlib.org.  

Compiling under Windows is possible using Cygwin, using MSYS, or using the
MSVC project files provided in the subdirectory "win" of the sgd distribution.
Make sure to read the instructions as you need to compile zlib adequately.
You then need to copy the executable files in this directory.


2. USAGE

Synopsis (training): 
        crfsgd [options] model template traindata [validdata]
        crfasgd [options] model template traindata [validdata]

Synopsis (tagging): 
        crfsgd -t model testdata
        crfasgd -t model testdata

Program "crfsgd" and "crfasgd" implement stochastic gradient algorithms for
training conditional random field models. The program inputs are modeled after
the well known CRF++ program (http://crfpp.sourceforge.net).  In particular
these programs use the same format for the template files and the data
files. These formats are well documented on the CRF++ web page. Programs
crfsgd and crfasgd can also directly read gzipped data files, provided that
the file name ends with suffix ".gz".

When operating in training mode, these program construct a CRF according to
the template file and perform a predefined number of training epochs on the
training data. Every so many epochs are followed by a performance evaluation
pass over the training set and optionally over a validation set. The
performance evaluation procedure can pipe the tags into an external evaluation
program such as the standard CONLL evaluation script conlleval. The training
set performance is useful to monitor the progress of the optimization. The
validation set performance is useful to estimate the generalization
performance. The recommended procedure is to monitor the validation
performance and stop the algorithm when the validation metrics no longer
improve. In the limit of large number of examples, program "crfasgd" should
reach this point after one or two epochs only. The model is saved in the
specified model file (which in fact is a compressed text file.)

Both programs accept the same options when used in training mode.

 -c <num> : capacity control parameter (1.0)
 -f <num> : threshold on the occurences of each feature (3)
 -r <num> : total number of epochs (50)
 -h <num> : epochs between each testing phase (5)
 -e <cmd> : performance evaluation command (conlleval -q)
 -s <num> : initial learning rate
 -q       : silent mode

Using option -t switches to the tagging mode. When operating in tagging mode, the
program reads the model, tags every sentence from the provided test data file,
and outputs the tags on the standard output using a format suitable for the
standard evaluation script conlleval.


2. RUNNING THE STOCHASTIC GRADIENT CRF ON THE CONLL CHUNKING TASK

Please follow the instructions in file "data/README.txt" to populate the
directories "data/conll". The gzipped files are directly usable.
No further preprocessing is necessary.

Training a model using stochastic gradient descent.

    $ ./crfsgd -c 1.0 -f 3 model.gz template \
          ../data/conll2000/train.txt.gz ../data/conll2000/test.txt.gz
    Reading template file template.
    ...
    Reading and preprocessing ../data/conll2000/train.txt.gz.
    ...
    Reading and preprocessing ../data/conll2000/test.txt.gz.
    ...
    [Calibrating] --  1000 samples
    ...
    [Epoch 1] -- wnorm=3428.22 time=15.66s.
    [Epoch 2] -- wnorm=4981.97 time=21.59s.
    [Epoch 3] -- wnorm=6099.82 time=27.5s.
    [Epoch 4] -- wnorm=6888.25 time=33.41s.
    [Epoch 5] -- wnorm=7465.87 time=39.29s.
    Training perf: sentences=8936 loss=0.8069 obj=1.22464 err=2454 (1.15904%)
    accuracy:  98.84%; precision:  97.95%; recall:  98.04%; FB1:  98.00
    Testing perf: sentences=2012 loss=2.35348 obj=2.77122 err=1997 (4.21513%)
    accuracy:  95.78%; precision:  93.31%; recall:  93.47%; FB1:  93.39
    [Epoch 6] -- wnorm=7904.99 time=45.19s.
    [Epoch 7] -- wnorm=8238.51 time=51.07s.
    [Epoch 8] -- wnorm=8494.34 time=56.96s.
    [Epoch 9] -- wnorm=8695.67 time=62.84s.
    [Epoch 10] -- wnorm=8859.06 time=68.73s.
    Training perf: sentences=8936 loss=0.592674 obj=1.08837 err=1492 (0.704681%)
    accuracy:  99.30%; precision:  98.81%; recall:  98.64%; FB1:  98.72
    Testing perf: sentences=2012 loss=2.27945 obj=2.77514 err=1950 (4.11592%)
    accuracy:  95.88%; precision:  93.60%; recall:  93.43%; FB1:  93.51
    ...
    [Epoch 46] -- wnorm=9670.23 time=281.04s.
    [Epoch 47] -- wnorm=9670.91 time=286.93s.
    [Epoch 48] -- wnorm=9670.6 time=292.83s.
    [Epoch 49] -- wnorm=9670.94 time=298.74s.
    [Epoch 50] -- wnorm=9669.67 time=304.64s.
    Training perf: sentences=8936 loss=0.476964 obj=1.01802 err=692 (0.326836%)
    accuracy:  99.67%; precision:  99.42%; recall:  99.26%; FB1:  99.34
    Testing perf: sentences=2012 loss=2.20519 obj=2.74624 err=1889 (3.98717%)
    accuracy:  96.01%; precision:  93.94%; recall:  93.55%; FB1:  93.74
    Saving model file model.gz.
    Done!  304.64 seconds.



Training a model using averaged stochastic gradient descent.


    $ ./crfasgd -c 1.0 -f 3 -r 10 model.gz template \
          ../data/conll2000/train.txt.gz ../data/conll2000/test.txt.gz
    Reading template file template.
    ...
    Reading and preprocessing ../data/conll2000/train.txt.gz.
    ...
    Reading and preprocessing ../data/conll2000/test.txt.gz.
    ...
    [Calibrating] --  1000 samples
    ...
    [Epoch 1] -- wnorm=3471.68 anorm=2365.67 time=16.23s.
    [Epoch 2] -- wnorm=5093.8 anorm=3242.53 time=22.47s.
    [Epoch 3] -- wnorm=6281.57 anorm=3953.69 time=28.68s.
    [Epoch 4] -- wnorm=7128.28 anorm=4545.72 time=34.89s.
    [Epoch 5] -- wnorm=7748.74 anorm=5040.22 time=41.11s.
    Training perf: sentences=8936 loss=0.993147 obj=1.27516 err=3642 (1.72%)
    accuracy:  98.28%; precision:  97.15%; recall:  96.99%; FB1:  97.07
    Testing perf: sentences=2012 loss=2.2536 obj=2.53562 err=1925 (4.06%)
    accuracy:  95.94%; precision:  93.71%; recall:  93.47%; FB1:  93.59
    [Epoch 6] -- wnorm=8219.54 anorm=5456.28 time=47.33s.
    [Epoch 7] -- wnorm=8569.78 anorm=5814.36 time=53.52s.
    [Epoch 8] -- wnorm=8858.13 anorm=6126.72 time=59.7s.
    [Epoch 9] -- wnorm=9059.94 anorm=6403.14 time=65.91s.
    [Epoch 10] -- wnorm=9230.78 anorm=6637.07 time=72.1s.
    Training perf: sentences=8936 loss=0.744621 obj=1.11599 err=2332 (1.1%)
    accuracy:  98.90%; precision:  98.14%; recall:  97.99%; FB1:  98.06
    Testing perf: sentences=2012 loss=2.21817 obj=2.58953 err=1862 (3.93%)
    accuracy:  96.07%; precision:  93.88%; recall:  93.69%; FB1:  93.79
    Saving model file model.gz.
    Done!  73.69 seconds.


Testing the final model (using crfsgd or crfasgd is equivalent.)

    $ ./crfsgd -t model.gz ../data/conll2000/test.txt.gz | ./conlleval
    processed 47377 tokens with 23852 phrases; found: 23805 phrases; correct: 22348.
    accuracy:  96.07%; precision:  93.88%; recall:  93.69%; FB1:  93.79
                 ADJP: precision:  80.00%; recall:  73.97%; FB1:  76.87  405
                 ADVP: precision:  82.94%; recall:  80.83%; FB1:  81.87  844
                CONJP: precision:  55.56%; recall:  55.56%; FB1:  55.56  9
                 INTJ: precision: 100.00%; recall:  50.00%; FB1:  66.67  1
                  LST: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
                   NP: precision:  94.44%; recall:  94.16%; FB1:  94.30  12386
                   PP: precision:  96.61%; recall:  97.84%; FB1:  97.22  4872
                  PRT: precision:  77.23%; recall:  73.58%; FB1:  75.36  101
                 SBAR: precision:  88.65%; recall:  84.67%; FB1:  86.62  511
                   VP: precision:  93.73%; recall:  94.10%; FB1:  93.91  4676


Comparing with CRF++ (on a different machine, about twice slower.)

    $ crf_learn -c 1.0 -f 3 template train.txt model
    ...
    Number of sentences: 8936
    Number of features:  1679700
    ...    iter=18 terr=0.04522 serr=0.45636 act=1679700 obj=24917.57905 diff=0.02882
    ...    iter=36 terr=0.02188 serr=0.27775 act=1679700 obj=13697.78077 diff=0.01717
    ...    iter=71 terr=0.00518 serr=0.09109 act=1679700 obj=9654.43394 diff=0.00167
    ...    iter=142 terr=0.00340 serr=0.06256 act=1679700 obj=9042.07254 diff=0.00007
    Done!4335.34 s

    $ crf_test -m model test.txt | tr '\t' ' ' | ./conlleval 
    processed 47377 tokens with 23852 phrases; found: 23799 phrases; correct: 22334.
    accuracy:  96.02%; precision:  93.84%; recall:  93.64%; FB1:  93.74
                 ADJP: precision:  79.71%; recall:  74.43%; FB1:  76.98  409
                 ADVP: precision:  83.18%; recall:  81.06%; FB1:  82.11  844
                CONJP: precision:  55.56%; recall:  55.56%; FB1:  55.56  9
                 INTJ: precision: 100.00%; recall:  50.00%; FB1:  66.67  1
                  LST: precision:   0.00%; recall:   0.00%; FB1:   0.00  0
                   NP: precision:  94.36%; recall:  94.03%; FB1:  94.19  12378
                   PP: precision:  96.71%; recall:  97.82%; FB1:  97.26  4866
                  PRT: precision:  79.05%; recall:  78.30%; FB1:  78.67  105
                 SBAR: precision:  88.65%; recall:  84.67%; FB1:  86.62  511
                   VP: precision:  93.63%; recall:  93.99%; FB1:  93.81  4676



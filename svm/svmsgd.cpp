// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2007- Leon Bottou

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "assert.h"
#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include "loss.h"
#include "data.h"

using namespace std;

// ---- Loss function
// Compile with -DLOSS=xxxx to define the loss function.
// Loss functions are defined in file loss.h)
#ifndef LOSS
# define LOSS HingeLoss
#endif

// ---- Bias term
// Compile with -DBIAS=[1/0] to enable/disable the bias term.
// Compile with -DREGULARIZED_BIAS=1 to enable regularization on the bias term

#ifndef BIAS
# define BIAS 1
#endif
#ifndef REGULARIZED_BIAS
# define REGULARIZED_BIAS 0
#endif



// ---- Plain stochastic gradient descent

class SvmSgd
{
public:
  SvmSgd(int dim, double lambda);
  void train(int imin, int imax, const xvec_t &x, const yvec_t &y,
             const char *prefix = "");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, 
            const char *prefix = "");
private:
  double  t;
  double  lambda;
  FVector w;
  double  wscale;
  double  bias;
};



SvmSgd::SvmSgd(int dim, double l)
  : t(0), lambda(l), w(dim), wscale(1), bias(0)
{
  // Shift t in order to have a 
  // reasonable initial learning rate.
  // This assumes |x|=1 
  double maxw = 1.0 / sqrt(lambda);
  double typw = sqrt(maxw);
  double eta0 = typw / max(1.0,LOSS::dloss(-typw,1));
  t = 1 / (eta0 * lambda);
}


void 
SvmSgd::train(int imin, int imax, 
              const xvec_t &xp, const yvec_t &yp,
              const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  for (int i=imin; i<=imax; i++)
    {
      // process pattern x
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double a = dot(w,x) * wscale + bias;
      // determine gain
      double eta = 1.0 / (lambda * t);
#if BIAS
      double etab = eta * 0.01; // slower on the bias!
#endif
      // update for regularization term
      wscale *= (1 - eta * lambda);
      if (wscale < 1e-9)
        {
          w.scale(wscale);
          wscale = 1;
        }
#if REGULARIZED_BIAS
      bias *= (1 - etab * lambda);
#endif
      // update for loss term
      double d = LOSS::dloss(a, y);
      if (d != 0)
        {
          w.add(x, eta * d / wscale);
#if BIAS
          bias += etab * d;
#endif
        }
      t += 1;
    }
  double wnorm =  dot(w,w) * wscale * wscale;
  cout << prefix << setprecision(6) 
       << "Norm: " << wnorm 
#if BIAS
       << ", Bias: " << bias 
#endif
       << endl;
}


void 
SvmSgd::test(int imin, int imax, 
             const xvec_t &xp, const yvec_t &yp, 
             const char *prefix)

{
  cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  int nerr = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double a = dot(w,x) * wscale + bias;
      if (y * a <= 0)
        nerr += 1;
      cost += LOSS::loss(a, y);
    }
  int n = imax - imin + 1;
  double wnorm =  dot(w,w) * wscale * wscale;
  cost = cost / n + 0.5 * lambda * wnorm;
  cout << prefix << setprecision(4)
       << "Misclassification: " << (double)nerr * 100.0 / n << "%." << endl;
  cout << prefix << setprecision(12) 
       << "Cost: " << cost << "." << endl;
}




// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 5;

void
usage(const char *progname)
{
  const char *s = ::strchr(progname,'/');
  progname = (s) ? s + 1 : progname;
  cerr << "Usage: " << progname << " [options] trainfile [testfile]" << endl
       << "Options:" << endl;
#define NAM(n) "    " << setw(16) << left << n << setw(0) << ": "
#define DEF(v) " (default: " << v << ".)"
  cerr << NAM("-lambda x")
       << "Regularization parameter" << DEF(lambda) << endl
       << NAM("-epochs n")
       << "Number of training epochs" << DEF(epochs) << endl
       << NAM("-dontnormalize")
       << "Do not normalize the L2 norm of patterns." << endl;
#undef NAM
#undef DEF
  ::exit(10);
}

void
parse(int argc, const char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
        {
          if (trainfile == 0)
            trainfile = arg;
          else if (testfile == 0)
            testfile = arg;
          else
            usage(argv[0]);
        }
      else
        {
          while (arg[0] == '-') 
            arg += 1;
          string opt = arg;
          if (opt == "lambda" && i+1<argc)
            {
              lambda = atof(argv[++i]);
              assert(lambda>0 && lambda<1e4);
            }
          else if (opt == "epochs" && i+1<argc)
            {
              epochs = atoi(argv[++i]);
              assert(epochs>0 && epochs<1e6);
            }
          else if (opt == "dontnormalize")
            {
              normalize = false;
            }
          else
            {
              cerr << "Option " << argv[i] << " not recognized." << endl;
              usage(argv[0]);
            }

        }
    }
  if (! trainfile)
    usage(argv[0]);
}

void 
config(const char *progname)
{
#define NAME(x) #x
#define NAME2(x) NAME(x)
  cout << "# Running: " << progname
       << " -lambda " << lambda
       << " -epochs " << epochs
       << (!normalize ? "-dontnormalize" : "")
       << endl;
  cout << "# Compiled with: "
       << " -DLOSS=" << NAME2(LOSS)
       << " -DBIAS=" << BIAS
#if BIAS
       << " -DREGULARIZED_BIAS=" << REGULARIZED_BIAS
#endif
       << endl;
}

// --- main function

int dim;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;

int main(int argc, const char **argv)
{
  parse(argc, argv);
  config(argv[0]);
  if (trainfile)
    load_datafile(trainfile, xtrain, ytrain, dim, normalize);
  if (testfile)
    load_datafile(testfile, xtest, ytest, dim, normalize);
  cout << "# Number of features " << dim << "." << endl;
  // train
  SvmSgd svm(dim, lambda);
  Timer timer;
  int imin = 0;
  int imax = xtrain.size() - 1;
  int tmin = 0;
  int tmax = xtest.size() - 1;
  for(int i=0; i<epochs; i++)
    {
      cout << "--------- Epoch " << i+1 << "." << endl;
      timer.start();
      svm.train(imin, imax, xtrain, ytrain);
      timer.stop();
      cout << "Total training time " << setprecision(6) 
           << timer.elapsed() << " secs." << endl;
      svm.test(imin, imax, xtrain, ytrain, "train: ");
      if (tmax >= tmin)
        svm.test(tmin, tmax, xtest, ytest, "test:  ");
    }
  return 0;
}

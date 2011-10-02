// -*- C++ -*-
// SVM with averaged stochastic gradient (ASGD)
// Copyright (C) 2010- Leon Bottou

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

#include <algorithm>
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
# define LOSS LogLoss
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

class SvmAsgd
{
public:
  SvmAsgd(int dim, double lambda);
  void renorm();
  void train(int imin, int imax, 
             const xvec_t &x, const yvec_t &y,
             const char *prefix = "");
  void test(int imin, int imax, 
            const xvec_t &x, const yvec_t &y, 
            const char *prefix = "");
private:
  double  t;
  double  eta0;
  double  eta1;
  double  lambda;
  FVector u;
  FVector w;
  double  ubias;
  double  wbias;
  double  alpha;
  double  beta;
  double  tau;
  double  tstart;
  double  mu0;
};



SvmAsgd::SvmAsgd(int dim, double l)
  : t(0),
    lambda(l), u(dim), w(dim), ubias(0), wbias(0),
    alpha(1), beta(1), tau(0), tstart(10000), mu0(0.01)
{
  // Compute a reasonable initial learning rate.
  // This assumes |x|=1 
  double maxw = 1.0 / sqrt(lambda);
  double typw = sqrt(maxw);
  eta0 = eta1 = typw / max(1.0,LOSS::dloss(-typw,1));
}

void 
SvmAsgd::renorm()
{
  w.combine(1/beta, u, tau/beta);
  u.scale(1/alpha);
  alpha = beta = 1;
  tau = 0;
}

void 
SvmAsgd::train(int imin, int imax, 
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
      double ux = dot(u,x);
      double d = LOSS::dloss(ubias + ux / alpha, y);
      if (t < tstart)
        {
          // PLAIN SGD
          double eta = eta0 / (1 + eta0 * lambda * t);
          double etab = eta * 0.01;
          alpha = alpha / (1 - lambda * eta);
#if REGULARIZED_BIAS
          ubias *= (1 - lambda * etab);
#endif
          if (d != 0)
            u.add(x, alpha * eta * d);
#if BIAS
          ubias += etab * d;
#endif
          // copy into averaged vector
          w = u;
          wbias = ubias;
          beta = alpha;
          tau = 0;
          eta1 = eta0;
        }
      else
        {
          // continue with asgd
          double eta = eta1 / pow(1 + eta1 * lambda * (t - tstart), 0.75);
          double mu = mu0 / (1 + mu0 * (t - tstart));
          if (alpha > 1e6 || beta > 1e6 || mu > 1e6) renorm();
          alpha = alpha / (1 - lambda * eta);
          beta = beta / (1 - mu);
          tau = tau + mu * beta / alpha;
#if REGULARIZED_BIAS
          ubias *= (1 - lambda * eta);
#endif
          if (d != 0)
            {
              double etd = alpha * eta * d;
              u.add(x, etd);
              w.add(x, - tau * etd);
#if BIAS
              ubias += eta * d;
              wbias += (ubias - wbias) * mu;
#endif
            }
        }
      // sparse asgd procedure
      t += 1;
    }
  // renormalize and display
  renorm();
  cout << prefix << setprecision(6) 
       << "UNorm=" << dot(u,u)
       << " WNorm=" << dot(w,w)
#if BIAS
       << " UBias=" << ubias 
       << " WBias=" << wbias 
#endif
       << endl;
}


void 
SvmAsgd::test(int imin, int imax, 
             const xvec_t &xp, const yvec_t &yp, 
             const char *prefix)

{
  cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  renorm();
  double nerr = 0;
  double loss = 0;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double a = dot(w,x) + wbias;
      nerr += (y * a <= 0) ? 1 : 0;
      loss += LOSS::loss(a, y);
    }
  nerr = nerr / (imax - imin + 1);
  loss = loss / (imax - imin + 1);
  double cost = loss + 0.5 * lambda * dot(w,w);
  cout << prefix 
       << "Loss=" << setprecision(12) << loss
       << " Cost=" << setprecision(12) << cost 
       << " Misclassification: " << setprecision(4) << 100 * nerr << "%." 
       << endl;
}



// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 2;
int maxtrain = -1;


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
       << "Do not normalize the L2 norm of patterns." << endl
       << NAM("-maxtrain n")
       << "Restrict training set to n examples." << endl;
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
          else if (opt == "maxtrain" && i+1 < argc)
            {
              maxtrain = atoi(argv[++i]);
              assert(maxtrain > 0);
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
  cout << "# Running: " << progname;
  cout << " -lambda " << lambda;
  cout << " -epochs " << epochs;
  if (! normalize) cout << " -dontnormalize";
  if (maxtrain > 0) cout << " -maxtrain " << maxtrain;
  cout << endl;
#define NAME(x) #x
#define NAME2(x) NAME(x)
  cout << "# Compiled with: "
       << " -DLOSS=" << NAME2(LOSS)
       << " -DBIAS=" << BIAS
       << " -DREGULARIZED_BIAS=" << REGULARIZED_BIAS
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
    load_datafile(trainfile, xtrain, ytrain, dim, normalize, maxtrain);
  if (testfile)
    load_datafile(testfile, xtest, ytest, dim, normalize);
  cout << "# Number of features " << dim << "." << endl;
  // train
  SvmAsgd svm(dim, lambda);
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

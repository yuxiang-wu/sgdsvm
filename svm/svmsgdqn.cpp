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

class SvmSgdQn
{
public:
  SvmSgdQn(int dim, double lambda);
  double wnorm();
  void setEta0(double eta);
  double testOne(const SVector &x, double y, double *ploss, double *pnerr);
  void trainOne(const SVector &x, double y);
  void train(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
public:
  double evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
  void determineEta0(int imin, int imax, const xvec_t &x, const yvec_t &y);
  void determineSkip(int imin, int imax, const xvec_t &x, const yvec_t &y);
private:
  double  lambda;
  FVector w;
  FVector w1;
  FVector b;
  int     ibias;
  bool    updateB;
  int     skip;
  int     count;
  double  t;
};


/// Constructor
SvmSgdQn::SvmSgdQn(int dim, double lambda)
  : lambda(lambda), 
#if BIAS
    w(dim+1), b(dim+1), ibias(dim),
#else
    w(dim), b(dim), ibias(dim),
#endif
    updateB(false), skip(0), count(0), 
    t(0)
{
}

/// Compute norm of the weights (without the bias)
double
SvmSgdQn::wnorm()
{
  double s = dot(w,w);
#if BIAS
  s -= w[ibias] * w[ibias];
#endif
  return s;
}

/// Set initial learning rate
void
SvmSgdQn::setEta0(double eta)
{
  assert(eta > 0);
  for (int i=0; i < ibias; i++)
    b.set(i, eta);
#if BIAS
  b.set(ibias, eta);
#endif
  updateB = false;
  count = skip;
}


/// Compute the output for one example.
double
SvmSgdQn::testOne(const SVector &x, double y, double *ploss, double *pnerr)
{
#if BIAS
  double bias = w[ibias];
  double s = dot(w,x) + bias;
#else
  double s = dot(w,x);
#endif
  if (ploss)
    *ploss += LOSS::loss(s, y);
  if (pnerr)
    *pnerr += (s * y <= 0) ? 1 : 0;
  return s;
}

/// Perform one iteration of the SGD algorithm with specified gains
void
SvmSgdQn::trainOne(const SVector &x, double y)
{
#if BIAS
  double bias = w[ibias];
  double s = dot(w,x) + bias;
#else
  double s = dot(w,x);
#endif
  double d = LOSS::dloss(s, y);
  // compute gains
  if (updateB)
    {
#if BIAS
      double bias = w1[ibias];
      double s1 = dot(w1,x) + bias;
#else
      double s1 = dot(w1,x);
#endif
#if QUASINEWTON
      const double cmin = 0.1 * lambda;
      const double cmax = 100 * lambda;
      const SVector::Pair *xp = x;
      int n = b.size();
      double ds = s - s1;
      double dl = LOSS::dloss(s1, y) - d;
      for (int i=0; i<n; i++)
        {
          double ratio = lambda;
          double xi = 0;
          if (i == xp->i)
            { 
              xi = xp->v; 
              xp++;
            }
#if BIAS
          else if (i == ibias)
            xi = 1;
#endif
          if (ds)
            {
              ratio += xi * xi * dl / ds;
              if (ratio < cmin)
                ratio = cmin;
              if (ratio > cmax)
                ratio = cmax;
            }
          else if (dl)
            {
              ratio = cmax;
            }
          b[i] = 1.0 / ( 1.0 / b[i] + skip * ratio );
        }
#else
      double diffloss = d - LOSS::dloss(s1, y);
      const double cmin = 0.1 * lambda;
      const double cmax = 100 * lambda;
      const SVector::Pair *xp = x;
      int n = b.size();
      for (int i=0; i<n; i++)
        {
          double ratio = lambda;
          double dw = w1[i] - w[i];
          double dg = 0;
          if (i == xp->i)
            { 
              dg += diffloss * xp->v;  
              xp++; 
            }
#if BIAS
          else if (i == ibias)
            dg += diffloss;  
#endif
          if (dw)
            {
              ratio += dg / dw;
              if (ratio < cmin)
                ratio = cmin;
              if (ratio > cmax)
                ratio = cmax;
            }
          else if (dg)
            {
              ratio = cmax;
            }
          b[i] = 1.0 / ( 1.0 / b[i] + skip * ratio );
        }
#endif
      updateB = false;
    }
  // update for regularization term
  if (--count <= 0)
    {
      w1 = w;
      updateB = true;
      w.add(w, - skip * lambda, b);
#if REGULARIZED_BIAS
      w[ibias] = bias * (1 - skip * lambda * b[ibias]);
#elif BIAS
      w[ibias] = bias;
#endif
      count = skip;
    }
  // update for loss term
  if (d != 0)
    {
      w.add(x, d, b);
#if BIAS
      w[ibias] += d * b[ibias];
#endif
    }
}


/// Perform a training epoch
void 
SvmSgdQn::train(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  for (int i=imin; i<=imax; i++)
    {
      trainOne(xp.at(i), yp.at(i));
      t += 1;
    }
  cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
  cout << " wBias=" << w[ibias];
#endif
  cout << endl;
}

/// Perform a test pass
void 
SvmSgdQn::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  double nerr = 0;
  double loss = 0;
  for (int i=imin; i<=imax; i++)
    testOne(xp.at(i), yp.at(i), &loss, &nerr);
  nerr = nerr / (imax - imin + 1);
  loss = loss / (imax - imin + 1);
  double cost = loss + 0.5 * lambda * wnorm();
  cout << prefix 
       << "Loss=" << setprecision(12) << loss
       << " Cost=" << setprecision(12) << cost 
       << " Misclassification=" << setprecision(4) << 100 * nerr << "%." 
       << endl;
}

/// Perform one epoch with fixed eta and return cost

double 
SvmSgdQn::evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta)
{
  SvmSgdQn clone(*this); // take a copy of the current state
  assert(imin <= imax);
  clone.setEta0(eta);
  for (int i=imin; i<=imax; i++)
    clone.trainOne(xp.at(i), yp.at(i));
  double loss = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    clone.testOne(xp.at(i), yp.at(i), &loss, 0);
  loss = loss / (imax - imin + 1);
  cost = loss + 0.5 * lambda * clone.wnorm();
  cout << "Trying eta=" << eta << " yields cost " << cost << endl;
  return cost;
}

void 
SvmSgdQn::determineEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp)
{
  const double factor = 2.0;
  double loEta = 1;
  double loCost = evaluateEta(imin, imax, xp, yp, loEta);
  double hiEta = loEta * factor;
  double hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
  if (loCost < hiCost)
    while (loCost < hiCost)
      {
        hiEta = loEta;
        hiCost = loCost;
        loEta = hiEta / factor;
        loCost = evaluateEta(imin, imax, xp, yp, loEta);
      }
  else if (hiCost < loCost)
    while (hiCost < loCost)
      {
        loEta = hiEta;
        loCost = hiCost;
        hiEta = loEta * factor;
        hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
      }
  setEta0(loEta);
  cout << "Using eta0=" << loEta << endl;
}

void 
SvmSgdQn::determineSkip(int imin, int imax, const xvec_t &x, const yvec_t &y)
{
  cout << "Estimating sparsity" << endl;
  double n = 0;
  double s = 0;
  for (int j = imin; j <= imax; j++)
    {
      n += 2;
      s += x.at(j).npairs();
    }
  skip = (int) ((8 * n * ibias) / s);
  cout << "Using skip=" << skip << endl;
}


// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 5;
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

int dims;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;

int main(int argc, const char **argv)
{
  parse(argc, argv);
  config(argv[0]);
  if (trainfile)
    load_datafile(trainfile, xtrain, ytrain, dims, normalize, maxtrain);
  if (testfile)
    load_datafile(testfile, xtest, ytest, dims, normalize);
  cout << "# Number of features " << dims << "." << endl;
  // prepare svm
  int imin = 0;
  int imax = xtrain.size() - 1;
  int tmin = 0;
  int tmax = xtest.size() - 1;
  SvmSgdQn svm(dims, lambda);
  Timer timer;
  // determine eta0 using sample
  int smin = 0;
  timer.start();
  svm.determineSkip(smin, smin + min(5000, imax), xtrain, ytrain);
  svm.determineEta0(smin, smin + min(1000, imax), xtrain, ytrain);
  timer.stop();
  // train
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

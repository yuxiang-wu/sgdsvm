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

#if __cplusplus >= 201103L 
# define HAS_UNIFORMINTDISTRIBUTION 1
#elif defined(_MSC_VER) && _MSC_VER >= 1600
# define HAS_UNIFORMINTDISTRIBUTION 1
#elif defined(__GXX_EXPERIMENTAL_CXX0X__)
# define HAS_UNIFORMINT 1
#endif

#if HAS_UNIFORMINTDISTRIBUTION
# include <random>
typedef std::uniform_int_distribution<int> uniform_int_generator
#elif HAS_UNIFORMINT
# include <tr1/random>
typedef std::tr1::uniform_int<int> uniform_int_generator
#else
struct uniform_int_generator { 
  int imin, imax; 
  uniform_int_generator(int imin, int imax) : imin(imin),imax(imax) {}
  int operator()() { return imin + std::rand() % (imax - imin + 1); }
};
#endif

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

class SvmSag
{
public:
  SvmSag(int dim, double lambda, double eta=0);
  void renorm();
  double wnorm();
  double testOne(const SVector &x, double y, double *ploss, double *pnerr);
  void trainSgdOne(const SVector &x, double y, double eta, int i=-1);
  void trainOne(const SVector &x, double y, double eta, int i);
public:
  void trainInit(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix="");
  void trainSag(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix="");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix="");
public:
  double evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
  void determineEta(int imin, int imax, const xvec_t &x, const yvec_t &y);
  void setEta(double x) { eta=x; }
private:
  double  lambda;
  double  eta;
  FVector w;       // weights
  FVector g;       // gradient sum
  int     m;       // gradient count
  double  wa;      // actual w is wa*w + wb*g
  double  wb;      // initially wa=1, wb=0
  FVector sd;      // saved dloss
  int     sdimin;  // low index
  int     sdimax;  // high index
  double  wBias;   // bias
  double  gBias;   // bias gradient sum
  double  t;       // iteration counter
};

/// Constructor
SvmSag::SvmSag(int dim, double lambda, double eta)
  : lambda(lambda), eta(eta), 
    w(dim), g(dim), m(0), wa(1), wb(0), 
    sdimin(0), sdimax(-1), wBias(0), gBias(0), t(0)
{
}

/// Renormalize the weights
void
SvmSag::renorm()
{
  if (wb != 0)
    w.combine(wa, g, wb);
  else if (wa != 1)
    w.scale(wa);
  wa = 1;
  wb = 0;
}


/// Compute the norm of the weights
double
SvmSag::wnorm()
{
  renorm();
  double norm = dot(w,w);
#if REGULARIZED_BIAS
  norm += wBias * wBias;
#endif
  return norm;
}

/// Compute the output for one example.
double
SvmSag::testOne(const SVector &x, double y, double *ploss, double *pnerr)
{
  double s = dot(w,x) * wa + wBias;
  if (wb != 0)
    s += dot(g,x) * wb;
  if (ploss)
    *ploss += LOSS::loss(s, y);
  if (pnerr)
    *pnerr += (s * y <= 0) ? 1 : 0;
  return s;
}

/// Perform one SGD iteration (used to determine eta)
void
SvmSag::trainSgdOne(const SVector &x, double y, double eta, int i)
{
  assert(wb == 0);
  double s = dot(w,x) * wa + wBias;
  wa = wa * (1 - eta * lambda);
  if (wa < 1e-5) 
    renorm();
  double d = LOSS::dloss(s, y);
  if (i >= 0)
    sd[i-sdimin] = d;
  if (d != 0)
    w.add(x, eta * d / wa);
#if BIAS
  double etab = eta * 0.01;
#if REGULARIZED_BIAS
  wBias *= (1 - etab * lambda);
#endif
  wBias += etab * d;
#endif
}

/// Perform one iteration of the SAG algorithm with gain eta
/// Argument i is the index of the loss in the saved dloss vector.
void
SvmSag::trainOne(const SVector &x, double y, double eta, int i)
{
  // compute loss
  double s = dot(w,x) * wa + wBias;
  if (wb != 0)
    s += dot(g,x) * wb;
  // compute dloss
  double d = LOSS::dloss(s, y);
  double od = sd[i - sdimin];
  sd[i - sdimin] = d;
  d = d - od;
  // update weights
  g.add(x, d);
  w.add(x, - d * wb / wa);
  double decay = 1 - lambda * eta;
  wa = wa * decay;
  wb = wb * decay + eta / m;
  if (wa < 1e-5)
    renorm();
  // same for the bias
#if BIAS
  double etab = eta * 0.01;
  gBias += d;
#if REGULARIZED_BIAS
  wBias *= (1 - etab * lambda);
#endif
  wBias += etab * gBias / m;
#endif
}


/// Perform initial training epoch
void 
SvmSag::trainInit(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  assert(eta > 0);
  assert(m == 0);
  sd.resize(imax - imin + 1);
  sdimin = imin;
  sdimax = imax;
  for (int i=imin; i<=imax; i++)
    {
      m += 1;
      trainOne(xp.at(i), yp.at(i), eta, i);
      t += 1;
    }
  cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
  cout << " wBias=" << wBias;
#endif
  cout << endl;
}


/// Perform a SAG training epoch
void 
SvmSag::trainSag(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  assert(imin >= sdimin);
  assert(imax <= sdimax);
  assert(eta > 0);
  uniform_int_generator generator(imin, imax);
  for (int i=imin; i<=imax; i++)
    {
      int ii = generator(); 
      trainOne(xp.at(ii), yp.at(ii), eta, ii);
      t += 1;
    }
  cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
  cout << " wBias=" << wBias;
#endif
  cout << endl;
}

/// Perform a test pass
void 
SvmSag::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
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
SvmSag::evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta)
{
  renorm();
  SvmSag clone(*this); // take a copy of the current state
  assert(imin <= imax);
  for (int i=imin; i<=imax; i++)
    clone.trainSgdOne(xp.at(i), yp.at(i), eta);
  double loss = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    clone.testOne(xp.at(i), yp.at(i), &loss, 0);
  loss = loss / (imax - imin + 1);
  cost = loss + 0.5 * lambda * clone.wnorm();
  //cout << "Trying eta=" << eta << " yields cost " << cost << endl;
  return cost;
}

void 
SvmSag::determineEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp)
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
  eta = loEta / 2;   /// a bit more conservative?
  cout << "# Using eta=" << eta << endl;
}


// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 5;
int maxtrain = -1;
double eta = -1;

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
       << NAM("-eta x")
       << "Set the learning rate to x." << endl
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
          else if (opt == "eta" && i+1<argc)
            {
              eta = atof(argv[++i]);
              assert(eta>0 && eta<1e4);
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
  if (eta > 0) cout << " -eta " << eta;
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
  SvmSag svm(dims, lambda);
  Timer timer;
  // determine eta0 using sample
  int smin = 0;
  int smax = imin + min(1000, imax);
  timer.start();
  if (eta > 0)
    svm.setEta(eta);
  else
    svm.determineEta(smin, smax, xtrain, ytrain);
  timer.stop();
  // train
  for(int i=0; i<epochs; i++)
    {
      cout << "--------- Epoch " << i+1 << "." << endl;
      timer.start();
      if (i == 0)
        svm.trainInit(imin, imax, xtrain, ytrain);
      else
        svm.trainSag(imin, imax, xtrain, ytrain);
      timer.stop();
      cout << "Total training time " << setprecision(6) 
           << timer.elapsed() << " secs." << endl;
      svm.test(imin, imax, xtrain, ytrain, "train: ");
      if (tmax >= tmin)
        svm.test(tmin, tmax, xtest, ytest, "test:  ");
    }
  return 0;
}

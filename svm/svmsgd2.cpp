 

// -*- C++ -*-
// SVM with stochastic gradient

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA



// $Id$


#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cmath>

using namespace std;


typedef vector<SVector> xvec_t;
typedef vector<double> yvec_t;


// Zero for regular hinge loss.
// One for rounded hinge loss.
#define SQHINGE 0

// Zero when no bias
// One when bias term
#define BIAS 1




// -- stochastic gradient

class SvmSgd
{
public:
  SvmSgd(int dim, double lambda);
  
  void measure(int imin, int imax, 
               const xvec_t &x, const yvec_t &y);
  
  void train(int imin, int imax, 
             const xvec_t &x, const yvec_t &y,
             const char *prefix = "# ");
  void test(int imin, int imax, 
            const xvec_t &x, const yvec_t &y, 
            const char *prefix = "# ");
private:
  double  t;
  double  lambda;
  FVector w;
  double  bias;
  double  bscale;
  int skip;
  int count;
};



SvmSgd::SvmSgd(int dim, double l)
  : lambda(l), w(dim), bias(0),
    bscale(1), skip(1000)
{
  // Shift t in order to have a 
  // reasonable initial learning rate
  double eta0 = 0.1 / sqrt(lambda);
  t = 1 / (eta0 * lambda);
}

inline 
double loss(double z)
{
#if SQHINGE
  if (z < 0)
    return 0.5 - z;
  if (z < 1)
    return 0.5 * (1-z) * (1-z);
  return 0;
#else
  if (z < 1)
    return 1 - z;
  return 0;
#endif
}

inline 
double dloss(double z)
{
#if SQHINGE
  if (z < 0)
    return 1;
  if (z < 1)
    return 1-z;
  return 0;
#else
  if (z < 1)
    return 1;
  return 0;
#endif
}


void 
SvmSgd::measure(int imin, int imax, 
                const xvec_t &xp, const yvec_t &yp)
{
  cerr << "# Estimating sparsity and bscale." << endl;
  double n = 0;
  double r = 0;
  double m = 0;
  FVector c(w.size());
  for (int j=imin; j<=imax && m<=1000; j++,n++)
    {
      const SVector &x = xp.at(j);
      n += 1;
      r += x.npairs();
      const SVector::Pair *p = x;
      while (p->i >= 0 && p->i < c.size())
        {
          double z = c.get(p->i) + fabs(p->v);
          c.set(p->i, z);
          m = max(m, z);
          p += 1;
        }
    }
  skip = (int) ((8 * n * w.size()) / r);
  bscale = m / n;
  cerr << "#  using " << n << " examples." << endl;
  cerr << "#  skip: " << skip << " bscale: " << bscale << endl;
}


void 
SvmSgd::train(int imin, int imax, 
              const xvec_t &xp, const yvec_t &yp,
              const char *prefix)
{
  // -------------
  cerr << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  count = skip;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double wx = dot(w,x);
      double z = y * (wx + bias);
      double eta = 1.0 / (lambda * t);
      if (z < 1)
        {
          double etd = eta * dloss(z);
          w.add(x, etd * y);
#if BIAS
          bias += etd * y * bscale;
#endif
        }
      if (--count <= 0)
        {
          w.scale(1 - eta * lambda * skip);
          count = skip;
        }
      t += 1;
    }
  cerr << prefix << "Norm: " << dot(w,w) << ", Bias: " << bias << endl;
}


void 
SvmSgd::test(int imin, int imax, 
             const xvec_t &xp, const yvec_t &yp, 
             const char *prefix)

{
  cerr << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  int nerr = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double wx = dot(w,x);
      double z = y * (wx + bias);
      if (z <= 0)
        nerr += 1;
      if (z < 1)
        cost += loss(z);
    }
  int n = imax - imin + 1;
  cost = cost / n + 0.5 * lambda * dot(w,w);
  cerr << prefix << "Misclassification: " << (double)nerr * 100.0 / n << "%." << endl;
  cerr << prefix << "Cost: " << cost << "." << endl;
}




// --- options

string trainfile;
string testfile;
double lambda = 1e-4;
int epochs = 3;
int maxtrain = -1;

void 
usage()
{
  cerr << "Usage: svmsgd [options] trainfile [testfile]" << endl
       << "Options:" << endl
       << " -lambda <lambda>" << endl
       << " -epochs <epochs>" << endl
       << " -maxtrain <n>" << endl
       << endl;
  exit(10);
}

void 
parse(int argc, const char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
        {
          if (trainfile.empty())
            trainfile = arg;
          else if (testfile.empty())
            testfile = arg;
          else
            usage();
        }
      else
        {
          while (arg[0] == '-') arg += 1;
          string opt = arg;
          if (opt == "lambda" && i+1<argc)
            {
              lambda = atof(argv[++i]);
              cerr << "# Using lambda=" << lambda << "." << endl;
              assert(lambda>0 && lambda<1e4);
            }
          else if (opt == "epochs" && i+1<argc)
            {
              epochs = atoi(argv[++i]);
              cerr << "# Going for " << epochs << " epochs." << endl;
              assert(epochs>0 && epochs<1e6);
            }
          else if (opt == "maxtrain" && i+1<argc)
            {
              maxtrain = atoi(argv[++i]);
              assert(maxtrain > 0);
            }
          else
            usage();
        }
    }
  if (trainfile.empty())
    usage();
}


// --- loading data

int dim;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;

void
load(const char *fname, xvec_t &xp, yvec_t &yp)
{
  cerr << "# Loading " << fname << "." << endl;
  
  igzstream f;
  f.open(fname);
  if (! f.good())
    {
      cerr << "ERROR: cannot open " << fname << "." << endl;
      exit(10);
    }
  int pcount = 0;
  int ncount = 0;

  bool binary;
  string suffix = fname;
  if (suffix.size() >= 7)
    suffix = suffix.substr(suffix.size() - 7);
  if (suffix == ".dat.gz")
    binary = false;
  else if (suffix == ".bin.gz")
    binary = true;
  else
    {
      cerr << "ERROR: filename should end with .bin.gz or .dat.gz" << endl;
      exit(10);
    }

  while (f.good())
    {
      SVector x;
      double y;
      if (binary)
        {
          y = (f.get()) ? +1 : -1;
          x.load(f);
        }
      else
        {
          f >> y >> x;
        }
      if (f.good())
        {
          assert(y == +1 || y == -1);
          xp.push_back(x);
          yp.push_back(y);
          if (y > 0)
            pcount += 1;
          else
            ncount += 1;
          if (x.size() > dim)
            dim = x.size();
        }
    }
  cerr << "# Read " << pcount << "+" << ncount 
       << "=" << pcount + ncount << " examples." << endl;
}



int 
main(int argc, const char **argv)
{
  parse(argc, argv);

  // load training set
  load(trainfile.c_str(), xtrain, ytrain);
  cerr << "# Number of features " << dim << "." << endl;
  int imin = 0;
  int imax = xtrain.size() - 1;
  if (maxtrain > 0 && imax >= maxtrain)
    imax = imin + maxtrain -1;
  // prepare svm
  SvmSgd svm(dim, lambda);
  Timer timer;

  // load testing set
  if (! testfile.empty())
    load(testfile.c_str(), xtest, ytest);
  int tmin = 0;
  int tmax = xtest.size() - 1;

  svm.measure(imin, imax, xtrain, ytrain);
  for(int i=0; i<epochs; i++)
    {
      
      cerr << "# --------- Epoch " << i+1 << "." << endl;
      timer.start();
      svm.train(imin, imax, xtrain, ytrain);
      timer.stop();
      cerr << "# Total training time " << timer.elapsed() << " secs." << endl;
      svm.test(imin, imax, xtrain, ytrain, "# train: ");
      if (tmax >= tmin)
        svm.test(tmin, tmax, xtest, ytest, "# test:  ");
    }
}

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



// $Id$


#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>

using namespace std;


typedef vector<SVector> xvec_t;
typedef vector<double> yvec_t;


// Available losses
#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11

// Select loss
#define LOSS LOGLOSS


// Add bias at index zero during load.
#define REGULARIZEDBIAS 1


inline 
double loss(double z)
{
#if LOSS == LOGLOSS
  if (z > 18)
    return exp(-z);
  if (z < -18)
    return -z;
  return log(1+exp(-z));
#elif LOSS == LOGLOSSMARGIN
  if (z > 18)
    return exp(1-z);
  if (z < -18)
    return 1-z;
  return log(1+exp(1-z));
#elif LOSS == SMOOTHHINGELOSS
  if (z < 0)
    return 0.5 - z;
  if (z < 1)
    return 0.5 * (1-z) * (1-z);
  return 0;
#elif LOSS == SQUAREDHINGELOSS
  if (z < 1)
    return 0.5 * (1 - z) * (1 - z);
  return 0;
#elif LOSS == HINGELOSS
  if (z < 1)
    return 1 - z;
  return 0;
#else
# error "Undefined loss"
#endif
}


inline 
double dloss(double z)
{
#if LOSS == LOGLOSS
  if (z > 18)
    return exp(-z);
  if (z < -18)
    return 1;
  return 1 / (exp(z) + 1);
#elif LOSS == LOGLOSSMARGIN
  if (z > 18)
    return exp(1-z);
  if (z < -18)
    return 1;
  return 1 / (exp(z-1) + 1);
#elif LOSS == SMOOTHHINGELOSS
  if (z < 0)
    return 1;
  if (z < 1)
    return 1-z;
  return 0;
#elif LOSS == SQUAREDHINGELOSS
  if (z < 1)
    return (1 - z);
  return 0;
#else
  if (z < 1)
    return 1;
  return 0;
#endif
}



// -- conjugate gradient

class SvmCg
{
public:
  SvmCg(int dim, double lambda);
  void train(int imin, int imax, const xvec_t &x, const yvec_t &y,
             const char *prefix = "");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, 
            const char *prefix = "");
private:
  double  lambda;
  FVector w;
  FVector g;
  FVector u;

  int n;
  FVector ywx;
  FVector yux;
  double ww;
  double wu;
  double uu;

  double search(double tol=1e-9);
  double f(double t);
};



SvmCg::SvmCg(int dim, double l)
  : lambda(l), w(dim)
{
}


double 
SvmCg::f(double t)
{
  double cost = 0;
  for (int i=0; i<n; i++)
    cost += loss( ywx[i] + t * yux[i] );
  double norm = ww + 2 * t * wu + t * t * uu;
  return 0.5 * lambda * norm + cost / n;
}


double 
SvmCg::search(double tol)
{
  double a = 0;
  double fa = f(a);
  double b = 1;
  double fb = f(b);
  double c = b;
  double fc = fb;

  while (fb >= fa)
    {
      c = b; fc = fb;
      b = b / 2;
      assert(b >= 1e-80);
      fb = f(b);
    }
  while (fc <= fb)
    {
      c = c * 2;
      assert(c <= 1e+80);
      fc = f(c);
    }
  double e = min(b-a,c-b);
  double d = e;
  while (c - a > 2 * tol)
    {
      double x;
      double olde = e;
      e = d / 2;
      double ba = b-a;
      double bc = b-c;
      double hba = ba * (fb - fc);
      double hbc = bc * (fb - fa);
      double num = ba * hba - bc * hbc;
      bool ok = false;
      if (hba != hbc)
        {
          d = -0.5 * num / ( hba - hbc );
          x = b + d;
          if (x > a && x < c && fabs(d) < fabs(olde))
            ok = true;
        }
      else if (num == 0)
        {
          if (c - b > b - a)
            d = tol;
          else
            d = -tol;
          x = b + d;
          ok = true;
        }
      if (! ok)
        {
          const double gold = 0.3819660;
          if (c - b > b - a)
            d = gold * (c - b);
          else
            d = gold * (a - b);
          x = b + d;
        }
      double fx;
      fx = f(x);
      if (fx < fb)
        {
          if  (x < b)
            { fc=fb; c=b; fb=fx; b=x; }
          else
            { fa=fb; a=b; fb=fx; b=x; }
        }
      else
        {
          if (x < b)
            { fa=fx; a=x; }
          else
            { fc=fx; c=x; }
        }
    }
  return b;
}



void 
SvmCg::train(int imin, int imax, 
              const xvec_t &xp, const yvec_t &yp,
              const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);

  n = imax - imin + 1;
  ywx.resize(n);
  yux.resize(n);

  FVector oldg = g;
  g.clear();
  g.add(w, -lambda);
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double wx = dot(w,x);
      double z = y * wx;
      ywx[i-imin] = z;
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        {
          cost += loss(z);
          g.add(x, dloss(z) * y);
        }
    }
  ww= dot(w,w);
  cost = 0.5 * lambda * ww + cost / n;

#if 1
  if (u.size() && oldg.size())
    {
      // conjugate gradient
      oldg.add(g, -1);
      double beta = - dot(g, oldg) / dot(u, oldg);
      u.combine(beta, g, 1);
    }
  else
#endif
    {
      // first iteration
      u = g;
    }
  // line search and step
  wu = dot(w,u);
  uu = dot(u,u);
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      yux[i-imin] = y * dot(u,x);
    }
  cout << prefix << setprecision(6) 
       << "Before: ww=" << ww 
       << ", uu=" << uu
       << ", cost=" << cost << endl;


  double eta = search();
  w.add(u, eta);
}


void 
SvmCg::test(int imin, int imax, 
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
      double wx = dot(w,x);
      double z = y * wx;
      if (z <= 0)
        nerr += 1;
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        cost += loss(z);
    }
  int n = imax - imin + 1;
  double wnorm =  dot(w,w);
  cost = cost / n + 0.5 * lambda * wnorm;
  cout << prefix << setprecision(4)
       << "Misclassification: " << (double)nerr * 100.0 / n << "%." << endl;
  cout << prefix << setprecision(12) 
       << "Cost: " << cost << "." << endl;
}




// --- options

string trainfile;
string testfile;
double lambda = 1e-4;
int epochs = 5;
int trainsize = -1;

void 
usage()
{
  cerr << "Usage: svmsgd [options] trainfile [testfile]" << endl
       << "Options:" << endl
       << " -lambda <lambda>" << endl
       << " -epochs <epochs>" << endl
       << " -trainsize <n>" << endl
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
              cout << "Using lambda=" << lambda << "." << endl;
              assert(lambda>0 && lambda<1e4);
            }
          else if (opt == "epochs" && i+1<argc)
            {
              epochs = atoi(argv[++i]);
              cout << "Going for " << epochs << " epochs." << endl;
              assert(epochs>0 && epochs<1e6);
            }
          else if (opt == "trainsize" && i+1<argc)
            {
              trainsize = atoi(argv[++i]);
              assert(trainsize > 0);
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
  cout << "Loading " << fname << "." << endl;
  
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
#if REGULARIZEDBIAS
      x.set(0,1);
#endif
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
      if (trainsize > 0 && xp.size() > (unsigned int)trainsize)
        break;
    }
  cout << "Read " << pcount << "+" << ncount 
       << "=" << pcount + ncount << " examples." << endl;
}



int 
main(int argc, const char **argv)
{
  parse(argc, argv);

  // load training set
  load(trainfile.c_str(), xtrain, ytrain);
  cout << "Number of features " << dim << "." << endl;
  int imin = 0;
  int imax = xtrain.size() - 1;
  if (trainsize > 0 && imax >= trainsize)
    imax = imin + trainsize -1;
  // prepare svm
  SvmCg svm(dim, lambda);
  Timer timer;

  // load testing set
  if (! testfile.empty())
    load(testfile.c_str(), xtest, ytest);
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
}

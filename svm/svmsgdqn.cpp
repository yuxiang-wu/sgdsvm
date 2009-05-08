// -*- C++ -*-
// SVM with Stochastic Gradient Descent and diagonal Quasi-Newton approximation
// Copyright (C) 2008- Antoine Bordes & Leon Bottou

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



#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cfloat>

using namespace std;


// Select loss
#define LOSS HINGELOSS

// Magic to find loss name
#define _NAME(x) #x
#define _NAME2(x) _NAME(x)
const char *lossname = _NAME2(LOSS);

// Available losses
#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11


// set this value to 1 if you are not using sparse data
#define DENSE_DATA 0


// -- custom vector functions


void 
compute_ratio(FVector &w,
              const SVector &x, double lambda, 
              const FVector &wp, const FVector &wpp, 
              double loss)
{
  int m = max(x.size(), w.size());
  if (w.size() < m) 
    w.resize(m);
  VFloat *d = w;
  const VFloat *s = (const VFloat*) wp;
  const VFloat *sp = (const VFloat*) wpp;
  int npairs = x.npairs();
  const SVector::Pair *pairs = x;
  int j = 0;
  double diffw=0;
  for (int i=0; i<npairs; i++, pairs++)
    {
      for (; j < pairs->i; j++)
	d[j] += 1/lambda;
      j = pairs->i;
      diffw = s[j]-sp[j];
      if(diffw)
	  d[j] += diffw/ (lambda*diffw+ loss*pairs->v);
      else
      	d[j] += 1/lambda;
      j++;
    }
  for (; j<m; j++)
    d[j] += 1/lambda;
}


void
compute_ratio(FVector &w,
              const FVector &x, double lambda, 
              const FVector &wp, const FVector &wpp, 
              double loss)
{
  int m = max(x.size(), w.size());
  if (w.size() < m) 
    w.resize(m);
  VFloat *d = w;
  const VFloat *sx = (const VFloat*) x;
  const VFloat *s = (const VFloat*) wp;
  const VFloat *sp = (const VFloat*) wpp;
  for (int i=0; i<m; i++)
    {
      double diffw = s[i]-sp[i];
      if(diffw)
	d[i] += diffw/ (lambda*diffw+ loss*sx[i]);
      else
      	d[i] += 1/lambda;
    }
}


void
combine_and_clip(FVector &w,
                 double c1, const FVector &v2, double c2,
                 double vmin, double vmax)
{
  int m = max(w.size(), v2.size());
  if (w.size() < m)
    w.resize(m);
  VFloat *d = w;
  const VFloat *s = (const VFloat*) v2;
  for (int i=0; i<m; i++)
    if(s[i])
      {
	d[i] = d[i] * c1 + s[i] * c2;
	d[i] = min(max((double)d[i], vmin), vmax);
      }
}


#if DENSE_DATA == 1
# define SVector FVector
#endif

typedef vector<SVector> xvec_t;
typedef vector<double> yvec_t;


inline 
double loss(double z)
{
#if LOSS == LOGLOSS
  if (z >= 0)
    return log(1+exp(-z));
  else
    return -z + log(1+exp(z));
#elif LOSS == LOGLOSSMARGIN
  if (z >= 1)
    return log(1+exp(1-z));
  else
    return 1-z + log(1+exp(z-1));
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
  if (z < 0)
    return 1 / (exp(z) + 1);
  double ez = exp(-z);
  return ez / (ez + 1);
#elif LOSS == LOGLOSSMARGIN
  if (z < 1)
    return 1 / (exp(z-1) + 1);
  double ez = exp(1-z);
  return ez / (ez + 1);
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



// -- stochastic gradient

class SgdQn
{
public:
  SgdQn(int dim, double lambda);
  
  double printQnInfo(const FVector &Bb, double init);
  
  void calibrate(int imin, int imax, 
		 const xvec_t &xp, const yvec_t &yp);

  void train(int imin, int imax, 
             const xvec_t &x, const yvec_t &y,
	     const char *prefix);
  
  void train2(int imin, int imax, 
	      const xvec_t &x, const yvec_t &y,
	      const char *prefix);

  double test(int imin, int imax, 
            const xvec_t &x, const yvec_t &y, 
            const char *prefix);

  FVector copy_w()
  {return w;}

private:
  double  t;
  double  lambda;
  FVector w;
  double  bias;
  int skip;
  int count;

  FVector B;
  FVector Bc;
};



SgdQn::SgdQn(int dim, double l)
  : lambda(l), w(dim), skip(1000),
    B(dim), Bc(dim)
{
  double maxw = 1.0 / sqrt(lambda);
  double typw = sqrt(maxw);
  double eta0 = typw / max(1.0,dloss(-typw));
  t = 1 / (eta0 * lambda);
  for(int i=0; i<dim; i++)
    Bc.set(i, 1/lambda);
  cout << "t0 =" << t << endl;
}


double
SgdQn::printQnInfo(const FVector &Bb, double init)
{
  double bmax=-DBL_MAX, bmin=DBL_MAX, bmean=0.;
  int minb=0, maxb=0, notchg=0, imin=0, imax=0;
  for(int i=0; i<Bb.size();i++)
    {
      if(Bb[i]<bmin)
	bmin = Bb[i], imin=i;
      if(Bb[i]>bmax)
	bmax = Bb[i], imax=i;
      bmean+=Bb[i];      
      if(Bb[i]==init)
	notchg++;
    }
  for(int i=0; i<Bb.size();i++)
    {
      if(Bb[i]==bmax)
	maxb++;
      if(Bb[i]==bmin)
	minb++;
    }
  bmean /= Bb.size();
  cout  << "Bmax: " << bmax << " (i:"<<imax <<" ,@max: " << maxb << "/" << Bb.size()<<  ")\n"
	<< "Bmin: " << bmin << " (i:"<<imin <<" ,(@min: " << minb << "/" << Bb.size()<<  ")\n"
	<< "Bmean: " << bmean << "\n"
	<< "Didnt Change: " << notchg << "\n";
  return bmean;
}


void 
SgdQn::calibrate(int imin, int imax, 
		    const xvec_t &xp, const yvec_t &yp)
{
  cout << "Estimating sparsity" << endl;
  int j;
  
  // compute average gradient size
  double n = 0;
  double r = 0;
  
#if DENSE_DATA==1
  for (j=imin; j<=imax; j++,n++)
    {
      const FVector &x = xp.at(j);
      n += 1;
      r += x.size();
    }
#else
  for (j=imin; j<=imax; j++,n++)
    {
      const SVector &x = xp.at(j);
      n += 1;
      r += x.npairs();
    }
#endif    

  // compute weight decay skip
  skip = (int) ((8 * n * w.size()) / r);
  cout << " using " << n << " examples." << endl;
  cout << " skip: " << skip << endl;
}

void 
SgdQn::train(int imin, int imax, 
	     const xvec_t &xp, const yvec_t &yp,
	     const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);

  count = skip;
  bool updateB = false;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double z = y * dot(w, x);
      double eta = 1.0 / t ;

      if(updateB==true)
	{
#if LOSS < LOGLOSS
	  if (z < 1)
#endif
	    {
	      FVector w_1=w;
	      double loss_1 = dloss(z);	  
	      w.add(x, eta*loss_1*y, Bc);
	      
	      double z2 = y * dot(w,x);
	      double diffloss = dloss(z2) - loss_1;  
	      if (diffloss)
		{
		  compute_ratio(B, x, lambda, w_1, w, y*diffloss);
		  if(t>skip)
		    combine_and_clip(Bc, 
                                     (t-skip)/(t+skip),B,2*skip/(t+skip),
                                     1/(100*lambda),100/lambda);
                  else
                    combine_and_clip(Bc,
                                     t/(t+skip),B,skip/(t+skip),
                                     1/(100*lambda),100/lambda);
		  B.clear();
		  B.resize(w.size());
		}
	    }
	  updateB=false;	
	}
      else
	{
	  if(--count <= 0)
	    {
	      w.add(w,-skip*lambda*eta,Bc);	  
	      count = skip;
	      updateB=true;
	    }      
#if LOSS < LOGLOSS
	  if (z < 1)
#endif
	    {
	      w.add(x, eta*dloss(z)*y, Bc);
	    }
	}
      t += 1;
    }
  // printQnInfo(Bc, 1/lambda);
  cout << prefix << setprecision(6) 
       << "Norm2: " << dot(w,w) << ", Bias: " << 0 << endl;
}


// This train function implements a 2nd way f updating the scaling matrix
// separating the example on which a parameter update is performed 
// and the example on which the B scaling matrix is updated.
void 
SgdQn::train2(int imin, int imax, 
	      const xvec_t &xp, const yvec_t &yp,
	      const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  count = skip;
  bool updateB = false;
  FVector w_1=w;
  for (int i=imin; i<=imax; i++)
    {
      const SVector &x = xp.at(i);
      double y = yp.at(i);
      double z = y * dot(w,x);
      double loss=dloss(z);
      double eta = 1.0 / t ;

      if(updateB==true)
	{
	  double z_1 = y * dot(w_1, x);
	  double diffloss = loss - dloss(z_1);      
	    
	  if (diffloss)
	    {
	      compute_ratio(B, x, lambda, w_1, w, y*diffloss);
		      
	      if(t>skip)
		combine_and_clip(Bc, (t-skip)/(t+skip),B,2*skip/(t+skip),
                                 1/(100*lambda),100/lambda);
	      else
		combine_and_clip(Bc, t/(t+skip),B,skip/(t+skip),
                                 1/(100*lambda),100/lambda);
	      B.clear();
	      B.resize(w.size());
	    }
	  updateB=false;	
	}
      else
	{
	  if(--count <= 0)
	    {
	      w_1=w;	    
	      w.add(w,-skip*lambda*eta,Bc);
	      count = skip;
	      updateB=true;
	    }      
	}
#if LOSS < LOGLOSS
      if (z < 1)
#endif
	{
	  w.add(x, eta*loss*y, Bc);       
	}
      t += 1;      
    }
  printQnInfo(Bc, 1/lambda);
  cout << prefix << setprecision(6) 
       << "Norm2: " << dot(w,w) << ", Bias: " << 0 << endl;
}


double
SgdQn::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
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
      double z = y * (wx + bias);
      if (z <= 0)
        nerr += 1;
#if LOSS < LOGLOSS
      if (z < 1)
#endif
        cost += loss(z);
    }
  int n = imax - imin + 1;
  double loss = cost / n;
  cost = loss + 0.5 * lambda * dot(w,w);

  cout << prefix << setprecision(4)
       << "Misclassification: " << (double)nerr * 100.0 / n << "%." << endl;
  cout << prefix << setprecision(12) 
       << "Cost: " << cost << "." << endl;
  cout << prefix << setprecision(12) 
       << "Loss: " << loss << "." << endl;
  return cost/lambda; 
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
  cerr << "Usage: sgdqn [options] trainfile [testfile]" << endl
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


void
rearrange(xvec_t& xp, int dim)
{
  cout << "Preprocessing ..." << endl;
  double n = xp.size();
  FVector sum(dim);
  FVector var(dim);

  for(int ex=0; ex<n; ex++) //compute means
    sum.add(xp.at(ex));
  sum.scale(1/n);

  for(int ex=0; ex<n; ex++) //compute standard deviations
    for(int feat=1; feat<dim; feat++)
      {
	double old_var= var.get(feat);
	double val = (xp.at(ex).get(feat)-sum.get(feat))*(xp.at(ex).get(feat)-sum.get(feat));
	var.set(feat, old_var+val);
      }
  var.scale(1/n);

  for(int ex=0; ex<n; ex++) //normalize
    xp.at(ex).add(sum,-1);

  for(int ex=0; ex<n; ex++) //center
    for(int feat=1; feat<dim; feat++)
      {
	double old_x = xp.at(ex).get(feat);
	xp.at(ex).set(feat, old_x/sqrt(var.get(feat)));
      }

  for(int ex=0; ex<n; ex++) //|x|=1
    {
      double norm = sqrt(dot(xp.at(ex),xp.at(ex)));
      xp.at(ex).scale(1/norm);
    }
}

int 
main(int argc, const char **argv)
{
  parse(argc, argv);
  cout << "Loss=" << lossname 
       << " Bias=0" 
       << " RegBias=0" 
       << " Lambda=" << lambda
       << endl;

  // load training set
  load(trainfile.c_str(), xtrain, ytrain);
#if DENSE_DATA==1
  rearrange(xtrain,dim);
#endif
  cout << "Number of features " << dim << "." << endl;
  int imin = 0;
  int imax = xtrain.size() - 1;
  if (trainsize > 0 && imax >= trainsize)
    imax = imin + trainsize -1;

  // prepare svm
  SgdQn svm(dim, lambda);
  Timer timer;

  // load testing set
  if (! testfile.empty())
    {
      load(testfile.c_str(), xtest, ytest);
#if DENSE_DATA==1
      rearrange(xtest,dim);
#endif
    }
  int tmin = 0;
  int tmax = xtest.size() - 1;
  
  svm.calibrate(0, imax, xtrain, ytrain);
  for(int i=0; i<epochs; i++)
    {
      cout << "--------- Epoch " << i+1 << "." << endl;
      timer.start();
      svm.train(imin, imax, xtrain, ytrain, "train: ");
      timer.stop();
      cout << "Total training time " << setprecision(6)
           << timer.elapsed() << " secs." << endl;
      svm.test(imin, imax, xtrain, ytrain, "train: ");
      if (tmax >= tmin)
        svm.test(tmin, tmax, xtest, ytest, "test:  ");
    }
}

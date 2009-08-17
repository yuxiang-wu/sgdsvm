// -*- C++ -*-
// CRF with stochastic gradient
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


#include "wrapper.h"
#include "vectors.h"
#include "matrices.h"
#include "gzstream.h"
#include "pstream.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cctype>
#include <cmath>

using namespace std;

#if defined(_GXX_EXPERIMENTAL_CXX0X__)
# include <unordered_map>
# define hash_map unordered_map
#elsif defined(__GNUC__)
# include <ext/hash_map>
using __gnu_cxx::hash_map;
namespace __gnu_cxx {
  template<>
  struct hash<string> {
    hash<char*> h;
    inline size_t operator()(const string &s) const { return h(s.c_str());
    };
  };
};
#else
# define hash_map map
#endif

#ifndef HUGE_VAL
# define HUGE_VAL 1e+100
#endif

typedef vector<string> strings_t;
typedef vector<int> ints_t;

bool verbose = true;


// ============================================================
// Utilities


static int
skipBlank(istream &f)
{
  int c = f.get();
  while (f.good() && isspace(c) && c!='\n' && c!='\r')
    c = f.get();
  f.unget();  
  return c;
}


static int
skipSpace(istream &f)
{
  int c = f.get();
  while (f.good() && isspace(c))
    c = f.get();
  f.unget();  
  return c;
}


inline double
expmx(double x)
{
#define EXACT_EXPONENTIAL 0
#if EXACT_EXPONENTIAL
  return exp(-x);
#else
  // fast approximation of exp(-x) for x positive
# define A0   (1.0)
# define A1   (0.125)
# define A2   (0.0078125)
# define A3   (0.00032552083)
# define A4   (1.0172526e-5) 
  if (x < 13.0) 
    {
      assert(x>=0);
      double y;
      y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
      y *= y;
      y *= y;
      y *= y;
      y = 1/y;
      return y;
    }
  return 0;
# undef A0
# undef A1
# undef A2
# undef A3
# undef A4
#endif
}


static double
logSum(const VFloat *v, int n)
{
  int i;
  VFloat m = v[0];
  for (i=0; i<n; i++)
    m = max(m, v[i]);
  double s = 0;
  for (i=0; i<n; i++)
    s += expmx(m-v[i]);
  return m + log(s);
}


static double
logSum(const FVector &v)
{
  return logSum(v, v.size());
}


static void
dLogSum(double g, const VFloat *v, VFloat *r, int n)
{
  int i;
  VFloat m = v[0];
  for (i=0; i<n; i++)
    m = max(m, v[i]);
  double z = 0;
  for (i=0; i<n; i++)
    {
      double e = expmx(m-v[i]);
      r[i] = e;
      z += e;
    }
  for (i=0; i<n; i++)
    r[i] = g * r[i] / z;
}


static void
dLogSum(double g, const FVector &v, FVector &r)
{
  assert(v.size() <= r.size());
  dLogSum(g, v, r, v.size());
}




// ============================================================
// Parsing data file


int
readDataLine(istream &f, strings_t &line, int &expected)
{
  int obtained = 0;
  while (f.good())
    {
      int c = skipBlank(f);
      if (c == '\n' || c == '\r')
        break;
      string s;
      f >> s;
      if (! s.empty())
        {
          line.push_back(s);
          obtained += 1;
        }
    }
  int c = f.get();
  if (c == '\r' && f.get() != '\n')
    f.unget();
  if (obtained > 0)
    {
      if (expected <= 0)
        expected = obtained;
      else if (expected > 0 && expected != obtained)
        {
          cerr << "ERROR: expecting " << expected 
               << " columns in data file." << endl;
          exit(10);
        }
    }
  else
    skipSpace(f);
  return obtained;
}


int 
readDataSentence(istream &f, strings_t &s, int &expected)
{
  strings_t line;
  s.clear();
  while (f.good())
    if (readDataLine(f, s, expected))
      break;
  while (f.good())
    if (! readDataLine(f, s, expected))
      break;
  if (expected)
    return s.size() / expected;
  return 0;
}



// ============================================================
// Processing templates


void
checkTemplate(string tpl)
{
  const char *p = tpl.c_str();
  if (p[0]!='U' && p[0]!='B')
    {
      cerr << "ERROR: Unrecognized template type (neither U nor B.)" << endl
           << "       Template was \"" << tpl << "\"." << endl;
      exit(10);
    }
  while (p[0])
    {
      if (p[0]=='%' && p[1]=='x')
        {
          bool okay = false;
          char *n = const_cast<char*>(p);
          if (n[2]=='[') {
            strtol(n+3,&n, 10);
            while (isspace(n[0]))
              n += 1;
            if (n[0] == ',') {
              strtol(n+1, &n, 10);
              while (isspace(n[0]))
                n += 1;
              if (n[0] == ']')
                okay = true;
            }
          }
          if (okay)
            p = n;
          else {
            cerr << "ERROR: Syntax error in %x[.,,] expression." << endl
                 << "       Template was \"" << tpl << "\"." << endl;
            exit(10);
          }
        }
      p += 1;
    }
}


string
expandTemplate(string tpl, const strings_t &s, int columns, int pos)
{
  string e;
  int rows = s.size() / columns;
  const char *t = tpl.c_str();
  const char *p = t;
  
  static const char *BOS[4] = { "_B-1", "_B-2", "_B-3", "_B-4"};
  static const char *EOS[4] = { "_B+1", "_B+2", "_B+3", "_B+4"};

  while (*p)
    {
      if (p[0]=='%' && p[1]=='x' && p[2]=='[')
        {
          if (p > t)
            e.append(t, p-t);
          // parse %x[A,B] assuming syntax has been verified
          char *n;
          int a = strtol(p+3, &n, 10);
          while (n[0] && n[0]!=',')
            n += 1;
          int b = strtol(n+1, &n, 10);
          while (n[0] && n[0]!=']')
            n += 1;
          p = n;
          t = n+1;
          // catenate
          a += pos;
          if (b>=0 && b<columns)
            {
              if (a>=0 && a<rows)
                e.append(s[a*columns+b]);
              else if (a<0)
                e.append(BOS[min(3,-a-1)]);
              else if (a>=rows)
                e.append(EOS[min(3,a-rows)]);
            }
        }
      p += 1;
    }
  if (p > t)
    e.append(t, p-t);
  return e;
}


void
readTemplateFile(const char *fname, strings_t &templateVector)
{
  ifstream f(fname);
  if (! f.good())
    {
      cerr << "ERROR: Cannot open " << fname << " for reading." << endl;
      exit(10);
    }
  while(f.good())
    {
      int c = skipSpace(f);
      while (c == '#')
        {
          while (f.good() && c!='\n' && c!='\r')
            c = f.get();
          f.unget();
          c = skipSpace(f);
        }
      string s;
      getline(f,s);
      if (! s.empty())
        {
          checkTemplate(s);
          templateVector.push_back(s);        
        }
    }
  if (! f.eof())
    {
      cerr << "ERROR: Cannot read " << fname << " for reading." << endl;
      exit(10);
    }
}



// ============================================================
// Dictionary


typedef hash_map<string,int> dict_t;

class Dictionary
{
private:
  dict_t outputs;
  dict_t features;
  strings_t templates;
  strings_t outputnames;
  mutable dict_t internedStrings;
  int index;

public:
  Dictionary() : index(0) { }

  int nOutputs() const { return outputs.size(); }
  int nFeatures() const { return features.size(); }
  int nTemplates() const { return templates.size(); }
  int nParams() const { return index; }

  int output(string s) const { 
    dict_t::const_iterator it = outputs.find(s);
    return (it != outputs.end()) ? it->second : -1;
  }
  
  int feature(string s) const { 
    dict_t::const_iterator it = features.find(s);
    return (it != features.end()) ? it->second : -1;
  }

  string outputString(int i) const { return outputnames.at(i); }
  string templateString(int i) const { return templates.at(i); }

  string internString(string s) const;
  
  int initFromData(const char *tFile, const char *dFile, int cutoff=1);

  friend istream& operator>> ( istream &f, Dictionary &d );
  friend ostream& operator<< ( ostream &f, const Dictionary &d );
};


string
Dictionary::internString(string s) const
{
  dict_t::const_iterator it = internedStrings.find(s);
  if (it != internedStrings.end())
    return it->first;
#if defined(mutable)
  const_cast<Dictionary*>(this)->
#endif
  internedStrings[s] = 1;
  return s;
}


ostream&
operator<<(ostream &f, const Dictionary &d)
{
  typedef map<int,string> rev_t;
  rev_t rev;
  strings_t::const_iterator si;
  dict_t::const_iterator di;
  rev_t::const_iterator ri;
  for (di=d.outputs.begin(); di!=d.outputs.end(); di++)
    rev[di->second] = di->first;
  for (ri=rev.begin(); ri!=rev.end(); ri++)
    f << "Y" << ri->second << endl;
  for (si=d.templates.begin(); si!=d.templates.end(); si++)
    f << "T" << *si << endl;
  rev.clear();
  for (di=d.features.begin(); di!=d.features.end(); di++)
    rev[di->second] = di->first;
  for (ri=rev.begin(); ri!=rev.end(); ri++)
    f << "X" << ri->second << endl;
  return f;
}


istream& 
operator>>(istream &f, Dictionary &d)
{
  d.outputs.clear();
  d.features.clear();
  d.templates.clear();
  d.index = 0;
  int findex = 0;
  int oindex = 0;
  while (f.good())
    {
      string v;
      skipSpace(f);
      int c = f.get();
      if  (c == 'Y')
        {
          f >> v;
          if (v.empty())
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Invalid Y record in model file." << endl;
              exit(10);
            }
          if (findex>0)
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Found Y record occuring after X record." << endl;
              exit(10);
            }
          d.outputs[v] = oindex++;
        }
      else if (c == 'T')
        {
          f >> v;
          if (v.empty())
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Invalid T record." << endl;
              exit(10);
            }
          checkTemplate(v);
          d.templates.push_back(v);
        }
      else if (c == 'X')
        {
          f >> v;
          if (v.empty())
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Invalid X record." << endl;
              exit(10);
            }
          int nindex = findex;
          if (v[0]=='U')
            nindex += oindex;
          else if (v[0]=='B')
            nindex += oindex * oindex;
          else
            {
              cerr << "ERROR (reading dictionary): " 
                   << "Invalid feature in X record: " << v << endl;
              exit(10);
            }
          d.features[v] = findex;
          findex = nindex;
        }
      else
        {
          f.unget();
          break;
        }
    }
  d.index = findex;
  if (!f.good() && !f.eof())
    {
      d.outputs.clear();
      d.features.clear();
      d.templates.clear();
      d.index = 0;
    }
  d.outputnames.resize(oindex);
  for (dict_t::const_iterator it=d.outputs.begin(); 
       it!=d.outputs.end(); it++)
    d.outputnames[it->second] = it->first;
  return f;
}



typedef pair<string,int> sipair_t;
typedef vector<sipair_t> sivector_t;

struct SIPairCompare {
  bool operator() (const sipair_t &p1, const sipair_t &p2) {
    if (p1.second > p2.second) return true;
    else if (p1.second < p2.second) return false;
    else return (p1.first < p2.first);
  }
} siPairCompare;

int
Dictionary::initFromData(const char *tFile, const char *dFile, int cutoff)
{
  // clear all
  templates.clear();
  outputs.clear();
  features.clear();
  index = 0;
  
  // read templates
  if (verbose)
    cout << "Reading template file " << tFile << "." << endl;
  readTemplateFile(tFile, templates);
  int nu = 0;
  int nb = 0;
  for (unsigned int t=0; t<templates.size(); t++)
    if (templates[t][0]=='U')
      nu += 1;
    else if (templates[t][0]=='B')
      nb += 1;
  if (verbose)
    cout << "  u-templates: " << nu 
         << "  b-templates: " << nb << endl;
  if (nu + nb != (int)templates.size())
    {
      cerr << "ERROR (building dictionary): "
           << "Problem counting templates" << endl;
      exit(10);
    }
  
  // process compressed datafile
  if (verbose)
    cerr << "Scanning " << dFile << " to build dictionary." << endl;
  typedef hash_map<string,int> hash_t;
  hash_t fcount;
  int columns = 0;
  int oindex = 0;
  int sentences = 0;
  strings_t s;
  igzstream f(dFile);
  Timer timer;
  timer.start();
  while (readDataSentence(f, s, columns))
    {
      sentences += 1;
      // intern strings to save memory
      for (strings_t::iterator it=s.begin(); it!=s.end(); it++)
        *it = internString(*it);
      // expand features and count them
      int rows = s.size()/columns;
      for (int pos=0; pos<rows; pos++)
        {
          // check output keyword
          string &y = s[pos*columns+columns-1];
          dict_t::iterator di = outputs.find(y);
          if (di == outputs.end())
            outputs[y] = oindex++;
          // expand templates
          for (unsigned int t=0; t<templates.size(); t++)
            {
              string x = expandTemplate(templates[t], s, columns, pos);
              hash_t::iterator hi = fcount.find(x);
              if (hi != fcount.end())
                hi->second += 1;
              else
                fcount[x] = 1;
            }
        }
    }
  if (! f.eof())
    {
      cerr << "ERROR (building dictionary): "
           << "Problem reading data file " << dFile << endl;
      exit(10);
    }
  outputnames.resize(oindex);
  for (dict_t::const_iterator it=outputs.begin(); it!=outputs.end(); it++)
    outputnames[it->second] = it->first;
  if (verbose)
    cout << "  sentences: " << sentences 
         << "  outputs: " << oindex << endl;
  
  // sorting in frequency order
  sivector_t keys;
  for (hash_t::iterator hi = fcount.begin(); hi != fcount.end(); hi++)
    if (hi->second >= cutoff)
      keys.push_back(*hi);
  if (keys.size() <= 0)
    {
      cerr << "ERROR (building dictionary): "
           << "No features satisfy the cutoff frequency" << endl;
      exit(10);
    }
  sort(keys.begin(), keys.end(), siPairCompare);

  // allocating parameters
  for (unsigned int j=0; j<keys.size(); j++)
    {
      string k = keys[j].first;
      features[k] = index;
      if (k[0] == 'B')
        index += oindex * oindex;
      else
        index += oindex;
    }
  if (verbose)
    cout << "  cutoff: " << cutoff 
         << "  features: " << features.size() 
         << "  parameters: " << index << endl
         << "  duration: " << timer.elapsed() << " seconds." << endl;

  return sentences;
}



// ============================================================
// Preprocessing data


typedef vector<SVector> svec_t;
typedef vector<int> ivec_t;


class Sentence
{
private:
  struct Rep 
  {
    int refcount;
    int columns;
    strings_t data;
    svec_t uFeatures;
    svec_t bFeatures;
    ivec_t yLabels;
    Rep *copy() { return new Rep(*this); }
  };
  Wrapper<Rep> w;
  Rep *rep() { return w.rep(); }
  const Rep *rep() const { return w.rep(); }

public:
  Sentence() {}

  void init(const Dictionary &dict, const strings_t &s, int columns);

  int size() const { return rep()->uFeatures.size(); }
  SVector u(int i) const { return rep()->uFeatures.at(i); }
  SVector b(int i) const { return rep()->bFeatures.at(i); }
  int y(int i) const { return rep()->yLabels.at(i); }
  
  int columns() const { return rep()->columns; }
  string data(int pos, int col) const;

  friend ostream& operator<<(ostream &f, const Sentence &s);
};


void
Sentence::init(const Dictionary &dict, const strings_t &s, int columns)
{
  w.detach();
  Rep *r = rep();
  int maxcol = columns - 1;
  int maxpos = s.size()/columns - 1;
  int ntemplat = dict.nTemplates();
  r->uFeatures.clear();
  r->bFeatures.clear();
  r->yLabels.clear();
  r->columns = columns;
  // intern strings to save memory
  for (strings_t::const_iterator it=s.begin(); it!=s.end(); it++)
    r->data.push_back(dict.internString(*it));
  // expand features
  for (int pos=0; pos<=maxpos; pos++)
    {
      // labels
      string y = s[pos*columns+maxcol];
      int yindex = dict.output(y);
      r->yLabels.push_back(yindex);
      // features
      SVector u;
      SVector b;
      for (int t=0; t<ntemplat; t++)
        {
          string tpl = dict.templateString(t); 
          int findex = dict.feature(expandTemplate(tpl, s, columns, pos));
          if (findex >= 0)
            {
              if (tpl[0]=='U')
                u.set(findex, 1);
              else if (tpl[0]=='B')
                b.set(findex, 1);
            }
        }
      r->uFeatures.push_back(u);
      if (pos < maxpos)
        r->bFeatures.push_back(b);
    }
}


string
Sentence::data(int pos, int col) const
{
  const Rep *r = rep();
  if (pos>=0 && pos<size())
    if (col>=0 && col<r->columns)
      return r->data[pos*r->columns+col];
  return string();
}


ostream&
operator<<(ostream &f, const Sentence &s)
{
  int maxpos = s.size() - 1;
  int columns = s.columns();
  for (int pos = 0; pos<=maxpos; pos++) {
    for (int col = 0; col<columns; col++)
      f << s.data(pos, col) << " ";
    f << endl << "   Y" << pos << " " << s.y(pos) << endl;
    f << "   U" << pos << s.u(pos);
    if (pos < maxpos)
      f << "   B" << pos << s.b(pos);
  }
  return f;
}


typedef vector<Sentence> dataset_t;


void
loadSentences(const char *fname, const Dictionary &dict, dataset_t &data)
{
  if (verbose)
    cout << "Reading and preprocessing " << fname << "." << endl;
  Timer timer;
  int sentences = 0;
  int columns = 0;
  strings_t s;
  igzstream f(fname);
  timer.start();
  while (readDataSentence(f, s, columns))
    {
      Sentence ps;
      ps.init(dict, s, columns);
      data.push_back(ps);
      sentences += 1;
    }
  if (verbose)
    cout << "  processed: " << sentences << " sentences." << endl
         << "  duration: " << timer.elapsed() << " seconds." << endl;
}


double 
trainSparsity(const dataset_t &data, const Dictionary &dict)
{
  int nu = 0;
  int nb = 0;
  int nout = dict.nOutputs();
  for (int i=0; i<dict.nTemplates(); i++)
    if (dict.templateString(i).at(0) == 'B')
      nb += 1;
    else
      nu += 1;
  double sl = 0;
  for (int i=0; i<(int)data.size(); i++)
    sl += data.at(i).size();
  sl = sl / data.size();
  return (double)((sl*nu + (sl-1)*nb*nout)*nout)/dict.nParams();
}




// ============================================================
// Weights

struct Weights
{
  FVector w;
  FVector b;
  
  void clear() { resize(0); }
  void resize(int n);
  void setup_b(double lambda);
};


  
void 
Weights::resize(int n) 
{
  w.resize(n);
  b.clear();
}

void
Weights::setup_b(double lambda)
{
  int size = w.size();
  b.resize(size);
  b.zero();
  b.add(1.0/lambda);
}




// ============================================================
// Scorer


class Scorer
{
public:
  Sentence s;
  const Dictionary &d;
  Weights &ww;
  bool scoresOk;
  vector<FVector> uScores;
  vector<FMatrix> bScores;

  Scorer(const Sentence &s_, const Dictionary &d_, Weights &ww_);
  void computeScores();
  virtual ~Scorer() {}
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny) {}
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y) {}
  
  double viterbi(ints_t &path);
  int test();
  int test(ostream &f);
  double scoreCorrect();
  double gradCorrect(double g);
  double scoreForward();
  double gradForward(double g);
};


Scorer::Scorer(const Sentence &s_, const Dictionary &d_, Weights &ww_)
  : s(s_), d(d_), ww(ww_), scoresOk(false)
{
  assert(ww.w.size() == d.nParams());
}

void
Scorer::computeScores()
{
  if (! scoresOk)
    {
      const VFloat *w = ww.w;
      int nout = d.nOutputs();
      int npos = s.size();
      int pos;
      // compute uScores
      uScores.resize(npos);
      for (pos=0; pos<npos; pos++)
        {
          FVector &u = uScores[pos];
          u.resize(nout);
          SVector x = s.u(pos);
          for (const SVector::Pair *p = x; p->i >= 0; p++)
            for (int j=0; j<nout; j++)
              u[j] += w[p->i + j] * p->v;
        }
      // compute bScores
      bScores.resize(npos-1);
      for (pos=0; pos<npos-1; pos++)
        {
          FMatrix &b = bScores[pos];
          b.resize(nout,nout);
          SVector x = s.b(pos);
          for (const SVector::Pair *p = x; p->i >= 0; p++)
            { 
              int k = 0;
              for (int i=0; i<nout; i++)
                {
                  FVector &bi = b[i];
                  for (int j=0; j<nout; j++, k++)
                    bi[j] += w[p->i + k] * p->v;
                }
            }
        }
    }
  scoresOk = true;
}


double 
Scorer::viterbi(ints_t &path)
{
  computeScores();
  int npos = s.size();
  int nout = d.nOutputs();
  int pos, i, j;
  
  // allocate backpointer array
  vector<ints_t> pointers(npos);
  for (int i=0; i<npos; i++)
    pointers[i].resize(nout);
  
  // process scores
  FVector scores = uScores[0];
  for (pos=1; pos<npos; pos++)
    {
      FVector us = uScores[pos];
      for (i=0; i<nout; i++)
        {
          FVector bs = bScores[pos-1][i];
          bs.add(scores);
          int bestj = 0;
          double bests = bs[0];
          for (j=1; j<nout; j++)
            if (bs[j] > bests)
              { bests = bs[j]; bestj = j; }
          pointers[pos][i] = bestj;
          us[i] += bests;
        }
      scores = us;
    }
  // find best final score
  int bestj = 0;
  double bests = scores[0];
  for (j=1; j<nout; j++)
    if (scores[j] > bests)
      { bests = scores[j]; bestj = j; }
  // backtrack
  path.resize(npos);
  for (pos = npos-1; pos>=0; pos--)
    {
      path[pos] = bestj;
      bestj = pointers[pos][bestj];
    }
  return bests;
}


int
Scorer::test()
{
  ints_t path;
  int npos = s.size();
  int errors = 0;
  viterbi(path);
  for (int pos=0; pos<npos; pos++)
    if (path[pos] != s.y(pos))
      errors += 1;
  return errors;
}

int
Scorer::test(ostream &f)
{
  ints_t path;
  int npos = s.size();
  int ncol = s.columns();
  int errors = 0;
  viterbi(path);
  for (int pos=0; pos<npos; pos++)
    {
      if (path[pos] != s.y(pos))
        errors += 1;
      for (int c=0; c<ncol; c++)
        f << s.data(pos,c) << " ";
      f << d.outputString(path[pos]) << endl;
    }
  f << endl;
  return errors;
}


double 
Scorer::scoreCorrect()
{
  computeScores();
  int npos = s.size();
  int y = s.y(0);
  double sum = uScores[0][y];
  for (int pos=1; pos<npos; pos++)
    {
      int fy = y;
      y = s.y(pos);
      if (y>=0 && fy>=0)
        sum += bScores[pos-1][y][fy];
      if (y>=0)
        sum += uScores[pos][y];
    }
  return sum;
}


double
Scorer::gradCorrect(double g)
{
  computeScores();
  int npos = s.size();
  int y = s.y(0);
  VFloat vf = g;
  uGradients(&vf, 0, y, 1);
  double sum = uScores[0][y];
  for (int pos=1; pos<npos; pos++)
    {
      int fy = y;
      y = s.y(pos);
      if (y>=0 && fy>=0)
        sum += bScores[pos-1][y][fy];
      if (y>=0)
        sum += uScores[pos][y];
      if (y>=0 && fy>=0)
        bGradients(&vf, pos-1, fy, 1, y);
      if (y>=0)
      uGradients(&vf, pos, y, 1);
    }
  return sum;
}


double 
Scorer::scoreForward()
{
  computeScores();
  int npos = s.size();
  int nout = d.nOutputs();
  int pos, i;

  FVector scores = uScores[0];
  for (pos=1; pos<npos; pos++)
    {
      FVector us = uScores[pos];
      for (i=0; i<nout; i++)
        {
          FVector bs = bScores[pos-1][i];
          bs.add(scores);
          us[i] += logSum(bs);
        }
      scores = us;
    }
  return logSum(scores);
}



double 
Scorer::gradForward(double g)
{
  computeScores();
  int npos = s.size();
  int nout = d.nOutputs();
  int pos;

#define USE_FORWARD_BACKWARD 0
#if USE_FORWARD_BACKWARD
  
  FMatrix alpha(npos,nout);
  FMatrix beta(npos,nout);
  // forward
  alpha[0] = uScores[0];
  for (pos=1; pos<npos; pos++)
    {
      FVector us = uScores[pos];
      for (int i=0; i<nout; i++)
        {
          FVector bs = bScores[pos-1][i];
          bs.add(alpha[pos-1]);
          us[i] += logSum(bs);
        }
      alpha[pos] = us;
    }

  // backward
  beta[pos-1] = uScores[pos-1];
  for (pos=pos-2; pos>=0; pos--)
    {
      FVector us = uScores[pos];
      for (int i=0; i<nout; i++)
        {
          FVector bs(nout);
          const FMatrix &bsc = bScores[pos];
          for (int j=0; j<nout; j++)
            bs[j] = bsc[j][i];
          bs.add(beta[pos+1]);
          us[i] += logSum(bs);
        }
      beta[pos] = us;
    }
  // score
  double score = logSum(beta[0]);

  // collect gradients
  for (pos=0; pos<npos; pos++)
    {
      FVector b = beta[pos];
      if (pos > 0)
        {
          FVector a = alpha[pos-1];
          const FMatrix &bsc = bScores[pos-1];
          for (int j=0; j<nout; j++)
            {
              FVector bs = bsc[j];
              FVector bgrad(nout);
              for (int i=0; i<nout; i++)
                bgrad[i] = g * expmx(max(0.0, score - bs[i] - a[i] - b[j]));
              bGradients(bgrad, pos-1, 0, nout, j);
            }
        }
      FVector a = alpha[pos];
      FVector us = uScores[pos];
      FVector ugrad(nout);
      for (int i=0; i<nout; i++)
        ugrad[i] = g * expmx(max(0.0, score + us[i] - a[i] - b[i]));
      uGradients(ugrad, pos, 0, nout);
    }

#else

  // forward
  FMatrix scores(npos, nout);
  scores[0] = uScores[0];
  for (pos=1; pos<npos; pos++)
    {
      FVector us = uScores[pos];
      for (int i=0; i<nout; i++)
        {
          FVector bs = bScores[pos-1][i];
          bs.add(scores[pos-1]);
          us[i] += logSum(bs);
        }
      scores[pos] = us;
    }
  double score = logSum(scores[npos-1]);

  // backward with chain rule
  FVector tmp(nout);
  FVector grads(nout);
  dLogSum(g, scores[npos-1], grads);
  for (pos=npos-1; pos>0; pos--)
    {
      FVector ug;
      uGradients(grads, pos, 0, nout);
      for (int i=0; i<nout; i++)
        if (grads[i])
          { 
            FVector bs = bScores[pos-1][i];
            bs.add(scores[pos-1]);
            dLogSum(grads[i], bs, tmp);
            bGradients(tmp, pos-1, 0, nout, i);
            ug.add(tmp);
          }
      grads = ug;
    }
  uGradients(grads, 0, 0, nout);

#endif

  return score;
}



// ============================================================
// GScorer - compute gradients as SVectors


class GScorer : public Scorer
{
private:
  SVector grad;
public:
  GScorer(const Sentence &s_, const Dictionary &d_, Weights &ww_);
  void clear() { grad.clear(); }
  SVector gradient() { return grad; }
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny);
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y);
};


GScorer::GScorer(const Sentence &s_,const Dictionary &d_, Weights &ww_)
  : Scorer(s_, d_, ww_)
{
}


void 
GScorer::uGradients(const VFloat *g, int pos, int fy, int ny)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = fy;
  SVector x = s.u(pos);
  SVector a;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      a.set(p->i + off + j, g[j] * p->v);
  grad.add(a);
}


void 
GScorer::bGradients(const VFloat *g, int pos, int fy, int ny, int y)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(y>=0 && y<n);
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = y * n + fy;
  SVector x = s.b(pos);
  SVector a;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      a.set(p->i + off + j, g[j] * p->v);
  grad.add(a);
}



// ============================================================
// TScorer - training score: update weights directly


class TScorer : public Scorer
{
private:
  double eta;
public:
  TScorer(const Sentence &s_, const Dictionary &d_, Weights &ww_, double eta_);
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny);
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y);
};


TScorer::TScorer(const Sentence &s_, const Dictionary &d_, Weights &ww_, double eta_)
  : Scorer(s_,d_,ww_), eta(eta_)
{
}


void 
TScorer::uGradients(const VFloat *g, int pos, int fy, int ny)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = fy;
  SVector x = s.u(pos);
  VFloat *w = ww.w;
  const VFloat *b = ww.b;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    {
      int k = p->i + off;
      for (int j=0; j<ny; j++, k++)
        w[k] += g[j] * b[k] * p->v * eta;
    }
}


void 
TScorer::bGradients(const VFloat *g, int pos, int fy, int ny, int y)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(y>=0 && y<n);
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = y * n + fy;
  SVector x = s.b(pos);
  VFloat *w = ww.w;
  const VFloat *b = ww.b;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    {
      int k = p->i + off;
      for (int j=0; j<ny; j++, k++)
        w[k] += g[j] * b[k] * p->v * eta;
    }
}




// ============================================================
// Main class CrfSgd




class CrfSgd
{
  Dictionary dict;
  Weights ww;
  double lambda;
  double t0;
  double t;
  int epoch;
  double sparsity;
  double count;
  double skip;
  
  void load(istream &f);
  void save(ostream &f) const;
  void trainOnce(const Sentence &sentence, double eta);
  double findObjBySampling(const dataset_t &data, const ivec_t &sample);
  double tryEtaBySampling(const dataset_t &data, const ivec_t &sample, 
                          double eta);
  
public:
  
  CrfSgd();
  
  int getEpoch() const { return epoch; }
  const Dictionary& getDict() const { return dict; }
  double getLambda() const { return lambda; }
  double getEta() const { return 1/(t0+t); }
  double getT() const { return t; }
  double getT0() const { return t0; }
  FVector getW() const { return ww.w; }
  

  void initialize(const char *templatefile, 
                  const char *datafile, 
                  double c = 4,
                  int cutoff = 3);
  
  double adjustEta(const dataset_t &data, int sample=5000, double eta=1);

  double adjustEta(double eta=1);

  void train(const dataset_t &data, int epochs=1, Timer *tm=0);

  void test(const dataset_t &data, const char *conlleval=0, Timer *tm=0);
  
  friend istream& operator>> ( istream &f, CrfSgd &d );
  friend ostream& operator<< ( ostream &f, const CrfSgd &d );
};


CrfSgd::CrfSgd()
  : lambda(0), 
    t0(0), t(0), epoch(0),
    sparsity(.1), count(0), skip(0)
{
}

void
CrfSgd::load(istream &f)
{
  f >> dict;
  t = 0;
  epoch = 0;
  ww.clear();
  
  while (f.good())
    {
      skipSpace(f);
      int c = f.get();
      if (f.eof())
        break;
      if (c == 'T')
        {
          t = -1;
          f >> t;
          if (!f.good() || t <= 0)
            {
              cerr << "ERROR (reading model): "
                   << "Invalid iteration number: " << t << endl;
              exit(10);
            }
        }
      else if (c == 'E')
        {
          epoch = -1;
          f >> epoch;
          if (!f.good() || epoch < 0)
            {
              cerr << "ERROR (reading model): "
                   << "Invalid epoch number: " << epoch << endl;
              exit(10);
            }
        }
      else if (c == 'W')
        {
          ww.clear();
          f >> ww.w;
          if (! f.good() || ww.w.size() != dict.nParams())
            {
              cerr << "ERROR (reading model): "
                   << "Invalid weight vector size: " << ww.w.size() << endl;
              exit(10);
            }
        }
      else if (c == 'L')
        {
          lambda = -1;
          f >> lambda;
          if (! f.good() || lambda<=0)
            {
              cerr << "ERROR (reading model): "
                   << "Invalid lambda: " << lambda << endl;
              exit(10);
            }
        }
      else
        {
          cerr << "ERROR (reading model): "
               << "Unrecognized line: '" << c << "'" << endl;
          exit(10);
        }
    }
}


istream& 
operator>> (istream &f, CrfSgd &d )
{
  d.load(f);
  return f;
}


void
CrfSgd::save(ostream &f) const
{
  // save stuff
  f << dict;
  f << "L" << lambda << endl;
  f << "T" << t << endl;
  f << "E" << epoch << endl;
  f << "W" << ww.w << endl;
}


ostream& 
operator<<(ostream &f, const CrfSgd &d)
{
  d.save(f);
  return f;
}


void 
CrfSgd::initialize(const char *tfname, const char *dfname, 
                   double c, int cutoff)
{
  int n = dict.initFromData(tfname, dfname, cutoff);
  lambda = 1 / (c * n);
  t = t0 = 0;
  if (verbose)
    cout << "Using c=" << c << ", i.e. lambda=" << lambda << endl;
  ww.clear();
  ww.resize(dict.nParams());
  ww.setup_b(lambda);
}


double 
CrfSgd::findObjBySampling(const dataset_t &data, const ivec_t &sample)
{
  double loss = 0;
  int n = sample.size();
  for (int i=0; i<n; i++)
    {
      int j = sample[i];
      Scorer scorer(data[j], dict, ww);
      loss += scorer.scoreForward() - scorer.scoreCorrect();
    }
  double wnorm = dot(ww.w, ww.w);
  return loss + 0.5 * wnorm * lambda * n;
}


double
CrfSgd::tryEtaBySampling(const dataset_t &data, const ivec_t &sample,
                         double eta)
{
  double savedT = t;
  Weights savedWW = ww;
  skip = sample.size() + 1;
  count = 0;
  for (unsigned int i=0; i<sample.size(); i++)
    trainOnce(data[sample[i]], eta);
  double obj = findObjBySampling(data, sample);
  ww = savedWW;
  t = savedT;
  return obj;
}


double
CrfSgd::adjustEta(const dataset_t &data, int samples, double eta)
{
  ivec_t sample;
  assert(samples > 0);
  assert(dict.nOutputs() > 0);
  // choose sample
  int datasize = data.size();
  if (samples > datasize)
    samples = datasize;
  for (int i=0; i<samples; i++)
    sample.push_back((int)((double)rand()*datasize/RAND_MAX));
  // initial obj
  double sobj = findObjBySampling(data, sample);
  // empirically find eta that works best
  double obja = sobj;
  double objb = sobj;
  double objc = tryEtaBySampling(data, sample, eta/2);
  if (objc < sobj)
    {
      obja = tryEtaBySampling(data, sample, eta*2);
      objb = tryEtaBySampling(data, sample, eta);
      while (obja < objb)
        {
          eta = eta * 2;
          objc = objb;
          objb = obja;
          obja = tryEtaBySampling(data, sample, eta*2);
        }
    }
  while (objc < objb || objb >= sobj)
    {
      eta = eta / 2;
      obja = objb;
      objb = objc;
      objc = tryEtaBySampling(data, sample, eta/2);
    }
  
  // set t0
  return adjustEta(eta); 
}


double
CrfSgd::adjustEta(double eta)
{
  t0 = 1.0 / eta - t;
  return eta;
}


void
CrfSgd::trainOnce(const Sentence &sentence, double eta)
{
  if (++count >= skip)
    {
      // compute gradient of loss before
      SVector gb; 
      {
        GScorer scorer(sentence, dict, ww);        
        scorer.gradCorrect(+1);
        scorer.gradForward(-1);
        gb = scorer.gradient();
      }
      // apply update (with reweighted regularization)
      ww.w.add(gb, eta, ww.b);
      ww.w.add(ww.w, - count * eta * lambda, ww.b);
      count = 0;
      // compute gradient of loss after update
      SVector ga; 
      {
        GScorer scorer(sentence, dict, ww);        
        scorer.gradCorrect(+1);
        scorer.gradForward(-1);
        ga = scorer.gradient();
      }
      // update b
      double w1 = 2;
      double w2 = 20 + t/skip;
      double wt = w1+w2;
      w1 = w1 / wt;
      w2 = w2 / wt;
      int n = ww.b.size();
      VFloat *b = ww.b;
      const VFloat oneOverLambda = 1/lambda;
      const SVector::Pair *pb = gb;
      const SVector::Pair *pa = ga;
      for (int i=0; i<n; i++)
        {
          VFloat s = oneOverLambda;
          if (i == pb->i)
            {
              VFloat bi = b[i];
              VFloat gbi = pb->v;
              VFloat dwi = gbi * eta * bi;
              VFloat gai = 0;
              while (pa->i < i && pa->i >= 0) 
                pa++;
              if (pa->i == i)
                gai = pa->v;
              if (dwi != 0)
                {
                  s = dwi / (lambda * dwi - (gai - gbi));
                  s = min<VFloat>(s, oneOverLambda);
                  s = max<VFloat>(s, 0.00001 * oneOverLambda);
                }
              pb++;
            }
          b[i] = b[i] * w2 + s * w1;
        }
    }
  else
    {
      // normal path (unregularized)
      TScorer scorer(sentence, dict, ww, eta);
      scorer.gradCorrect(+1);
      scorer.gradForward(-1);
    }
  // t
  t += 1;
}


void 
CrfSgd::train(const dataset_t &data, int epochs, Timer *tm)
{
  if (t0 + t <= 0)
    {
      cerr << "ERROR (train): "
           << "Must call adjustEta() before train()." << endl;
      exit(10);
    }
  sparsity = trainSparsity(data, dict);
  ivec_t shuffle;
  shuffle.resize(data.size());
  for (int j=0; j<epochs; j++)
    {
      skip = min(1/(8*lambda*getEta()), 3.0/sparsity);
      count = 0;
      epoch += 1;
      for (unsigned int i=0; i<data.size(); i++) shuffle[i] = i;
      random_shuffle(shuffle.begin(), shuffle.end());
      if (verbose)
        cout << "[Epoch " << epoch << "] --" << " skip=" << skip << " eta=" << getEta();
      if (verbose)
        cout.flush();
      // perform epoch
      for (unsigned int i=0; i<data.size(); i++)
        trainOnce(data[shuffle[i]], getEta());
      // epoch done
      double wnorm = dot(ww.w,ww.w);
      cout << "  wnorm: " << wnorm;
      if (tm && verbose)
        cout << "  total time: " << tm->elapsed() << " seconds";
      if (verbose)
        cout << endl;
    }
}


void
CrfSgd::test(const dataset_t &data, const char *conlleval, Timer *tm)
{
   if (dict.nOutputs() <= 0)
    {
      cerr << "ERROR (test): "
           << "Must call load() or initialize() before test()." << endl;
      exit(10);
    }
   opstream f;
   string evalcommand;
   if (conlleval && conlleval[0] && verbose)
     f.open(conlleval);
   if (verbose)
     cout << "  sentences: " << data.size();
   double obj = 0;
   int errors = 0;
   int total = 0;
   for (unsigned int i=0; i<data.size(); i++)
     {
       Scorer scorer(data[i], dict, ww);
       obj += scorer.scoreForward() - scorer.scoreCorrect();
       if (conlleval && conlleval[0] && verbose)
         errors += scorer.test(f);
       else if (conlleval)
         errors += scorer.test(cout);
       else
         errors += scorer.test();
       total += data[i].size();
     }
   if (verbose)
     cout << "  loss: " << obj;
   double wnorm = dot(ww.w,ww.w);
   obj += 0.5 * wnorm * lambda * data.size();
   double misrate = (double)(errors*100)/(total ? total : 1);
   misrate = (int)(misrate*100)/100.0;
   if (verbose)
     cout << "  obj*n: " << obj 
          << "  misses: " << errors 
          << "(" << misrate << "%)";
   if (tm && verbose)
     cout << "  total time: " << tm->elapsed() << " seconds";
   if (verbose)
     cout << endl;
}





// ============================================================
// Main


string modelFile;
string templateFile;
string trainFile;
string testFile;

const char *conlleval = "./conlleval -q";

double c = 1;
int cutoff = 3;
int epochs = 50;
int cepochs = 5;
bool tag = false;

dataset_t train;
dataset_t test;


void 
usage()
{
  cerr 
    << "Usage (training): "
    << "crfsgd [options] model template traindata [devdata]" << endl
    << "Usage (tagging):  "
    << "crfsgd -t model testdata" << endl
    << "Options for training:" << endl
    << " -c <num> : capacity control parameter (1.0)" << endl
    << " -f <num> : threshold on the occurences of each feature (3)" << endl
    << " -r <num> : total number of epochs (50)" << endl
    << " -h <num> : epochs between each testing phase (5)" << endl
    << " -e <cmd> : performance evaluation command (conlleval -q)" << endl
    << " -q       : silent mode" << endl;
  exit(10);
}

void 
parseCmdLine(int argc, char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *s = argv[i];
      if (s[0]=='-')
        {
          while (s[0] == '-')
            s++;
          if (tag || s[1])
            usage();
          if (s[0] == 't')
            {
              if (i == 1)
                tag = true;
              else
                usage();
            }
          else if (s[0] == 'q')
            verbose = false;
          else if (++i >= argc)
            usage();
          else if (s[0 ] == 'c')
            {
              c = atof(argv[i]);
              if (c <= 0)
                {
                  cerr << "ERROR: "
                       << "Illegal C value: " << c << endl;
                  exit(10);
                }
            }
          else if (s[0] == 'f')
            {
              cutoff = atoi(argv[i]);
              if (cutoff <= 0 || cutoff > 1000)
                {
                  cerr << "ERROR: " 
                       << "Illegal cutoff value: " << cutoff << endl;
                  exit(10);
                }
            }
          else if (s[0] == 'r')
            {
              epochs = atoi(argv[i]);
              if (epochs <= 0)
                {
                   cerr << "ERROR: " 
                        << "Illegal number of epochs: " << epochs << endl;
                   exit(10);
                }
            }
          else if (s[0] == 'h')
            {
              cepochs = atoi(argv[i]);
              if (cepochs <= 0)
                {
                   cerr << "ERROR: " 
                        << "Illegal number of epochs: " << cepochs << endl;
                   exit(10);
                }
            }
          else if (s[0] == 'e')
            {
              conlleval = argv[i];
              if (!conlleval[0])
                conlleval = 0;
            }
          else
            {
                  cerr << "ERROR: " 
                       << "Unrecognized option: " << argv[i-1] << endl;
                  exit(10);
            }
        }
      else if (tag)
        {
          if (modelFile.empty())
            modelFile = argv[i];
          else if (testFile.empty())
            testFile = argv[i];
          else 
            usage();
        }
      else
        {
          if (modelFile.empty())
            modelFile = argv[i];
          else if (templateFile.empty())
            templateFile = argv[i];
          else if (trainFile.empty())
            trainFile = argv[i];
          else if (testFile.empty())
            testFile = argv[i];
          else 
            usage();
        }
    }
  if (tag)
    {
      verbose = false;
      if (modelFile.empty() || 
          testFile.empty())
        usage();
    }
  else
    {
      if (modelFile.empty() || 
          templateFile.empty() ||
          trainFile.empty())
        usage();
    }
}



int 
main(int argc, char **argv)
{
  // parse args
  parseCmdLine(argc, argv);
  // initialize crf
  CrfSgd crf;
  if (tag) 
    {
      igzstream f(modelFile.c_str()); 
      f >> crf;
      loadSentences(testFile.c_str(), crf.getDict(), test);
      // tagging
      crf.test(test, "");
    } 
  else 
    {
      crf.initialize(templateFile.c_str(), trainFile.c_str(), c, cutoff);
      loadSentences(trainFile.c_str(), crf.getDict(), train);
      if (! testFile.empty())
        loadSentences(testFile.c_str(), crf.getDict(), test);
      // training
      Timer tm;
      tm.start();
      crf.adjustEta(train, 1000, 0.1);
      tm.stop();
      if (verbose)
        cout << "Initial eta=" << crf.getEta()  << " t0=" << crf.getT0()
             << "  total time: " << tm.elapsed() << " seconds" << endl;
      while (crf.getEpoch() < epochs)
        {
          tm.start();
          crf.train(train, cepochs, &tm);
          tm.stop();
          if (verbose)
            {
              cout << "Training perf:";
              crf.test(train, conlleval);
            }
          if (verbose && test.size())
            {
              cout << "Testing perf:";
              crf.test(test, conlleval);
            }
        }
      if (verbose)
        cout << "Saving model file " << modelFile << "." << endl;
      ogzstream f(modelFile.c_str()); 
      f << crf; 
      if (verbose)
        cout << "Done!  " << tm.elapsed() << " seconds." << endl;
    }
  return 0;
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

// -*- C++ -*-
// CRF with stochastic gradient

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

#ifdef __GNUC__
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
#ifdef EXACT_EXPONENTIAL
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
  return s.size() / expected;
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
  
  void initFromData(const char *tFile, const char *dFile, int cutoff=1);

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


void
Dictionary::initFromData(const char *tFile, const char *dFile, int cutoff)
{
  // clear all
  templates.clear();
  outputs.clear();
  features.clear();
  index = 0;
  
  // read templates
  cerr << "Reading template file " << tFile << "." << endl;
  readTemplateFile(tFile, templates);
  int nu = 0;
  int nb = 0;
  for (unsigned int t=0; t<templates.size(); t++)
    if (templates[t][0]=='U')
      nu += 1;
    else if (templates[t][0]=='B')
      nb += 1;
  cerr << "  u-templates: " << nu 
       << "  b-templates: " << nb << endl;
  if (nu + nb != (int)templates.size())
    {
      cerr << "ERROR (building dictionary): "
           << "Problem counting templates" << endl;
      exit(10);
    }
  
  // process compressed datafile
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
  cerr << "  sentences: " << sentences 
       << "  outputs: " << oindex << endl;
  
  // allocating parameters
  strings_t keys;
  for (hash_t::iterator hi = fcount.begin(); hi != fcount.end(); hi++)
    if (hi->second >= cutoff)
      keys.push_back(hi->first);
  sort(keys.begin(), keys.end());
  for (strings_t::iterator si = keys.begin(); si != keys.end(); si++)
    {
      string k = *si;
      int size = (k[0] == 'B') ? oindex * oindex : oindex;
      features[k] = index;
      index += size;
    }
  cerr << "  cutoff: " << cutoff 
       << "  features: " << features.size() 
       << "  parameters: " << index << endl
       << "  duration: " << timer.elapsed() << " seconds." << endl;
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
  cerr << "Reading and preprocessing " << fname << "." << endl;
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
  cerr << "  processed: " << sentences << " sentences." << endl
       << "  duration: " << timer.elapsed() << " seconds." << endl;
}




// ============================================================
// Scorer


class Scorer
{
public:
  Sentence s;
  const Dictionary &d;
  VFloat *w;
  double &wscale;
  
  Scorer(const Sentence &s, const Dictionary &d, FVector &w, double &c);
  virtual ~Scorer() {}
  
  virtual void uScores(int pos, int fy, int ny, VFloat *scores);
  virtual void bScores(int pos, int fy, int ny, int y, VFloat *scores);
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny) {}
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y) {}

  double viterbi(ints_t &path);
  double test(ostream &f);
  double scoreCorrect();
  void gradCorrect(double g);
  double scoreForward();
  void gradForward(double g);
};


Scorer::Scorer(const Sentence &s, const Dictionary &d, FVector &w, double &c)
  : s(s), d(d), w(w), wscale(c) 
{
  assert(w.size() == d.nParams());
}


void
Scorer::uScores(int pos, int fy, int ny, VFloat *c)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = fy;
  SVector x = s.u(pos);
  for (int j=0; j<ny; j++)
    c[j] = 0;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      c[j] += w[p->i + off + j] * p->v;
  for (int j=0; j<ny; j++)
    c[j] *= wscale;
}


void
Scorer::bScores(int pos, int fy, int ny, int y, VFloat *c)
{
  int n = d.nOutputs();
  assert(pos>=0 && pos<s.size());
  assert(y>=0 && y<n);
  assert(fy>=0 && fy<n);
  assert(fy+ny>0 && fy+ny<=n);
  int off = y * n + fy;
  SVector x = s.b(pos);
  for (int j=0; j<ny; j++)
    c[j] = 0;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      c[j] += w[p->i + off + j] * p->v;
  for (int j=0; j<ny; j++)
    c[j] *= wscale;
}



double 
Scorer::viterbi(ints_t &path)
{
  int npos = s.size();
  int nout = d.nOutputs();
  int pos, i, j;
  
  // allocate backpointer array
  vector<ints_t> pointers(npos);
  for (int i=0; i<npos; i++)
    pointers[i].resize(nout);
  
  // process scores
  FVector scores(nout);
  uScores(0, 0, nout, scores);
  for (pos=1; pos<npos; pos++)
    {
      FVector us(nout);
      FVector bs(nout);
      uScores(pos, 0, nout, us);
      for (i=0; i<nout; i++)
        {
          bScores(pos-1, 0, nout, i, bs);
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


double
Scorer::test(ostream &f)
{
  ints_t path;
  double score = viterbi(path);
  int npos = s.size();
  int ncol = s.columns();
  for (int pos=0; pos<npos; pos++)
    {
      for (int c=0; c<ncol; c++)
        f << s.data(pos,c) << " ";
      f << d.outputString(path[pos]) << endl;
    }
  f << endl;
  return score;
}


double 
Scorer::scoreCorrect()
{
  int npos = s.size();
  int y = s.y(0);
  VFloat vf;
  uScores(0, y, 1, &vf);
  double sum = vf;
  for (int pos=1; pos<npos; pos++)
    {
      int fy = y;
      y = s.y(pos);
      bScores(pos-1, fy, 1, y, &vf);
      sum += vf;
      uScores(pos, y, 1, &vf);
      sum += vf;
    }
  return sum;
}


void
Scorer::gradCorrect(double g)
{
  int npos = s.size();
  int y = s.y(0);
  VFloat vf = g;
  uGradients(&vf, 0, y, 1);
  for (int pos=1; pos<npos; pos++)
    {
      int fy = y;
      y = s.y(pos);
      bGradients(&vf, pos-1, fy, 1, y);
      uGradients(&vf, pos, y, 1);
    }
}


double 
Scorer::scoreForward()
{
  int npos = s.size();
  int nout = d.nOutputs();
  int pos, i;
  
  FVector scores(nout);
  uScores(0, 0, nout, scores);
  for (pos=1; pos<npos; pos++)
    {
      FVector us(nout);
      FVector bs(nout);
      uScores(pos, 0, nout, us);
      for (i=0; i<nout; i++)
        {
          bScores(pos-1, 0, nout, i, bs);
          bs.add(scores);
          us[i] += logSum(bs);
        }
      scores = us;
    }
  return logSum(scores);
}



void 
Scorer::gradForward(double g)
{
  int npos = s.size();
  int nout = d.nOutputs();
  int pos, i;
  // forward pass
  FMatrix scores(npos, nout);
  uScores(0, 0, nout, scores[0]);
  for (pos=1; pos<npos; pos++)
    {
      FVector &us = scores[pos];
      FVector bs(nout);
      uScores(pos, 0, nout, us);
      for (i=0; i<nout; i++)
        {
          bScores(pos-1, 0, nout, i, bs);
          bs.add(scores[pos-1]);
          us[i] += logSum(bs);
        }
    }
  // backward pass
  FVector tmp(nout);
  FVector grads(nout);
  dLogSum(g, scores[npos-1], grads);
  for (pos=npos-1; pos>0; pos--)
    {
      FVector bs(nout);
      FVector ug(nout);
      uGradients(grads, pos, 0, nout);
      for (int i=0; i<nout; i++)
        if (grads[i])
          { // recomputing bScores is not efficient
            // when there are many B templates.
            bScores(pos-1, 0, nout, i, bs);
            bs.add(scores[pos-1]);
            dLogSum(grads[i], bs, tmp);
            bGradients(tmp, pos-1, 0, nout, i);
            ug.add(tmp);
          }
      grads = ug;
    }
  uGradients(grads, 0, 0, nout);
}



// ============================================================
// GScorer - compute gradients as SVectors


class GScorer : public Scorer
{
private:
  SVector grad;
public:
  GScorer(const Sentence &s,const Dictionary &d,FVector &w,double &c);
  void clear() { grad.clear(); }
  SVector gradient() { return grad; }
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny);
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y);
};


GScorer::GScorer(const Sentence &s,const Dictionary &d,FVector &w,double &c)
  : Scorer(s,d,w,c)
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
  TScorer(const Sentence &s, const Dictionary &d,
          FVector &w, double &c, double eta);
  virtual void uGradients(const VFloat *g, int pos, int fy, int ny);
  virtual void bGradients(const VFloat *g, int pos, int fy, int ny, int y);
};


TScorer::TScorer(const Sentence &s, const Dictionary &d,
                 FVector &w, double &c, double eta )
  : Scorer(s,d,w,c), eta(eta)
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
  double gain = eta / wscale;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      w[p->i + off + j] += g[j] * p->v * gain;
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
  SVector a;
  double gain = eta / wscale;
  for (const SVector::Pair *p = x; p->i>=0; p++)
    for (int j=0; j<ny; j++)
      w[p->i + off + j] += g[j] * p->v * gain;
}



// ============================================================
// Main function



string templateFile = "template";
string trainFile = "../data/conll2000/train.txt.gz";
string testFile = "../data/conll2000/test.txt.gz";

//string trainFile = "small.gz";
//string testFile = "small.gz";

int cutoff = 3;
dataset_t train;
dataset_t test;


int 
main(int argc, char **argv)
{
  Dictionary dict;
  dict.initFromData(templateFile.c_str(), trainFile.c_str(), cutoff);

  loadSentences(trainFile.c_str(), dict, train);
  loadSentences(testFile.c_str(), dict, test);

  double wscale = 1;
  FVector w(dict.nParams());

#if 0
  for (int i=0; i<10; i++)
    {
      GScorer scorer(train[i], dict, w, wscale);
      cout << "[" << i << "]" << "  size: " << train[i].size()
           << "  forward: " << scorer.scoreForward()
           << "  correct: " << scorer.scoreCorrect() 
           << endl;
      
      scorer.gradCorrect(+1);
      scorer.gradForward(-1);
      SVector grad = scorer.gradient();
      cout << " gradient norm: " << dot(grad,grad) 
           << " pairs: " << grad.npairs() << endl;
    }
#endif

  double eta0 = 0.1;
  double C = 4;
  double lambda = 1 / (C * train.size());
  double t = 1 / (lambda * eta0);

  Timer tm;
  for (int epoch=0; epoch<40; epoch++)
    {
      tm.start();
      for (unsigned int i=0; i<train.size(); i++)
        {
          double eta = 1/(lambda*t);
          TScorer scorer(train[i], dict, w, wscale, eta);
          scorer.gradCorrect(+1);
          scorer.gradForward(-1);
          wscale *= (1 - eta*lambda);
          t += 1;
        }
      tm.stop();
      double wnorm = dot(w,w)*wscale*wscale;
      cerr << "[" << epoch << "] ---------------------" << endl
           << " Total training time: " << tm.elapsed() << " seconds." << endl
           << " Norm: " << wnorm
           << " WScale: " << wscale << endl;

      if (epoch%5==4)
      {
        double obj = 0.5*wnorm*lambda*train.size();
        opstream f("./conlleval -q");    
        for (unsigned int i=0; i<train.size(); i++)
          {
            Scorer scorer(train[i], dict, w, wscale);
            scorer.test(f);
            obj += scorer.scoreForward() - scorer.scoreCorrect();
          }
        cout << "Training perf:  obj=" << obj << endl;
      }
      if (epoch%5==4)
      {
        cout << "Testing perf: " << endl;
        opstream f("./conlleval -q");    
        for (unsigned int i=0; i<test.size(); i++)
          {
            Scorer scorer(test[i], dict, w, wscale);
            scorer.test(f);
          }
      }

      {
        ofstream f("model");
        f << dict << endl;
        f << wscale << endl;
        f << w;
      }
    }
  return 0;
}


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

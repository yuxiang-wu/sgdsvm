 

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
#include "pstream.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
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




// ============================================================
// Strings_t, Sentence_t


typedef vector<string> strings_t;
typedef vector<strings_t> sentence_t;


ostream& 
operator<<(ostream &f, const strings_t &s)
{
  const char *sep = "";
  for (unsigned int i=0; i<s.size(); i++)
    {
      f << sep << s[i];
      sep = " ";
    }
  f << endl;
  return f;
}


ostream&
operator<<(ostream &f, const sentence_t &s)
{
  for (unsigned int i=0; i<s.size(); i++)
    f << s[i];
  f << endl;
  return f;
}



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



// ============================================================
// Parsing data file


int
readDataLine(istream &f, strings_t &line, int &expected)
{
  line.clear();
  while (f.good())
    {
      int c = skipBlank(f);
      if (c == '\n' || c == '\r')
        break;
      string s;
      f >> s;
      if (! s.empty())
        line.push_back(s);
    }
  int c = f.get();
  if (c == '\r' && f.get() != '\n')
    f.unget();
  int obtained = line.size();
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
readDataSentence(istream &f, sentence_t &s, int &expected)
{
  strings_t line;
  s.clear();
  while (f.good())
    if (readDataLine(f, line, expected))
      break;
  if (line.size())
    s.push_back(line);
  while (f.good() && readDataLine(f, line, expected))
    s.push_back(line);
  return s.size();
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
expandTemplate(string tpl, const sentence_t &s, unsigned int pos)
{
  string e;
  int rows = s.size();
  int columns = s[0].size();
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
                e.append(s[a][b]);
              else if (a<0 && -a-1<4)
                e.append(BOS[-a-1]);
              else if (a>=rows && a-rows<4)
                e.append(EOS[a-rows]);
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
  int index;

public:
  Dictionary() : index(0) { }

  int nOutputs() const { return outputs.size(); }
  int nFeatures() const { return features.size(); }
  int nTemplates() const { return templates.size(); }
  int nParams() const { return index; }
  
  int output(string s) const { 
    dict_t::const_iterator it = outputs.find(s);
    return (it == outputs.end()) ? -1 : it->second;
  }
  
  int feature(string s) const { 
    dict_t::const_iterator it = features.find(s);
    return (it == features.end()) ? -1 : it->second;
  }
  
  string templat(int i) const {
    return templates.at(i);
  }
  
  void initFromData(const char *tFile, const char *dFile, int cutoff=1);
  
  friend istream& operator>> ( istream &f, Dictionary &d );
  friend ostream& operator<< ( ostream &f, const Dictionary &d );
};


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
  cerr << "Reading " << tFile << "." << endl;
  readTemplateFile(tFile, templates);
  int nu = 0;
  int nb = 0;
  for (unsigned int t=0; t<templates.size(); t++)
    if (templates[t][0]=='U')
      nu += 1;
    else if (templates[t][0]=='B')
      nb += 1;
  cerr << "  U templates: " << nu << endl
       << "  B templates: " << nb << endl;
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
  sentence_t s;
  igzstream f(dFile);
  while (readDataSentence(f, s, columns))
    {
      sentences += 1;
      for (unsigned int p=0; p<s.size(); p++)
        {
          // check output keyword
          string &y = s[p][columns-1];
          dict_t::iterator di = outputs.find(y);
          if (di == outputs.end())
            outputs[y] = oindex++;
          // expand templates
          for (unsigned int t=0; t<templates.size(); t++)
            {
              string x = expandTemplate(templates[t], s, p);
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

  // allocating parameters
  cerr << "  sentences: " << sentences << endl
       << "  outputs: " << oindex << endl;
  for (hash_t::iterator hi = fcount.begin(); hi != fcount.end(); hi++)
    {
      if (hi->second >= cutoff)
        {
          int size = oindex;
          if (hi->first[0] == 'B')
            size = oindex * oindex;
          features[hi->first] = index;
          index += size;
        }
    }
  cerr << "  cutoff: " << cutoff << endl
       << "  features: " << features.size() << endl
       << "  parameters: " << index << endl;
}




// ============================================================
// Main function


int 
main(int argc, char **argv)
{
  Dictionary dict;
  dict.initFromData("template", "../data/conll2000/train.txt.gz", 3);

  { ofstream f("model"); f << dict; }

  Dictionary d2;

  { ifstream f("model"); f >> d2; }

  cerr << "  templates: " << d2.nTemplates() << endl
       << "  outputs: " << d2.nOutputs() << endl
       << "  features: " << d2.nFeatures() << endl
       << "  parameters: " << d2.nParams() << endl;

  return 0;
}


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */

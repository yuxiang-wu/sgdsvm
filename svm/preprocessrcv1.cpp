// -*- C++ -*-
// SVM with stochastic gradient (preprocessing)
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
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdlib>
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
    inline size_t operator()(const string &s) const { return h(s.c_str()); };
  };
};
#else
# define hash_map map
#endif


typedef hash_map<int,bool> classes_t;
classes_t classes;

void
readClasses(const char *fname)
{
  cerr << "# Reading " << fname << endl;

  igzstream f;
  f.open(fname);
  if (! f.good()) {
    cerr << "ERROR: cannot open file " << fname << endl;
    ::exit(10);
  }

  classes.clear();
  for(;;) {
    string topic;
    int id, rev;
    f >> topic >> id >> rev;
    if (! f.good())
      break;
    if (topic == "CCAT")
      classes[id] = true;
    else if (classes.find(id) == classes.end())
      classes[id] = false;
  }
  if (!f.eof()) {
    cerr << "ERROR: failed reading " << fname << endl;
    ::exit(10);
  }
  
  int pcount = 0;
  int ncount = 0;
  for (classes_t::const_iterator it=classes.begin(); it!=classes.end(); it++)
    if (it->second)
      pcount++;
    else
      ncount++;

  cerr << "# Done reading " 
       << pcount << " positives and " 
       << ncount << " negatives. " << endl;
}



typedef hash_map<string, int> dico_t;
dico_t dico;

typedef hash_map<int, SVector> docs_t;
docs_t train;
docs_t test;


void 
readDocs(const char *fname, docs_t &docs, bool freezedico=false)
{
  cerr << "# Reading " << fname << endl;

  igzstream f;
  f.open(fname);
  if (! f.good()) {
    cerr << "ERROR: cannot open file " << fname << endl;
    ::exit(10);
  }
  
  string token;
  f >> token;
  if (token != ".I")
    {
      cerr << "ERROR: Cannot read initial .I in " << fname << endl;
      ::exit(10);
    }
  int id = 0;
  int count = 0;
  while(f.good())
    {
      f >> id >> token;
      count += 1;
      if (! f.good() || token != ".W")
        {
          cerr << "ERROR (" << id << "): "
               << "Cannot read \"<id> .W\"." << endl;
          ::exit(10);
        }
      int wid = -1;
      string otoken;
      SVector s;
      for(;;)
        {
          f >> token;
          if (!f.good() || token == ".I")
            break;
          if (token != otoken)
            {
              dico_t::iterator it = dico.find(token);
              if (it != dico.end())
                wid = it->second;
              else if (freezedico)
                continue;
              else
                {
                  wid = dico.size() + 1;
                  dico[token] = wid;
                }
              otoken = token;
            }
          s.set(wid, s.get(wid)+1.0);
        }
      if (s.npairs() <= 0)
        {
          cerr << "ERROR (" << id << "): "
               << "Empty vector " << id << "?" << endl;
          ::exit(10);
        }
      docs[id] = s;
    }
  if (!f.eof())
    {
      cerr << "ERROR (" << id << "): "
           << "Failed reading words" << endl;
      ::exit(10);
    }

  cerr << "# Done reading " << count << " documents." << endl;
}


typedef vector<int> intvector_t;
intvector_t trainid;
intvector_t testid;

void
listKeys(docs_t &docs, intvector_t &ivec, bool shuffle=false)
{
  ivec.clear();
  for (docs_t::iterator it = docs.begin(); it != docs.end(); it++)
    ivec.push_back(it->first);
  if (shuffle)
    random_shuffle(ivec.begin(), ivec.end());
}



void 
computeNormalizedTfIdf()
{
  cerr << "# Computing document frequencies" << endl;

  int terms = dico.size();
  vector<double> nt(terms+1);
  
  double nd = trainid.size();
  for(int i=0; i<terms+1; i++)
    nt[i] = 0;
  for(int i=0; i<(int)trainid.size(); i++)
    {
      int id = trainid[i];
      SVector s = train[id];
      for (const SVector::Pair *p = s; p->i >= 0; p++)
        if (p->v > 0)
          nt[p->i] += 1;
    }
  
  cerr << "# Computing TF/IDF for training set" << endl;
  for(int i=0; i<(int)trainid.size(); i++)
    {
      int id = trainid[i];
      SVector s = train[id];
      SVector v;
      for (const SVector::Pair *p = s; p->i >= 0; p++)
        if (nt[p->i] > 0)
          v.set(p->i, (1.0 + log(p->v)) * log(nd/nt[p->i]));
      double norm = dot(v,v);
      v.scale(1.0 / sqrt(norm));
      train[id] = v;
    }
  cerr << "# Computing TF/IDF for testing set" << endl;
  for(int i=0; i<(int)testid.size(); i++)
    {
      int id = testid[i];
      SVector s = test[id];
      SVector v;
      for (const SVector::Pair *p = s; p->i >= 0; p++)
        if (nt[p->i] > 0)
          v.set(p->i, (1.0 + log(p->v)) * log(nd/nt[p->i]));
      double norm = dot(v,v);
      v.scale(1.0 / sqrt(norm));
      test[id] = v;
    }
  cerr << "# Done." << endl;
}




void
saveBinary(const char *fname, docs_t &docs, intvector_t &ids)
{
  cerr << "# Writing " << fname << "."  << endl;
  
  ogzstream f;
  f.open(fname);
  if (! f.good())
    {
      cerr << "ERROR: cannot open " << fname << " for writing." << endl;
      ::exit(10);
    }
  
  int pcount = 0;
  int ncount = 0;
  int npairs = 0;
  for(int i=0; i<(int)ids.size(); i++)
    {
      int id = ids[i];
      bool y = classes[id];
      if (y)
        pcount += 1;
      else
        ncount += 1;

      SVector s = docs[id];
      int p = s.npairs();
      npairs += p;
      if (p <= 0)
        {
          cerr << "ERROR: empty vector " << id << "." << endl;
          ::exit(10);
        }
      
      f.put( y ? 1 : 0);
      s.save(f);
      if (! f.good())
        {
          cerr << "ERROR: writing " << fname << " for writing." << endl;
          ::exit(10);
        }
    }

  cerr << "# Done. Wrote " << ids.size() << " examples." << endl;
  cerr << "#   with " << npairs << " pairs, " 
       << pcount << " positives, and "
       << ncount << " negatives." << endl;
}


void
saveSvmLight(const char *fname, docs_t &docs, intvector_t &ids)
{
  cerr << "# Writing " << fname << "."  << endl;
  
  ogzstream f;
  f.open(fname);
  if (! f.good())
    {
      cerr << "ERROR: cannot open " << fname << " for writing." << endl;
      ::exit(10);
    }
  
  for(int i=0; i<(int)ids.size(); i++)
    {
      int id = ids[i];
      bool y = classes[id];
      SVector s = docs[id];
      int p = s.npairs();
      if (p <= 0)
        {
          cerr << "ERROR: empty vector " << id << "." << endl;
          ::exit(10);
        }
      f << ((y) ? +1 : -1);
      f << s;
      if (! f.good())
        {
          cerr << "ERROR: writing " << fname << " for writing." << endl;
          ::exit(10);
        }
    }
  
  cerr << "# Done. Wrote " << ids.size() << " examples." << endl;
}




int 
main(int, const char**)
{
  readClasses("../data/rcv1/rcv1-v2.topics.qrels.gz");

  readDocs("../data/rcv1/lyrl2004_tokens_train.dat.gz", test);
  cerr << "# Dictionary size (so far) " << dico.size() << endl;

  // We freeze the dictionary at this point.
  // As a result we only use features common to both the training and testing set.
  // This is consistent with joachims svmperf experiments.
  readDocs("../data/rcv1/lyrl2004_tokens_test_pt0.dat.gz", train, true);
  readDocs("../data/rcv1/lyrl2004_tokens_test_pt1.dat.gz", train, true);
  readDocs("../data/rcv1/lyrl2004_tokens_test_pt2.dat.gz", train, true);
  readDocs("../data/rcv1/lyrl2004_tokens_test_pt3.dat.gz", train, true);
  
  cerr << "# Got " << test.size() << " testing documents." << endl;
  cerr << "# Got " << train.size() << " training documents." << endl;
  cerr << "# Dictionary size " << dico.size() << endl;
  
  listKeys(train, trainid, true);
  listKeys(test, testid);
  computeNormalizedTfIdf();

  saveBinary("train.bin.gz", train, trainid);
  saveSvmLight("train.dat.gz", train, trainid);
  saveBinary("test.bin.gz", test, testid);
  saveSvmLight("test.dat.gz", test, testid);
  
  cerr << "# The End." << endl;
}

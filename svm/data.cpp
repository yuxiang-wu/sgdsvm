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


#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gzstream.h"
#include "assert.h"
#include "data.h"

using namespace std;




// Compat

int load_datafile(const char *filename, 
                  xvec_t &xp, yvec_t &yp, int &maxd,
                  bool norm, int maxn)
{
  Loader loader(filename);
  int maxdim = 0;
  int pcount, ncount;
  loader.load(xp, yp, norm, maxn, &maxdim, &pcount, &ncount);
  if (pcount + ncount > 0)
    cout << "# Read " << pcount << "+" << ncount 
         << "=" << pcount+ncount << " examples " 
         << "from \"" << filename << "\"." << endl;
  return pcount + ncount;
}



// Loader

struct Loader::Private
{
  string filename;
  bool compressed;
  bool binary;
  igzstream gs;
  ifstream fs;
};

Loader::~Loader()
{
  delete p;
}

Loader::Loader(const char *name)
  : p(new Private)
{
  p->filename = name;
  p->compressed = p->binary = false;
  int len = p->filename.size();
  if (len > 7 && p->filename.substr(len-7) == ".txt.gz")
    p->compressed = true;
  else if (len > 7 && p->filename.substr(len-7) == ".bin.gz")
    p->compressed = p->binary = true;
  else if (len > 4 && p->filename.substr(len-4) == ".bin")
    p->binary = true;
  else if (len > 4 && p->filename.substr(len-4) == ".txt")
    p->binary = false;
  else
    assertfail("Filename suffix should be one of: "
               << ".bin, .txt, .bin.gz, .txt.gz");
  if (p->compressed)
    p->gs.open(name);
  else
    p->fs.open(name);
  if (! (p->compressed ? p->gs.good() : p->fs.good()))
    assertfail("Cannot open " << p->filename);
}


int Loader::load(xvec_t &xp, yvec_t &yp, bool normalize, int maxrows,
                 int *p_maxdim, int *p_pcount, int *p_ncount)
{
  istream &f = (p->compressed) ? (istream&)(p->gs) : (istream&)(p->fs);
  bool binary = p->binary;
  int ncount = 0;
  int pcount = 0;
  while (f.good() && maxrows--)
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
          f >> std::skipws >> y >> std::ws;
          if (f.peek() == '|') f.get();
          f >> x;
        }
      if (f.good())
        {
          if (normalize)
            {
              double d = dot(x,x);
              if (d > 0 && d != 1.0)
                x.scale(1.0 / sqrt(d)); 
            }
          if (y != +1 && y != -1)
            assertfail("Label should be +1 or -1.");
          xp.push_back(x);
          yp.push_back(y);
          if (y > 0)
            pcount += 1;
          else
            ncount += 1;
          if (p_maxdim && x.size() > *p_maxdim)
            *p_maxdim = x.size();
        }
    }
  if (!f.eof() && !f.good())
    assertfail("Cannot read data from " << p->filename);
  if (p_pcount)
    *p_pcount = pcount;
  if (p_ncount)
    *p_ncount = ncount;
  return pcount + ncount;
}





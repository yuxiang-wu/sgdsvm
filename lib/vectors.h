// -*- C++ -*-
// Little library of vectors and sparse vectors

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
#include <cstring>
#include <cassert>
#include <cstdio>


class FVector;
class SVector;


class FVector
{
private:
  struct Rep
  {
    int refcount;
    int size;
    float *data;
    Rep() : size(0), data(0) {}
    ~Rep() { delete data; }
    void resize(int n);
    Rep *copy();
  };
  
  Wrapper<Rep> w;
  Rep *rep() { return w.rep(); }
  
public:
  FVector();
  FVector(int n);
  FVector(SVector v);
  int size() const { return rep()->size; }
  float get(int i) const;
  double set(int i, float v);
  operator const float* () const { return rep()->data; }

  void add(double c1);
  void add(FVector v2);
  void add(SVector v2);
  void add(FVector v2, double c2);
  void add(SVector v2, double c2);
  void add(FVector v2, FVector c2);
  void add(SVector v2, FVector c2);
  void scale(double c1);
  void combine(double c1, FVector v2, double c2);
  void combine(double c1, SVector v2, double c2);

  void save(FILE *f) const;
  void load(FILE *f);
};



class SVector
{
public:
  struct Pair 
  { 
    int i; 
    float v; 
  };
private:
  struct Rep
  {
    int refcount;
    int npairs;
    int mpairs;
    int size;
    struct Pair *pairs;
    
    Rep() : npairs(0), mpairs(0), size(0) {}
    ~Rep() { delete data; }
    void resize(int n);
    void addpair();
    Rep *copy();
  };
  
  Wrapper<Rep> w;
  Rep *rep() { return w.rep(); }
  
public:
  SVector();
  SVector(FVector v);
  int size() const { return rep()->size; }
  float get(int i) const;
  double set(int i, float v);
  int npairs() const { return rep()->npairs; }
  operator const Pair* () const { return rep->pairs; }

  void add(SVector v2);
  void add(SVector v2, double c2);
  void scale(double c1);
  void combine(double c1, SVector v2, double c2);

  void save(FILE *f) const;
  void load(FILE *f);
};

double dot(FVector v1, FVector v2);
double dot(FVector v1, SVector v2);
double dot(SVector v1, FVector v2);
double dot(SVector v1, SVector v2);

SVector combine(SVector v1, double a1, SVector v2, double a2);
FVector combine(FVector v1, double a1, SVector v2, double a2);
FVector combine(SVector v1, double a1, FVector v2, double a2);
FVector combine(FVector v1, double a1, FVector v2, double a2);





/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

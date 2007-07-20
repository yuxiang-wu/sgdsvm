

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


#include "vectors.h"
#include <cassert>


namespace
{
  template <typename T> inline T min(T a, T b) {
    return (a < b) ? a  : b;
  }
  template <typename T> inline T max(T a, T b) {
    return (a < b) ? b  : a;
  }
}



void 
FVector::Rep::resize(int n)
{
  if (size != n)
    {
      if (n > 0)
        {
          float *newdata = new float[n];
          int i = 0;
          int s = min(i,n);
          for (; i<s; i++)
            newdata[i] = data[i];
          for (; i<n; i++)
            newdata[i] = 0;
          data = newdata;
          size = n;
        }
      else
        {
          delete [] data;
          data = 0;
          size = 0;
        }
    }
}


FVector::Rep *
FVector::Rep::copy()
{
  Rep *newrep = new Rep;
  newrep->resize(size);
  for (int i=0; i<size; i++)
    newrep->data[i] = data[i];
  return newrep;
}


FVector::FVector()
{
}


FVector::FVector(int n)
{
  Rep *r = rep();
  r->resize(n);
}


FVector::FVector(const SVector &v)
{
  Rep *r = rep();
  r->resize(v.size());
  int npairs = v.npairs();
  const SVector::Pair *pairs = v;
  for (int i=0; i<npairs; i++, pairs++)
    r->data[pairs->i] = pairs->v;
}


double
FVector::get(int i) const
{
  const Rep *r = rep();
  if (i >=0 && i<r->size)
    return r->data[i];
  assert(i>=0);
  return 0;
}


double 
FVector::set(int i, double v)
{
  w.detach();
  Rep *r = rep();
  if (i > r->size)
    r->resize(i+1);
  assert(i>=0);
  r->data[i] = v;
  return v;
}


void 
FVector::clear()
{
  w.detach();
  rep()->resize(0);
}


void 
FVector::resize(int n)
{
  w.detach();
  assert(n >= 0);
  rep()->resize(n);
}


void 
FVector::add(double c1)
{
  w.detach();
  Rep *r = rep();
  float *d = r->data;
  for (int i=0; i<r->size; i++)
    d[i] += c1;
}


void 
FVector::add(const FVector &v2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *s = (const float*) v2;
  for (int i=0; i<m; i++)
    d[i] += s[i];
}


void 
FVector::add(const SVector &v2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  int npairs = v2.npairs();
  const SVector::Pair *pairs = v2;
  for (int i=0; i<npairs; i++, pairs++)
    d[pairs->i] += pairs->v;
}


void 
FVector::add(const FVector &v2, double c2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *s = (const float*) v2;
  for (int i=0; i<m; i++)
    d[i] += s[i] * c2;
}


void 
FVector::add(const SVector &v2, double c2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  int npairs = v2.npairs();
  const SVector::Pair *pairs = v2;
  for (int i=0; i<npairs; i++, pairs++)
    d[pairs->i] += pairs->v * c2;
}


void 
FVector::add(const FVector &v2, const FVector &c2)
{
  w.detach();
  Rep *r = rep();
  int m = r->size;
  m = max(m, v2.size());
  m = min(m, c2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *s = (const float*) v2;
  const float *c = (const float*) c2;
  for (int i=0; i<m; i++)
    d[i] += s[i] * c[i];
}


void 
FVector::add(const SVector &v2, const FVector &c2)
{
  w.detach();
  Rep *r = rep();
  int m = r->size;
  m = max(m, v2.size());
  m = min(m, c2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *c = (const float*) c2;
  int npairs = v2.npairs();
  const SVector::Pair *pairs = v2;
  for (int i=0; i<npairs; i++, pairs++)
    {
      int j = pairs->i;
      if (j >= m) break;
      d[j] += pairs->v * c[j];
    }
}


void 
FVector::scale(double c1)
{
  w.detach();
  Rep *r = rep();
  float *d = r->data;
  for (int i=0; i<r->size; i++)
    d[i] *= c1;
}


void 
FVector::combine(double c1, const FVector &v2, double c2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *s = (const float*) v2;
  for (int i=0; i<m; i++)
    d[i] = d[i] * c1 + s[i] * c2;
}


void 
FVector::combine(double c1, const SVector &v2, double c2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  int npairs = v2.npairs();
  const SVector::Pair *pairs = v2;
  for (int i=0; i<npairs; i++, pairs++)
    {
      int j = pairs->i;
      d[j] = d[j] * c1 + pairs->v * c2;
    }
}


bool 
FVector::save(FILE *f) const
{
  SVector s(*this);
  return s.save(f);
}


bool 
FVector::load(FILE *f)
{
  SVector s;
  if (! s.load(f))
    return false;
  FVector v(s);
  operator=(v);
  return true;
}


bool 
FVector::bsave(FILE *f) const
{
  int i = size();
  const float *d = rep()->data;
  if (::fwrite(&i, sizeof(int), 1, f) != (size_t)1)
    return false;
  if (::fwrite(d, sizeof(float), i, f) != (size_t)i)
    return false;
  return true;
}


bool
FVector::bload(FILE *f)
{
  int i;
  w.detach();
  if (::fread(&i, sizeof(int), 1, f) != (size_t)1)
    return false;
  resize(i);
  float *d = rep()->data;
  if (::fread(d, sizeof(float), i, f) == (size_t)i)
    return true;
  clear();
  return false;
}




// ----------------------------------------
#if 0

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
    ~Rep() { delete [] data; }
    void resize(int n);
    void addpair();
    Rep *copy();
  };
  
  Wrapper<Rep> w;
  Rep *rep() { return w.rep(); }
  void qset(int i, double v);
  
public:
  SVector();
  SVector(const FVector &v);
  int size() const { return rep()->size; }
  float get(int i) const;
  double set(int i, float v);
  int npairs() const { return rep()->npairs; }
  operator const Pair* () const { return rep->pairs; }

  void clear();

  void add(const SVector &v2);
  void add(const SVector &v2, double c2);
  void scale(double c1);
  void combine(double c1, const SVector &v2, double c2);

  bool save(FILE *f) const;
  bool load(FILE *f);
  bool bsave(FILE *f) const;
  bool bload(FILE *f);
};

double dot(const FVector &v1, const FVector &v2);
double dot(const FVector &v1, const SVector &v2);
double dot(const SVector &v1, const FVector &v2);
double dot(const SVector &v1, const SVector &v2);

SVector combine(const SVector &v1, double a1, const SVector &v2, double a2);
FVector combine(const FVector &v1, double a1, const SVector &v2, double a2);
FVector combine(const SVector &v1, double a1, const FVector &v2, double a2);
FVector combine(const FVector &v1, double a1, const FVector &v2, double a2);


#endif

/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

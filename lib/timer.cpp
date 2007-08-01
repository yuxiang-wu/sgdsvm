// -*- C++ -*-
// A simple timer.

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



#include "timer.h"


#ifdef WIN32
# include <windows.h>
#else
# include <time.h>
# include <sys/time.h>
#endif


#ifdef WIN32
# include <windows.h>

static double 
klock()
{
#if 0 // this does not work when laptops change frequency.
  static LARGE_INTEGER f;
  LARGE_INTEGER n;
  QueryPerformanceCounter(&n);
  if (! f.QuadPart)
    QueryPerformanceFrequency(&f);
  return (double)n.QuadPart / (double)f.QuadPart;
#else
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  return ((double)ft.dwHighDateTime*4294967296.0 + ft.dwLowDateTime) * 1e-7;
#endif
}

#else
# include <time.h>
# include <sys/time.h>
# include <unistd.h>

static double
klock()
{
  struct timeval tv;
#if _POSIX_TIMERS
  struct timespec ts;
# ifdef CLOCK_REALTIME_HR // wishful thinking
  if (::clock_gettime(CLOCK_REALTIME_HR, &ts) >= 0)
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
# endif
  if (::clock_gettime(CLOCK_REALTIME, &ts) >= 0)
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
  if (::gettimeofday(&tv, NULL) >= 0)
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
  return (double)::clock() / CLOCKS_PER_SEC;
}

#endif



Timer::Timer()
  : a(0), s(0), r(0)
{
}

void 
Timer::reset()
{
  a = 0;
  s = 0;
  r = 0;
}


double 
Timer::elapsed()
{
  double n = klock();
  if (r)
    a += n - s;
  s = n;
  return a;
}

double 
Timer::start()
{
  elapsed();
  r = 1;
  return a;
}



double
Timer::stop()
{
  elapsed();
  r = 0;
  return a;
}





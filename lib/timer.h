
// -*- C++ -*-
// A simple timer (copied from boost)

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


#ifndef TIMER_H
#define TIMER_H 1

#include <ctime>


class Timer
{
 public:
  Timer() 
    { start = std::clock(); }
  void   restart() 
    { start = std::clock(); }
  double elapsed() const 
    { double(std::clock() - start) / CLOCKS_PER_SEC; }
  double resolution const 
    { return double(1)/double(CLOCKS_PER_SEC); }
  
 private:
  std::clock_t start;
};


/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" "std::\\sw+_t")
   End:
   ------------------------------------------------------------- */


#endif

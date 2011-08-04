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

#ifndef LOSS_H
#define LOSS_H



struct LogLoss
{
  static double loss(double z)
  {
    if (z > 18) 
      return exp(-z);
    if (z < -18)
      return -z;
    return log(1 - exp(-z));
  }
  static double dloss(double z)
  {
    if (z > 18) 
      return exp(-z);
    if (z < -18)
      return 1;
    return 1 / (1 + exp(z));
  }
};


struct HingeLoss
{
  static double loss(double z)
  {
    if (z > 1) 
      return 0;
    return 1 - z;
  }
  static double dloss(double z)
  {
    if (z > 1) 
      return 0;
    return 1;
  }
};


struct SquaredHingeLoss
{
  static double loss(double z)
  {
    if (z > 1) 
      return 0;
    double d = 1 - z;
    return 0.5 * d * d;
  }
  static double dloss(double z)
  {
    if (z > 1) 
      return 0;
    return 1 - z;
  }
};


struct SmoothHingeLoss
{
  static double loss(double z)
  {
    if (z > 1)
      return 0;
    if (z < 0)
      return 0.5 - z;
    double d = 1 - z;
    return 0.5 * d * d;
  }
  static double dloss(double z)
  {
    if (z > 1) 
      return 0;
    if (z < 0)
      return 1;
    return 1 - z;
  }
};





#endif

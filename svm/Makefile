# SVM with stochastic gradient

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA


L=../lib


CXX=g++
OPTS=-g -O2
CXXFLAGS= ${OPTS} -Wall -I$L
LIBS = -lz -lm

PROGRAMS = preprocessrcv1 

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS} 2>/dev/null
	-rm *.o 2>/dev/null

vectors.o: $L/vectors.cpp $L/vectors.h  $L/wrapper.h
	${CXX} ${CXXFLAGS} -c -o $@ $L/vectors.cpp

gzstream.o: $L/gzstream.cpp $L/gzstream.h 
	${CXX} ${CXXFLAGS} -c -o $@ $L/gzstream.cpp

timer.o: $L/timer.cpp $L/timer.h 
	${CXX} ${CXXFLAGS} -c -o $@ $L/timer.cpp

preprocessrcv1: preprocessrcv1.o vectors.o gzstream.o
	${CXX} ${CXXFLAGS} -o $@ preprocessrcv1.o vectors.o gzstream.o ${LIBS}

#svmsgd: svmsgd.o vectors.o gzstream.o timer.o
#	${CXX} ${CXXFLAGS} -o $@ svmsgd.o vectors.o gzstream.o timer.o ${LIBS}







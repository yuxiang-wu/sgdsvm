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
VPATH=$L
CXX=g++
OPT=
OPTS=-g -O2
CXXFLAGS= ${OPTS} ${OPT} -Wall -I$L
LIBS = -lz -lm

PROGRAMS = preprocessrcv1 svmsgd

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS} 2>/dev/null
	-rm *.o 2>/dev/null

vectors.o: vectors.cpp vectors.h wrapper.h assert.h
gzstream.o: gzstream.cpp gzstream.h 
timer.o: timer.cpp timer.h 
data.o: vectors.h gzstream.h
preprocessrcv1.o: vectors.h gzstream.h assert.h
svmsgd.o: data.h loss.h vectors.h timer.h assert.h

preprocessrcv1: preprocessrcv1.o vectors.o gzstream.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LIBS}

svmsgd: svmsgd.o data.o vectors.o gzstream.o timer.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LIBS}





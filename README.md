This Software is called LyapXoolV3 and its distribution and modifications
are in accordance with GNU GENERAL PUBLIC LICENSE.

This file is offered as-is, without any warranty.

LyapXoolV3 is an efford carried on to keep updating and maintaining previous versions of LyapXoolV1-V2, that were published to compute Complete Lyapunov functions to describe dynamical systems.


                        Installing LyapXoolV3 
                        =====================

Installation should be simple, please follow the general instructions and optimise at will:

0. Install Armadillo Library 10 or older. 
   
1. Install GSL library 

2. Install the QP library (Quadratic program solver, in C) writen and maintained by Patrick O. Perry.
 
3. If you want to run parallel computation of various routines, please do use openmp.

4. Once you have compiled and installed all these libraries, enjoy LyapXoolV3 

      Compile with the next command
	
       g++ -O3 *.cpp -m64 -march=native -larmadillo -lgsl
       /PATH/TO/YOUR/LIBRARY_FOLDER/libqp.a  -DARMA_64BIT_WORD -fopenmp 

A makefile is included in this distribution, the file is called: makeLyapXoolV3,
more detail on compilation is given there as well as direct path to the 
QP github library.

In case of trouble or doubts, please send a description of the problem to
<cargaezg@iingen.unam.mx>.
You are welcome to contact the develouper in English, Spanish, Italian or Yucatec Mayan.

##############################################################################

Please do consider the following:

Known problems:
===============

When LD_LIBRARY_PATH is set to various paths, the compiled libraries may be mixed 
and the compiler might get confused when compiling LyapXoolV3.
Please, do make sure you are setting your LD_LIBRARY_PATH correctly and pointing
to the correct libraries.

##############################################################################

Note for Mac OSX users:
=======================
This code has been develoup in Linux, which like OSX is a Unix based operating systems. The code has been tested on OSX and works. In case of any doubt, or error in its execution or compilation, please do contact the develouper at
<cargaezg@iingen.unam.mx>.




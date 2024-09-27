/* LyapXool – V3: Quadratic optimisations, is a program to compute Complete Lyapunov functions for dynamical systems described by non linear autonomous ordinary differential equations.

 -> This is a free software; you can redistribute it and/or
 -> modify it under the terms of the GNU General Public License
 -> as published by the Free Software Foundation; either version 3
 -> of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 -> but WITHOUT ANY WARRANTY; without even the implied warranty of
 -> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 -> GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 -> along with this program.  If not, see <http://www.gnu.org/licenses/>.
 Author and main maintainer: Carlos Argáez
 -> Bibliography attached to the corresponding publication.
 */

#ifndef generalities_hpp
#define generalities_hpp
#include <stdio.h>
#include "instructions.hpp"


void printinformation();

void printall(const std::string nombre, int ordernum, arma::mat &vectoraimprimmir);

void printall(const std::string nombre, int ordernum, arma::cube &vectoraimprimmir);

void printhour(const int &definition);

std::string printhour();

void iteration(int i);

void finalization();
#endif /* generalities_hpp */

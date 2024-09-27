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


#ifndef instructions_hpp
#define instructions_hpp


#include <stdio.h>
#include <iostream>
#include <armadillo>

namespace glovar {
extern unsigned long long int functionodecalls;
extern std::ofstream outputf;

enum choose_the_method {
    minusone, op, fpop
};

enum choose_the_calculation {
    only_directional, directional_and_cartesian, chain_recurrent_set_eigenvalues, norm_chain_recurrent_set, only_cartesian, check_dim, check_numerical_approximation
};

char const probnames[][110]={"only_directional","directional_and_cartesian","chain_recurrent_set_eigenvalues", "norm_chain_recurrent_set", "only_cartesian", "check_dim", "check_numerical_approximation"};

char const metnames[][110]={"minusone","op", "fpop"};

/* How do you want to solve your CLF?*/
const choose_the_method method_type=op;

const choose_the_calculation computation_type=only_cartesian;
/*%%%% SECTION TO DEFINE PROBLEM TO ANALYZE %%%%*/

const int ode_dimension=2;

const double min_geometric_limits[ode_dimension]={-1.6,-1.6};

const double max_geometric_limits[ode_dimension]={1.6,1.6};

const double alpha=0.04;

/*%%%% SECTION TO DEFINE CONDITIONS %%%%*/

const bool normal=true;    /*true FOR THE NORMALISED METHOD*/

const bool eigenvaluesjudge=false;

const bool printing=true;

const double critval=-3.0e-7;

const int points_directional=10; /* AMOUNT OF POINTS PER DIRECTION ON THE DIRECTIONAL GRID*/

const double radius=0.49;

/*%%%% SECTION TO DEFINE AMOUNT OF ITERATIONS %%%%*/

const int totaliterations=3;

/*%%%% SECTION TO DEFINE WENDLAND FUNCTION %%%%*/

const int l=3;
const int k=1;
const double c=1.0;

/*COMPUTE CONDITION NUMBER FOR THE COLLOCATION MATRIX*/
const bool condnumber=false;

/* For the cartisian evaluation grid */  
const double cart_grid_scaling=0.005;


/* For the fixed-point method */
const int num_fixed_points=1;
const double fix_points_fpop[num_fixed_points][ode_dimension]={{0.1846,0.0}};


const int OMP_NUM_THREADS=12;
};

#endif /* instructions_hpp */

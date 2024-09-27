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

#ifndef RBF_hpp
#define RBF_hpp

#include <stdio.h>
#include <armadillo>
#include "instructions.hpp"
#include "wendland.hpp"

class RBFMETHOD{
public:
    RBFMETHOD(double alpha, int points_directional, double radius, int dimension, double cart_grid_density, const double *min_geometric_limits, const double *max_geometric_limits, bool normal, bool cond_number, arma::mat fix_points, bool printing, std::ofstream &outputf);
    
    void wbase(bool talaakal);
    
    void add_fix_point_to_colgrid();

    void alphafunction();
    
    void interpolationmatrixA(WENDLAND &wendland);
    
    void choldecom();

    void direcgrid();

    void makeRBF(WENDLAND &wendland);

    void makeRBFS(WENDLAND &wendland);

    void makecolgrid(bool talaakal);

    void makeevalblpn(bool talaakal);
    
    void checkcond();
     
    void checkdim();

    void checkcapability(double v1, std::string filename);
    
    void checkcapability(double v1);

    double alpha;
    int dimension;
    int points_directional;
    double radius;
    bool printing;
    int alphasize;
    double cart_grid_density;
    const double *min_geometric_limits;
    const double *max_geometric_limits;
    bool normal;
    bool condnumber;
    arma::mat *wdlfunction;
    arma::mat *wdlf1;
    arma::mat *wdlf2;
    arma::mat *wdlf3;
    std::ofstream* outputf;
    
    arma::mat rbfbasis;
    arma::mat collocationpoints;
    arma::vec alphavector;
    arma::mat Amat;
    arma::mat R;
    arma::mat directgrid;
    arma::mat coldirectgrid;
    std::vector<bool> booldirectgrid;
    std::vector<bool> boolcoldirectgrid;
    arma::mat cartesianevalgrid;
    arma::rowvec ek;
    arma::mat fix_points;
    int stride;
};
#endif /* RBF_hpp */

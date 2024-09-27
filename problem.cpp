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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <armadillo>
#include <list>
#include "odesystem.hpp"
#include "instructions.hpp"
#include "wendland.hpp"
#include "RBF.hpp"
#include "generalities.hpp"
#include "lyapunovfunction.hpp"
#include "odetools.hpp"

using namespace std;
using namespace arma;

unsigned long long int glovar::functionodecalls=0;
std::ofstream glovar::outputf;
arma::span const All=span::all;

int main()
{
    system("pwd");
    cout << "esta corriendo" << endl;
    printhour(2);

    cout << printhour() << endl;
    wall_clock timer,timer1;
    timer.tic();
    glovar::outputf.open("output.lpx", fstream::out);
    printinformation();
    glovar::outputf  << setfill(' ') << setw(51) << " " <<  " LyapXool-V3 " << setfill(' ') << setw(51) <<  " " << endl;
    glovar::outputf  << setfill(' ') << setw(47) << " " <<  printhour() << setw(51) <<  " " << endl;
    glovar::outputf  << "\n" << endl;

    glovar::outputf  << setfill('=') << setw(52) << "=" <<  " BEGINNING " << setfill('=') << setw(52) <<  "=" << endl;
    glovar::outputf  << setfill(' ') << endl;

    glovar::outputf << "Do make sure that sufficient computer's memory is available after the matrices are loaded in." << endl;
    glovar::outputf << "If your computations fail due to a memory problem, then do perform a memory size check with the function check_dim." << endl;

    if(glovar::eigenvaluesjudge)
    {
        mat criticalpoints;
        criticalpoints.set_size(1,1);
        if((int)criticalpoints.n_cols!=glovar::ode_dimension)
        {
            glovar::outputf<<"The program has stopped because your critical values' matrix has a different dimension to: " << glovar::ode_dimension << endl;
        }
        criticalpoints<<0.0 << endr;
        crit_point_eigen_pairs(criticalpoints);
    }


    mat fix_points(glovar::num_fixed_points,glovar::ode_dimension);
    for(int i=0; i<glovar::num_fixed_points; ++i)
    {
        for(int j=0; j<glovar::ode_dimension; ++j)
        {
            fix_points(i,j) = glovar::fix_points_fpop[i][j];
        }
    }

    WENDLAND wendland(glovar::l,glovar::k,glovar::c,glovar::outputf);

    wendland.wendlandfunction();
    wendland.wendlandderivative(wendland.wdlfunction,wendland.wdlf1);
    wendland.wendlandderivative(wendland.wdlf1,wendland.wdlf2);
    wendland.wendlandderivative(wendland.wdlf2,wendland.wdlf3);

    
    RBFMETHOD rbf(glovar::alpha, glovar::points_directional, glovar::radius, glovar::ode_dimension, glovar::cart_grid_scaling, glovar::min_geometric_limits, glovar::max_geometric_limits, glovar::normal, glovar::condnumber, fix_points, glovar::printing, glovar::outputf);

    LYAPUNOV lyapunov(glovar::totaliterations, glovar::ode_dimension, glovar::cart_grid_scaling, glovar::l, glovar::k, glovar::c, glovar::points_directional, glovar::critval, glovar::normal, glovar::printing, glovar::outputf);

    printhour(0);

    switch (glovar::computation_type){
    case glovar::choose_the_calculation::only_directional:
        rbf.makeRBF(wendland);
        lyapunov.make_lyap_direcional(wendland, rbf);
        break;
    case glovar::choose_the_calculation::directional_and_cartesian:
        rbf.makeRBF(wendland);
        lyapunov.make_lyap_direc_and_cart(wendland, rbf);
        break;
    case glovar::choose_the_calculation::chain_recurrent_set_eigenvalues:
        rbf.makeRBF(wendland);
        lyapunov.make_chainrecurrent_eigenvalues(wendland, rbf);
        break;
    case glovar::choose_the_calculation::norm_chain_recurrent_set:
        rbf.makeRBF(wendland);
        lyapunov.make_norm_chain_recurrent_sets(wendland, rbf);
        break;
    case glovar::choose_the_calculation::only_cartesian:
            if(glovar::totaliterations>1)
            {
                rbf.makeRBF(wendland);
            }else{
                rbf.makeRBFS(wendland);
            }
        lyapunov.make_lyap_cartesian(wendland, rbf);
        break;
    case glovar::check_dim:
        rbf.checkdim();
        break;
    case glovar::check_numerical_approximation:
        rbf.makeRBFS(wendland);
        lyapunov.make_check_num_approximation(wendland, rbf);
        break;
    default:
        glovar::outputf << "COMPUTATION: THIS HAS NOT BEEN ASSIGNED" <<endl;
        exit(9);
    }

    finalization();

    glovar::outputf
    << left
    << setw(20)
    << "The whole procedure lasted "
    << left
    << setw(5)
    << timer.toc()
    << left
    << setw(5)
    << "s."
    << endl;

    printhour(1);
    glovar::outputf  << "\n" << endl;
    glovar::outputf  << setfill(' ') << setw(51) << " " <<  " LyapXool-V3 " << setfill(' ') << setw(51) <<  " " << endl;
    glovar::outputf  << setfill(' ') << setw(45) << " " <<  " cargaezg@iingen.unam.mx " << setfill(' ') << setw(51) <<  " " << endl;
    glovar::outputf.close();
    
    return 0;
}





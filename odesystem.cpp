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

#include <armadillo>

#include "odesystem.hpp"
#include "instructions.hpp"
#include "generalities.hpp"

using namespace arma;


void odesystem(const bool normal, rowvec const &x, rowvec &f)
{
    f(0)=-1.0*x(0)*((x(0)*x(0))+(x(1)*x(1))-(1.0/4.0))*((x(0)*x(0))+(x(1)*x(1))-1.0)-x(1);
    f(1)=-1.0*x(1)*((x(0)*x(0))+(x(1)*x(1))-(1.0/4.0))*((x(0)*x(0))+(x(1)*x(1))-1.0)+x(0);
    
/*
 *
    double temp = x(0)*x(0)*x(1);
    f(0)= 0.1 - x(0) + temp;
    f(1)= 0.7 - temp;

    double vm=2.0;
    double ks=1.0;
    double m=0.1;
    double rm=1.5;
    double l=1.0;
    double g=0.2;
    double gam=0.3;
    double P=x(0);
    double Z=x(1);
    double N=x(2);

    f(0)=(vm*N*P/(ks+N))-m*P-Z*rm*(1.0-exp(-l*P));
    f(1)=gam*Z*rm*(1.0-exp(-l*P))-g*Z;
    f(2)=-(vm*N*P/(ks+N))+m*P+g*Z+(1.0-gam)*Z*rm*(1.0-exp(-l*P));


    f(0)=-1.0*x(0)*((x(0)*x(0))+(x(1)*x(1))-(1.0/4.0))*((x(0)*x(0))+(x(1)*x(1))-1.0)-x(1);
    f(1)=-1.0*x(1)*((x(0)*x(0))+(x(1)*x(1))-(1.0/4.0))*((x(0)*x(0))+(x(1)*x(1))-1.0)+x(0);
 */
   /*
    double a=6.8927;
    double ro=0.3224;
    double del=0.00028058;
    double c=2.3952;
    double k=0.4032;
    double s=0.0010691;

    double ex=x(0);
    double y=x(1);
    double z=x(2);
    f(0)=ro*del*(ex*ex-a*ex)+s*ex*(ex+y+c-c*tanh(ex+z));
    f(1)=-ro*del*(a*y+ex*ex);
    f(2)=del*(k-z-0.5*ex);
*/

//DO NOT MODIFY BELOW THIS LINE.
    if(normal)
    {
        f/=arma::norm(f);
    }
    
    glovar::functionodecalls+=1;
}


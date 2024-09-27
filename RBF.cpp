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

#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <armadillo>
#include <fstream>
#include <list>

#include <string>
#include "RBF.hpp"
#include "instructions.hpp"
#include "odesystem.hpp"
#include "wendland.hpp"
#include <numeric>
#include "generalities.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace arma;
using namespace std;
arma::span const All=span::all;

RBFMETHOD::RBFMETHOD(double alpha, int points_directional, double radius, int dimension, double cart_grid_density, const double *min_geometric_limits, const double *max_geometric_limits, bool normal, bool condnumber, mat fix_points, bool printing, ofstream &outputf){
    this->alpha=alpha;
    this->points_directional=points_directional;
    this->radius=radius;
    this->dimension=dimension;
    this->cart_grid_density=cart_grid_density;
    this->min_geometric_limits=min_geometric_limits;
    this->max_geometric_limits=max_geometric_limits;
    this->normal=normal;
    this->condnumber=condnumber;
    this->printing=printing;
    this->outputf=&outputf;
    this->wdlfunction=0;
    this->wdlf1=0;
    this->wdlf2=0;
    this->wdlf3=0;
    this->fix_points=fix_points;
}
void RBFMETHOD::wbase(bool talaakal)
{
    wall_clock timer;
    timer.tic();
    rbfbasis.set_size(dimension,dimension);
    
    rbfbasis.zeros();
    ek.set_size(dimension);
    ek.zeros();
    for(int k=1; k<=dimension; ++k)
    {
        
        ek(k-1)=sqrt(1.0/(2*k*(k+1)));
        rbfbasis(k-1,span::all)=ek;
        
        rbfbasis(k-1,All)(k-1)=(k+1)*ek(k-1);
    }
    if(talaakal)
    {
        if(printing)
        {
            printall("rbfbasis",0,rbfbasis);
        }
        
        glovar::outputf
        << left
        << setw(65)
        << "Computing the basis lasted "
        << right
        << setw(15)
        << setprecision(9)
        << std::fixed
        << timer.toc()
        << left
        << setw(5)
        << "s"
        << left
        << setw(5)
        << printhour()
        << endl;
    }
}

void RBFMETHOD::add_fix_point_to_colgrid()
{
    int totalmatrixlength=(int)collocationpoints.n_rows+(int)fix_points.n_rows;
    int totalwidthlength=(int)collocationpoints.n_cols;
    int totaloldlength=(int)collocationpoints.n_rows;
    int firstlength=(int)fix_points.n_rows;
    
    
    mat inputintec(totalmatrixlength,totalwidthlength);
    inputintec(span(0,firstlength-1),All)=fix_points(All,All);
    inputintec(span(firstlength,firstlength+totaloldlength-1),All)=collocationpoints(All,All);
    
    collocationpoints.resize(totalmatrixlength,totalwidthlength);
    collocationpoints=inputintec(All,All);
}

void RBFMETHOD::alphafunction()
{
    wall_clock timer;
    timer.tic();
    alphasize=(int)collocationpoints.n_rows;
    double totalweight=(double)sizeof(double)*alphasize/1073741824.0;

    checkcapability(totalweight, "alphafunction");

    alphavector.set_size(alphasize);
    alphavector.fill(-1.0);
    if(printing)
    {
        printall("alphavector",  0, alphavector);
    }
    
    glovar::outputf
    << left
    << setw(65)
    << "The whole procedure to obtain the alpha vector lasted "
    << right
    << setw(15)
    << setprecision(9)
    << std::fixed
    << timer.toc()
    << left
    << setw(5)
    << "s"
    << left

    << setw(5)
    << printhour()
    << endl;
}



void RBFMETHOD::interpolationmatrixA(WENDLAND &wendland)
{
    wall_clock timer;
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    int j=0, k=0;
    
    *outputf  << " " << endl;
    if(normal)
    {
        *outputf  << "Computing Interpolation Matrix with the normalisation approach" << endl;
    }else{
        *outputf  << "Computing Interpolation Matrix without the normalisation approach" << endl;
    }
    
    double atzero;
    
    int dimA=(int)collocationpoints.n_rows;
    int dimAc=(int)collocationpoints.n_cols;
    
    *outputf
    << left
    << setw(69)
    << "The length of the matrix is "
    << left
    << setw(5)
    << dimA
    << endl;

    double totalweight=(double)sizeof(double)*dimA*dimA/1073741824.0;
    checkcapability(totalweight, "interpolationmatrixA");

    Amat.set_size(dimA,dimA);
    Amat.zeros();
    
    
    atzero=wendland.evawdlfn(0.0,*wdlf1);
    
    
    int chunk = int(floor(dimA/glovar::OMP_NUM_THREADS));
    
    timer.tic();
    
#pragma omp parallel shared(Amat,collocationpoints,wdlf1,wdlf2,atzero,normal,chunk) private(j,k)
    {
        rowvec diffsave(dimAc);
        rowvec savingcallj(dimAc);
        rowvec savingcallk(dimAc);
        rowvec resultj(dimAc);
        rowvec resultk(dimAc);
        
        diffsave.zeros();
        savingcallj.zeros();
        savingcallk.zeros();
        resultj.zeros();
        resultk.zeros();
        
        double twopointsdistance=0.0;
        
        double wdlfvalue1=0.0;
        double wdlfvalue2=0.0;
        double checking=0.0;
        
        
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk) nowait
        for(j=0; j<dimA; ++j)
        {
            savingcallj=collocationpoints(j, All);
            odesystem(normal,savingcallj,resultj);
            for(k=0; k<dimA; ++k)
            {
                savingcallk=collocationpoints(k, All);
                diffsave=savingcallj-savingcallk;
                if(k==j){
                    Amat(j,k)=-atzero*dot(resultj,resultj);
                }else{
                    twopointsdistance=sqrt(dot(diffsave,diffsave));
                    checking=1.0-glovar::c*twopointsdistance;
                    if(checking>0.0)
                    {
                        odesystem(normal,savingcallk,resultk);
                        wdlfvalue1=wendland.evawdlfn(twopointsdistance,*wdlf1);
                        wdlfvalue2=wendland.evawdlfn(twopointsdistance,*wdlf2);
                        Amat(j,k)=-wdlfvalue2*dot(diffsave,resultj)*dot(diffsave,resultk)-wdlfvalue1*dot(resultj,resultk);
                    }
                }
            }
        }
    }
    
    *outputf
    << left
    << setw(66)
    << "Computing the Interpolation matrix lasted "
    << right
    << setw(15)
    << setprecision(9)
    << std::fixed
    << timer.toc()
    << left
    << setw(5)
    << "s"
    << left

    << setw(5)
    << printhour()
    << endl;
}



void RBFMETHOD::choldecom()
{
    wall_clock timer;
    timer.tic();
    int maxite=(int)Amat.n_rows;
    R.set_size(maxite,maxite);
    try {
    R = chol(Amat);
    } catch (const std::exception& balam){
         *outputf << "Something has happened while computing the Cholesky decomposition." << endl;
         *outputf << __FILE__ << endl;
         *outputf << __LINE__ << endl;

    }


    
    *outputf
    << left
    << setw(65)
    << "The whole Cholesky decomposition lasted "
    << right
    << setw(15)
    << setprecision(9)
    << std::fixed
    << timer.toc()
    << left
    << setw(5)
    << "s"
    << left

    << setw(5)
    << printhour()
    << endl;
}


void RBFMETHOD::direcgrid()
{
    wall_clock timer;
    timer.tic();
    int lcols=(int)collocationpoints.n_cols;
    int lrows=(int)collocationpoints.n_rows;
    
    
    int newlenght=(int)(points_directional*2*lrows);
    
    
    int j,jd;
    stride=points_directional*2+1;
    
    double totalweight=(double)sizeof(double)*lrows*stride*lcols/1073741824.0;

    checkcapability(totalweight, "direcgrid");

    double norm;
    coldirectgrid.set_size(lrows*stride,lcols);
    directgrid.set_size(newlenght,lcols);

    
    mat domain(newlenght,lcols);
    rowvec savingdomain(lcols), evaldfunction(lcols);
    {
        for(int i=0; i<lrows; ++i)
        {
            j=stride*i;
            jd=(stride-1)*i;
            coldirectgrid(j,All)=collocationpoints(i,All);
            savingdomain=collocationpoints(i,All);
            
            odesystem(normal, savingdomain, evaldfunction);
            norm=sqrt(dot(evaldfunction,evaldfunction));
            int kp=0;
            for(int kd=0; kd<points_directional; kd+=1)
            {
                directgrid(jd+kp,All)=collocationpoints(i,All)+(radius/points_directional)*(kd+1)*alpha*(evaldfunction/norm);
                directgrid(jd+kp+1,All)=collocationpoints(i,All)-(radius/points_directional)*(kd+1)*alpha*(evaldfunction/norm);
                coldirectgrid(j+kp+1,All)=directgrid(jd+kp,All);
                coldirectgrid(j+kp+2,All)=directgrid(jd+kp+1,All);
                kp+=2;
                
            }
        }
    }

    list<int> counter,counterf;
    {
        int cdrows=(int)coldirectgrid.n_rows;
        int drows=(int)directgrid.n_rows;


        for(int i=0; i<cdrows; ++i)
        {
            for(int jc=0; jc<dimension; ++jc)
            {
                if((coldirectgrid(i,jc)<=max_geometric_limits[jc]) && (coldirectgrid(i,jc)>=min_geometric_limits[jc]))
                {
                   try {
                        counter.push_back(i);
                    } catch (const std::exception& pixan){
                        *outputf << "An exception has occured in computing the directional grid." << pixan.what() << endl;
                        *outputf << __FILE__ << endl;
                        *outputf << __LINE__ << endl;

                    }
                }

            }
        }

        for(int ii=0; ii<drows; ++ii)
        {
            for(int jc=0; jc<dimension; ++jc)
            {
                if((directgrid(ii,jc)<=max_geometric_limits[jc]) && (directgrid(ii,jc)>=min_geometric_limits[jc]))
                {
                    try {
                         counterf.push_back(ii);
                     } catch (const std::exception& pixan){
                        *outputf << "An exception has occured in computing the directional grid." << pixan.what() << endl;
                        *outputf << __FILE__ << endl;
                        *outputf << __LINE__ << endl;
                     }
                }
            }
        }
    }
    
    boolcoldirectgrid.resize(stride*lrows,false);
    booldirectgrid.resize(lrows*(stride-1),false);
    //cleanbigag.set_size(ana,lcols);
    //cleanbigfg.set_size(flo,lcols);
    
    int n=0;
    int m=0;
    

        for(list<int>::iterator i=counter.begin(); i!=counter.end(); ++i)
        {
       try {
             boolcoldirectgrid[*i]=true;
            //cleanbigag(n,All)=coldirectgrid(*i,All);
            n++;
            } catch (const std::exception& pixan){
                *outputf << "An exception has occured in computing the directional grid." << pixan.what() << endl;
                *outputf << __FILE__ << endl;
                *outputf << __LINE__ << endl;
            }
        }
    
    for(list<int>::iterator ii=counterf.begin(); ii!=counterf.end(); ++ii)
    {
       try {
            booldirectgrid[*ii]=true;
        //cleanbigfg(m,All)=directgrid(*ii,All);
        m++;
        } catch (const std::exception& pixan){
            *outputf << "An exception has occured in computing the directional grid." << pixan.what() << endl;
            *outputf << __FILE__ << endl;
            *outputf << __LINE__ << endl;
        }
    }

    counter.clear();
    counterf.clear();

    if(printing)
    {
        timer.tic();
        const std::string output_file_extension="m";
        int dim=(int)directgrid.n_rows;
        int dim2=(int)directgrid.n_cols;
        if((dim!=0)&&(dim2!=0))
        {
            for(int totalwidth=0; totalwidth<dim2; ++totalwidth)
            {
                ostringstream fileName;
                fileName<<"sdirecgrid"<<totalwidth<<"."<<output_file_extension;
                ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                valuesdocument << "direcgrid"<<totalwidth << "=[";
                
                for(int p=0; p<dim; ++p)
                {
                    valuesdocument << std::fixed << std::setprecision(18) << directgrid(p,totalwidth) << " " ;
                }
                valuesdocument << "];" << endl;
                valuesdocument.close();
            }
        }else{
            *outputf << "WARNING: The collocation grid does not contain values and its dimension is 0 " << endl;
        }
        
    }
    
    *outputf
    << left
    << setw(69)
    << "The cardinality of the directional grid is "
    << left
    << setw(5)
    << n
    << endl;

    *outputf
    << left
    << setw(65)
    << "The whole procedure to construct the directional grid lasted "
    << right
    << setw(15)
    << setprecision(9)
    << std::fixed
    << timer.toc()
    << left
    << setw(5)
    << "s"
    << left

    << setw(5)
    << printhour()
    << endl;
        
}

void RBFMETHOD::makecolgrid(bool talaakal)
{
    wall_clock timer;
    timer.tic();
    RBFMETHOD::wbase(talaakal);
    
    arma::rowvec  a(min_geometric_limits, glovar::ode_dimension);
    arma::rowvec  b(max_geometric_limits, glovar::ode_dimension);
    double tol = 1e-10;
    a -= tol * ones<rowvec>(dimension);
    b += tol * ones<rowvec>(dimension);
    list<rowvec> Ret;
    function<void(int, rowvec)> ML = [&](int r, rowvec x) {
        for (int i = int(ceil((a(r) - x(r)) / (alpha*(r + 2)*ek(r)))); i <= int(floor((b(r) - x(r)) / (alpha*(r + 2)*ek(r)))); i++) {
            if (r == 0) {
                Ret.push_back(x + i *alpha*rbfbasis(r,All));
            }
            else {
                ML(r - 1, x + i *alpha*rbfbasis(r,All));
            }
        }
    };
    ML(dimension - 1, 0.5*alpha*solve(rbfbasis, ones<vec>(dimension)).t());
    
    function<bool(const rowvec &)> BadColl = [&](const rowvec &x)->bool {
        rowvec f(dimension);
        
        odesystem(false, x, f);
        return norm(f) < 1e-10;
    };
    Ret.remove_if(BadColl);
    
    int NrOfPoints = (int)Ret.size();
    collocationpoints.set_size(NrOfPoints, dimension);
    auto Ri = Ret.begin();
    for (int i = 0; i < NrOfPoints; Ri++, i++) {
        collocationpoints(i, All) = *Ri;
    }
    
    Ret.clear();
    
    
    if(talaakal)
    {
        *outputf
        << left
        << setw(65)
        << "The total procedure to construct the collocation points lasted "
        << right
        << setw(15)
        << setprecision(9)
        << std::fixed
        << timer.toc()
        << left
        << setw(5)
        << "s"
        << left
        << printhour()
        << endl;
        
        if(printing)
        {
            timer.tic();
            const std::string output_file_extension="m";
            int dim=(int)collocationpoints.n_rows;
            int dim2=(int)collocationpoints.n_cols;
            if((dim!=0)&&(dim2!=0))
            {
                for(int totalwidth=0; totalwidth<dim2; ++totalwidth)
                {
                    ostringstream fileName;
                    fileName<<"scolgrid"<<totalwidth<<"."<<output_file_extension;
                    ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                    valuesdocument << "colgrid"<<totalwidth << "=[";
                    
                    for(int p=0; p<dim; ++p)
                    {
                        valuesdocument << std::fixed << std::setprecision(18) << collocationpoints(p,totalwidth) << " " ;
                    }
                    valuesdocument << "];" << endl;
                    valuesdocument.close();
                }
            }else{
                *outputf << "WARNING: The collocation grid does not contain values and its dimension is 0 " << endl;
            }
            
        }
    }
}

void RBFMETHOD::makeevalblpn(bool talaakal)
{
    wall_clock timer;
    timer.tic();
    RBFMETHOD::wbase(talaakal);
    
    arma::rowvec  a(min_geometric_limits, glovar::ode_dimension);
    arma::rowvec  b(max_geometric_limits, glovar::ode_dimension);
    double tol = 1e-10;
    a -= tol * ones<rowvec>(dimension);
    b += tol * ones<rowvec>(dimension);
    mat A;
    A.eye(dimension, dimension);
    list<rowvec> BP;
    function<void(int, rowvec)> ML = [&](int r, rowvec x) {
        for (int i = int(ceil((a(r) - x(r)) / (cart_grid_density))); i <= int(floor((b(r) - x(r)) / (cart_grid_density))); i++) {
            if (r == 0) {
                BP.push_back(x + i *cart_grid_density*A(r,All));
            }
            else {
                ML(r - 1, x + i *cart_grid_density*A(r,All));
            }
        }
    };
    
    ML(dimension - 1, cart_grid_density*solve(A, ones<vec>(dimension)).t());
    
    int n = (int)BP.size();
    
    cartesianevalgrid.set_size(n, dimension);
    auto raz31 = BP.begin();
    for (int i = 0; i < n; raz31++, i++) {
        cartesianevalgrid(i, All) = *raz31;
    }
    
    BP.clear();

    if(talaakal)
    {
        *outputf
        << left
        << setw(69)
        << "The cardinality of the Cartesian grid is "
        << left
        << setw(1)
        << n
        << endl;

        
        *outputf
        << left
        << setw(65)
        << "The construction of the Cartesian grid lasted "
        << right
        << setw(15)
        << setprecision(9)
        << std::fixed
        << timer.toc()
        << left
        << setw(5)
        << "s"
        << left
        << setw(5)
        << printhour()
        << endl;
        
        
        if(printing)
        {
            timer.tic();
            const std::string output_file_extension="m";
            int dim=(int)cartesianevalgrid.n_rows;
            int dim2=(int)cartesianevalgrid.n_cols;
            if((dim!=0)&&(dim2!=0))
            {
                for(int totalwidth=0; totalwidth<dim2; ++totalwidth)
                {
                    ostringstream fileName;
                    fileName<<"sdensevar"<<totalwidth<<"."<<output_file_extension;
                    ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                    valuesdocument << "densevar"<<totalwidth << "=[";
                    
                    for(int p=0; p<dim; ++p)
                    {
                        valuesdocument << std::fixed << std::setprecision(18) << cartesianevalgrid(p,totalwidth) << " " ;
                    }
                    valuesdocument << "];" << endl;
                    valuesdocument.close();
                }
            }else{
                *outputf << "WARNING: The evaluation cartesian grid does not contain values and its dimension is 0 " << endl;
            }
            
        }
    }
}


void RBFMETHOD::checkcond()
{
    wall_clock timer;
    timer.tic();
    double condnumber=0.0;
    condnumber=cond(Amat);
    *outputf
    << "\n"
    << left
    << setw(65)
    << "The condition number of the collocation matrix is "
    << left
    << setw(5)
    << condnumber
    << endl;
    
    *outputf
    << left
    << setw(65)
    << "Computing the condition number lasted "
    << right
    << setw(15)
    << setprecision(9)
    << std::fixed
    << timer.toc()
    << left
    << setw(5)
    << "s"
    << left

    << setw(5)
    << printhour()
    << endl;
    
}
void RBFMETHOD::checkdim()
{
    *outputf << "Checking the size of the grids\n" << endl;
    makecolgrid(false);
    makeevalblpn(false);
    
    double totalweight = (double)sizeof(double)*collocationpoints.n_rows*collocationpoints.n_rows/1073741824.0;
    double totalweightcol = (double)sizeof(double)*collocationpoints.n_rows*collocationpoints.n_cols/1073741824.0;
    double totalweighteval = (double)sizeof(double)*cartesianevalgrid.n_rows*cartesianevalgrid.n_cols/1073741824.0;
    uint64_t totalbyes = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGESIZE);
    double totalgigas = (double)totalbyes/1073741824.0;
    
    
    
    
    *outputf
    << left
    << setw(22)
    << "Array"
    << left
    << setw(5)
    << "Length (rows)"
    << left
    << setw(5)
    << " "
    << "Width (cols)"
    << endl;
    *outputf
    << left
    << setw(22)
    << "Collocation grid: "
    << left
    << setw(20)
    << collocationpoints.n_rows
    << " "
    << left
    << setw(5)
    << collocationpoints.n_cols
    << endl;
    *outputf
    << left
    << setw(22)
    << "Directional grid: "
    << left
    << setw(20)
    << collocationpoints.n_rows*2*points_directional
    << " "
    << left
    << setw(5)
    << collocationpoints.n_cols
    << endl;
    *outputf
    << left
    << setw(22)
    << "Evaluation grid: "
    << left
    << setw(20)
    << cartesianevalgrid.n_rows
    << " "
    << left
    << setw(5)
    << cartesianevalgrid.n_cols
    << endl;
    *outputf
    << left
    << setw(55)
    << setfill('=')
    << "="
    << endl;
    *outputf
    << left
    << setfill(' ')
    << endl;
    *outputf
    << left
    << setw(22)
    << "Array"
    << left
    << setw(5)
    << "Weight (GB)"
    << endl;
    *outputf
    << left
    << setw(22)
    << "Collocation matrix: "
    << left
    << setw(5)
    << totalweight
    << endl;
    *outputf
    << left
    << setw(22)
    << "Collocation grid: "
    << left
    << setw(5)
    << totalweightcol
    << endl;
    *outputf
    << left
    << setw(22)
    << "Directional grid: "
    << left
    << setw(5)
    << totalweightcol*(2*points_directional)
    << endl;
    *outputf
    << left
    << setw(22)
    << "Cartesian grid: "
    << left
    << setw(5)
    << totalweighteval
    << endl;
    *outputf
    << left
    << setw(35)
    << setfill('=')
    << "="
    << left
    << setfill(' ')
    << endl;
    *outputf
    << left
    << setw(22)
    << "Total: "
    << left
    << setw(5)
    << totalweight+totalweightcol*(2.0*points_directional+1.0)+totalweighteval
    << endl;
    *outputf
    << left
    << setw(22)
    << "\nTotal memory on this computer: "
    << left
    << setw(1)
    << totalgigas
    << "GB."
    << endl;
    
    checkcapability(totalweight);

    rbfbasis.clear();
}


void RBFMETHOD::makeRBF(WENDLAND &wendland)
{
    wendland.wendlandfunction();
    wendland.wendlandderivative(wendland.wdlfunction,wendland.wdlf1);
    wendland.wendlandderivative(wendland.wdlf1,wendland.wdlf2);
    wendland.wendlandderivative(wendland.wdlf2,wendland.wdlf3);
    wdlfunction=&wendland.wdlfunction;
    wdlf1=&wendland.wdlf1;
    wdlf2=&wendland.wdlf2;
    wdlf3=&wendland.wdlf3;
    makecolgrid(true);
    rbfbasis.clear();
    alphafunction();
    direcgrid();
    interpolationmatrixA(wendland);

    if(condnumber)
    {
        checkcond();
    }

}

void RBFMETHOD::makeRBFS(WENDLAND &wendland)
{
    wendland.wendlandfunction();
    wendland.wendlandderivative(wendland.wdlfunction,wendland.wdlf1);
    wendland.wendlandderivative(wendland.wdlf1,wendland.wdlf2);
    wendland.wendlandderivative(wendland.wdlf2,wendland.wdlf3);
    wdlfunction=&wendland.wdlfunction;
    wdlf1=&wendland.wdlf1;
    wdlf2=&wendland.wdlf2;
    wdlf3=&wendland.wdlf3;
    makecolgrid(true);
    rbfbasis.clear();
    alphafunction();
    interpolationmatrixA(wendland);
}

void RBFMETHOD::checkcapability(double v1, std::string filename)
{
    uint64_t totalbyes = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGESIZE);
    double totalgigas = (double)totalbyes/1073741824.0;

    if(v1>=0.5*totalgigas)
    {
        *outputf
        << left
        << setw(71)
        << setfill(':')
        << "-"
        << left
        << setfill(' ')
        << "-"
        << right
        << endl;
        *outputf << "The program stopped execution at the function: " << filename << ", check your parameters." << endl;
        *outputf << "Total available RAM memory:" << totalgigas << endl;
        *outputf << "Total weight of your array in the current function: " << v1 << endl;
        *outputf << "\n" << endl;
        *outputf << "The execution of this program will fail: the collocation matrix weights more or equal than the 50 percent of total RAM memory in this computer." << endl;
        *outputf << "Do make sure that sufficient computer's memory is available after the matrices are loaded in." << endl;
        *outputf << "If your computations fail, please do reduce the size of your matrices by increasing the alpha and the cart_grid_scaling parameters." << endl;
        *outputf
        << left
        << setw(71)
        << setfill(':')
        << "-"
        << left
        << setfill(' ')
        << "-"
        << right
        << endl;
        exit(0);
    }

}
    void RBFMETHOD::checkcapability(double v1)
    {
        uint64_t totalbyes = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGESIZE);
        double totalgigas = (double)totalbyes/1073741824.0;

        if(v1>=0.5*totalgigas)
        {
            *outputf
            << left
            << setw(35)
            << setfill(':')
            << "="
            << left
            << setfill(' ')
            << endl;
            *outputf << "The execution of this program will fail: the collocation matrix weights more or equal than the 50 percent of total RAM memory in this computer." << endl;
            *outputf << "Do make sure that sufficient computer's memory is available after the matrices are loaded in." << endl;
            *outputf << "If your computations fail, please do reduce the size of your matrices by increasing the alpha and the cart_grid_scaling parameters." << endl;
            *outputf
            << left
            << setw(35)
            << setfill('=')
            << "="
            << left
            << setfill(' ')
            << endl;
            exit(0);
        }


}

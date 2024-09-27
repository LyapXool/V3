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
#include <fstream>
#include <sstream>
#include <armadillo>
#include <fstream>
#include <list>

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_qp.h>

#include <string>
#include "RBF.hpp"
#include "instructions.hpp"
#include "odesystem.hpp"
#include "wendland.hpp"
#include <numeric>
#include "generalities.hpp"
#include "lyapunovfunction.hpp"














#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace arma;
using namespace std;
using namespace glovar;

arma::span const All=span::all;
LYAPUNOV::LYAPUNOV(int totaliterations, int ode_dimension, double cart_grid_density, int l, int k, double c, int points_directional, double critval, bool normal, bool printing, std::ofstream &outputf){
    this->totaliterations=totaliterations;
    this->ode_dimension=ode_dimension;
    this->cart_grid_density=cart_grid_density;
    this->l=l;
    this->k=k;
    this->c=c;
    this->points_directional=points_directional;
    this->critval=critval;
    this->normal=normal;
    this->printing=printing;
    this->outputf=&outputf;
}


void LYAPUNOV::lyapequation(int currentiteration, RBFMETHOD &rbf)
{
    wall_clock timer;
    int maxbet=(int)rbf.collocationpoints.n_rows;
    betaod.set_size(maxbet);
    
    timer.tic();

    switch (glovar::method_type){
    case glovar::choose_the_method::minusone:
        if(currentiteration==0)
        {
            rbf.choldecom();
            rbf.Amat.clear();
            try {
                betaod=solve(trimatu(rbf.R), solve(trimatl(trimatu(rbf.R).t()), rbf.alphavector),solve_opts::fast);
            }  catch (const std::exception& balam){
                *outputf << "Something has happened while solving the Lyapunov equation to get the beta vector." << endl;
                *outputf << __FILE__ << endl;
                *outputf << __LINE__ << endl;

           }
        }else{
            try {
                betaod=solve(trimatu(rbf.R), solve(trimatl(trimatu(rbf.R).t()), rbf.alphavector),solve_opts::fast);
            }  catch (const std::exception& balam){
                *outputf << "Something has happened while solving the Lyapunov equation to get the beta vector." << endl;
                *outputf << __FILE__ << endl;
                *outputf << __LINE__ << endl;
           }
        }
        break;
    case glovar::choose_the_method::op:
        quadraticoptimization(rbf);
        rbf.Amat.clear();
        break;
    case glovar::choose_the_method::fpop:
        quadratic_fixedpoint(rbf);
        rbf.Amat.clear();
        break;
    default:
        *outputf << "METHOD: THIS HAS NOT BEEN ASSIGNED" <<endl;
        exit(9);
    }




    *outputf
    << "\n"
    << endl;

    *outputf
    << left
    << setw(51)
    << "The whole procedure to solve the Lyapunov equation at iteration  "
    << left
    << setw(2)
    << currentiteration
    << left
    << setw(5)
    << "lasted "
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
        printall("betavector",  currentiteration, betaod);
    }
    
}

void LYAPUNOV::quadraticoptimization(RBFMETHOD &rbf)
{
    wall_clock timer, timeramat;
    timer.tic();
    int totalmatrixlength=(int)rbf.collocationpoints.n_rows;
    int totalwidthlength=(int)rbf.collocationpoints.n_cols;
    rbf.Amat*=2.0;
    Row <double> tentativeq(totalmatrixlength);
    tentativeq.zeros();
    for(int j=0; j<totalmatrixlength; ++j)
    {
        for(int i=0; i<totalmatrixlength; ++i)
        {
            tentativeq(i)+=rbf.Amat(i,j)*pow(rbf.alpha,totalwidthlength);
        }
    }
    tentativeq=tentativeq/totalmatrixlength;
    gsl_vector *a=gsl_vector_alloc(totalmatrixlength);
    for(int j=0; j<totalmatrixlength; ++j)
    {
        gsl_vector_set(a,j, tentativeq(j));
    }
    gsl_matrix *C=gsl_matrix_alloc(totalmatrixlength,totalmatrixlength);
    gsl_vector *b=gsl_vector_alloc(totalmatrixlength);
    gsl_vector *x=gsl_vector_alloc(totalmatrixlength);
    double f;
    gsl_matrix_view G
            = gsl_matrix_view_array(rbf.Amat.memptr(), totalmatrixlength, totalmatrixlength);
    void *work = malloc (gsl_qp_work_size (x->size));
    gsl_vector_set_zero(b);

    for(int i=0; i<totalmatrixlength; ++i)
    {
        for(int j=0; j<totalmatrixlength; ++j)
        {
            gsl_matrix_set(C,i,j, -1.0*gsl_matrix_get(&G.matrix, i, j));
        }
    }
    timeramat.tic();

    try {
         gsl_qp_solve (CblasNoTrans, CblasNoTrans, &G.matrix, a, C, b, NULL, NULL, x, &f, work);
    }  catch (const std::exception& balam){
        *outputf << "Something has happened while solving the quadratic optimisation." << endl;
        *outputf << __FILE__ << endl;
        *outputf << __LINE__ << endl;
   }

    
    
    
    *outputf
    << left
    << setw(65)
    << "Procedure to quadratically optimising "
    << right
    << setw(15)
    << setprecision(9)
    << std::fixed
    << timeramat.toc()
    << left
    << setw(5)
    << "s"
    << left
    << setw(5)
    << printhour()
    << endl;

    for(int j=0; j<totalmatrixlength; ++j)
    {
        betaod(j)=gsl_vector_get(x, j);
    }
    gsl_vector_free(a);
    gsl_matrix_free(C);
    gsl_vector_free(b);
    gsl_vector_free(x);
    free (work);
  
    *outputf
    << left
    << setw(65)
    << "Procedure to construct the input and quadratically optimised it "
    << right
    << setw(0)
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


void LYAPUNOV::quadratic_fixedpoint(RBFMETHOD &rbf)
{
    wall_clock timer, timeramat;
    int totalmatrixlength=(int)rbf.collocationpoints.n_rows;
    int firstlength=(int)rbf.fix_points.n_rows;
    int secondlength=(int)(totalmatrixlength-firstlength);

    rbf.Amat*=2.0;

    gsl_matrix_view G
            = gsl_matrix_view_array(rbf.Amat.memptr(), totalmatrixlength, totalmatrixlength);
    gsl_vector *a=gsl_vector_alloc(totalmatrixlength);
    gsl_matrix *C=gsl_matrix_alloc(secondlength,totalmatrixlength);
    gsl_vector *b=gsl_vector_alloc(secondlength);
    gsl_matrix *CE=gsl_matrix_alloc(firstlength,totalmatrixlength);
    gsl_vector *be=gsl_vector_alloc(firstlength);
    gsl_vector *x=gsl_vector_alloc(totalmatrixlength);
    double f;

    void *work = malloc(gsl_qp_work_size(x->size));

    gsl_vector_set_zero(b);
    gsl_vector_set_all(be, -1.0);
    gsl_vector_set_zero(a);

    for(int i=0; i<firstlength; ++i)
    {
        for(int j=0; j<totalmatrixlength; ++j)
        {
            gsl_matrix_set(CE,i,j, gsl_matrix_get(&G.matrix, i, j));
        }
    }

    for(int i=0; i<secondlength; ++i)
    {
        for(int j=0; j<totalmatrixlength; ++j)
        {
            gsl_matrix_set(C,i,j, -1.0*gsl_matrix_get(&G.matrix, firstlength+i, j));
        }
    }
    timeramat.tic();
    try {
         gsl_qp_solve(CblasNoTrans, CblasNoTrans, &G.matrix, a, C, b, CE, be, x, &f, work);
    }  catch (const std::exception& balam){
        *outputf << "Something has happened while solving the quadratic optimisation." << endl;
        *outputf << __FILE__ << endl;
        *outputf << __LINE__ << endl;
   }

    *outputf
    << left
    << setw(65)
    << "Procedure to quadratically optimising "
    << right
    << setw(15)
    << setprecision(9)
    << std::fixed
    << timeramat.toc()
    << left
    << setw(5)
    << "s"
    << left

    << setw(5)
    << printhour()
    << endl;

    for(int j=0; j<totalmatrixlength; ++j)
    {
        betaod(j)=gsl_vector_get(x, j);
    }

    gsl_vector_free(a);
    gsl_matrix_free(CE);
    gsl_vector_free(be);
    gsl_matrix_free(C);
    gsl_vector_free(b);
    gsl_vector_free(x);
    free (work);
    
    *outputf
    << left
    << setw(65)
    << "Procedure to construct the input and quadratically optimised it "
    << right
    << setw(0)
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

void LYAPUNOV::lyapunovfunctions(int currentiteration, bool type_of_grid, mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    int i=0, k=0;

    wall_clock timer;
    timer.tic();
    if(normal)
    {
        *outputf  << "Computing the Lyapunov function with the normalisation approach" << endl;
    }else{
        *outputf  << "Computing the Lyapunov function without the normalisation approach" << endl;
    }
    
    int chunk;
    int maxite=(int)evalcoordinates.n_rows;
    int maxbet=(int)rbf.collocationpoints.n_rows;
    int pointdim=(int)rbf.collocationpoints.n_cols;
    double totalweight=(double)sizeof(double)*(2.0*maxite)/1073741824.0;

    rbf.checkcapability(totalweight,"lyapunovfunctions");
    chunk = int(floor(maxite/glovar::OMP_NUM_THREADS));
#pragma omp parallel num_threads(glovar::OMP_NUM_THREADS) shared(lyapfunc,orbder,chunk) private(i,k)
    {
        lyapfunc.set_size(maxite);
        orbder.set_size(maxite);
        lyapfunc.zeros();
        orbder.zeros();
        rowvec diffpoints(pointdim), diffpointski(pointdim), diffpointskineg(pointdim);
        rowvec resulti(pointdim), resultk(pointdim), saving(pointdim), savingdomain(pointdim);
        diffpoints.zeros();
        diffpointski.zeros();
        diffpointskineg.zeros();
        resulti.zeros();
        resultk.zeros();
        saving.zeros();
        savingdomain.zeros();

        //double proctk=0.0;
        double producting=0.0;
        double twopointsdistance=0.0;
        double wdlfvalue1=0.0;
        double wdlfvalue2=0.0;
        double checking=0.0;
        
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk)// nowait
        for(i=0; i<maxite; ++i)
        {
            savingdomain=evalcoordinates(i,All);
            odesystem(normal,savingdomain,resulti);
            for(k=0; k<maxbet; ++k)
            {
                saving=rbf.collocationpoints(k,All);
                diffpoints=savingdomain-saving;
                twopointsdistance=sqrt(dot(diffpoints,diffpoints));
                checking=1.0-c*twopointsdistance;
                
                if(checking>0.0)
                {
                    odesystem(normal,saving,resultk);
                    wdlfvalue1=wendland.evawdlfn(twopointsdistance, wendland.wdlf1);
                    wdlfvalue2=wendland.evawdlfn(twopointsdistance, wendland.wdlf2);
                    diffpointski=saving-savingdomain;
                    //proctk=dot(diffpointski,resultk);
                    producting=betaod(k)*dot(diffpointski,resultk);
                    lyapfunc(i)+=producting*wdlfvalue1;
                    orbder(i)+=-wdlfvalue2*producting*dot(diffpointski,resulti)-betaod(k)*wdlfvalue1*dot(resulti,resultk);
                }
            }
        }
    }
    if(printing)
    {
        if(type_of_grid)
        {
            printall("lyapfuncdir",  currentiteration, lyapfunc);
            printall("orbderdir",  currentiteration, orbder);
        }else{
            printall("lyapfunccar",  currentiteration, lyapfunc);
            printall("orbdercar",  currentiteration, orbder);
        }
    }
    
        
    *outputf
    << left
    << setw(65)
    << "The whole procedure to compute the Lyapunov function lasted "
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


void LYAPUNOV::lyapunovfunctions(int currentiteration, mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    int i=0, k=0;

    wall_clock timer;
    timer.tic();
    if(normal)
    {
        *outputf  << "Computing the Lyapunov function with the normalisation approach" << endl;
    }else{
        *outputf  << "Computing the Lyapunov function without the normalisation approach" << endl;
    }

    int chunk;
    int maxite=(int)evalcoordinates.n_rows;
    int maxbet=(int)rbf.collocationpoints.n_rows;
    int pointdim=(int)rbf.collocationpoints.n_cols;
    double totalweight=(double)sizeof(double)*(2.0*maxite)/1073741824.0;

    rbf.checkcapability(totalweight,"lyapunovfunctions");
    chunk = int(floor(maxite/glovar::OMP_NUM_THREADS));
#pragma omp parallel num_threads(glovar::OMP_NUM_THREADS) shared(lyapfunc,orbder,chunk) private(i,k)
    {
        lyapfunc.set_size(maxite);
        orbder.set_size(maxite);
        lyapfunc.zeros();
        orbder.zeros();
        rowvec diffpoints(pointdim), diffpointski(pointdim), diffpointskineg(pointdim);
        rowvec resulti(pointdim), resultk(pointdim), saving(pointdim), savingdomain(pointdim);
        diffpoints.zeros();
        diffpointski.zeros();
        diffpointskineg.zeros();
        resulti.zeros();
        resultk.zeros();
        saving.zeros();
        savingdomain.zeros();

        //double proctk=0.0;
        double producting=0.0;
        double twopointsdistance=0.0;
        double wdlfvalue1=0.0;
        double wdlfvalue2=0.0;
        double checking=0.0;

#pragma omp barrier
#pragma omp for schedule (dynamic, chunk)// nowait
        for(i=0; i<maxite; ++i)
        {
            savingdomain=evalcoordinates(i,All);
            odesystem(normal,savingdomain,resulti);
            for(k=0; k<maxbet; ++k)
            {
                saving=rbf.collocationpoints(k,All);
                diffpoints=savingdomain-saving;
                twopointsdistance=sqrt(dot(diffpoints,diffpoints));
                checking=1.0-c*twopointsdistance;
                if(checking>0.0)
                {
                    odesystem(normal,saving,resultk);
                    wdlfvalue1=wendland.evawdlfn(twopointsdistance, wendland.wdlf1);
                    wdlfvalue2=wendland.evawdlfn(twopointsdistance, wendland.wdlf2);
                    diffpointski=saving-savingdomain;
                    //proctk=dot(diffpointski,resultk);
                    producting=betaod(k)*dot(diffpointski,resultk);
                    orbder(i)+=-wdlfvalue2*producting*dot(diffpointski,resulti)-betaod(k)*wdlfvalue1*dot(resulti,resultk);
                }
            }
        }
    }

    if(printing)
    {
        printall("orbderdir",  currentiteration, orbder);
    }
    
    *outputf
    << left
    << setw(65)
    << "The whole procedure to compute the Lyapunov function lasted "
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

void LYAPUNOV::firstderivative(int currentiteration, bool type_of_grid, mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    
    wall_clock timer;
    timer.tic();
    
    
    int i=0,j=0,k=0;
    int evaldim=(int)evalcoordinates.n_rows;
    
    fdvector.set_size(evaldim,ode_dimension);
    normed.set_size(evaldim);
    fdvector.zeros();
    double totalweight=(double)sizeof(double)*(ode_dimension*evaldim+evaldim)/1073741824.0;

    rbf.checkcapability(totalweight,"lyapunovfunctions");
    double checking=0.0;
    double twopointsdistance=0.0;
    double wdlfvalue1=0.0;
    double wdlfvalue2=0.0;
    double product=0.0;
    
    int chunk = int(floor(evaldim/glovar::OMP_NUM_THREADS));
#pragma omp parallel num_threads(glovar::OMP_NUM_THREADS) shared(fdvector,chunk) private(i,j,k,twopointsdistance,product,wdlfvalue1,wdlfvalue2,checking)
    {
        rowvec saving(ode_dimension), savingdomain(ode_dimension), diffpoints(ode_dimension),resultk(ode_dimension);
        int maxite=(int)betaod.n_rows;
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk) nowait
        for(j=0; j<evaldim; ++j)
        {
            saving=evalcoordinates(j,All);
            for(i=0; i<ode_dimension; ++i)
            {
                for(k=0; k<maxite; ++k)
                {
                    savingdomain=rbf.collocationpoints(k, All);
                    odesystem(normal,savingdomain,resultk);
                    diffpoints=saving-savingdomain;
                    twopointsdistance=sqrt(dot(diffpoints,diffpoints));
                    checking=1.0-c*twopointsdistance;
                    if(checking>0.0)
                    {
                        wdlfvalue1=wendland.evawdlfn(twopointsdistance, wendland.wdlf1);
                        wdlfvalue2=wendland.evawdlfn(twopointsdistance, wendland.wdlf2);
                        fdvector(j,i)+=betaod(k)*(-resultk(i)*wdlfvalue1
                                                  -diffpoints(i)
                                                  *dot(diffpoints,resultk)
                                                  *wdlfvalue2);
                    }
                }
            }
            product=dot(fdvector(j,All),fdvector(j,All));
            normed(j)=sqrt(product);
        }
    }
    
    if(printing)
    {
        if(type_of_grid)
        {
            printall("lyapprimexdir",  currentiteration, fdvector);
            printall("normeddire",  currentiteration, normed);
        }else{
            printall("lyapprimexcar",  currentiteration, fdvector);
            printall("normedcar",  currentiteration, normed);
        }
    }
    
    *outputf
    << left
    << setw(65)
    << "The whole procedure to compute the gradient lasted "
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

void LYAPUNOV::secondderivative(mat &evalcoordinates, WENDLAND &wendland, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    wall_clock timer;
    timer.tic();
    
    
    int i=0,j=0,k=0;
    int evaldim=(int)evalcoordinates.n_rows;
    double totalweight=(double)sizeof(double)*(ode_dimension*ode_dimension*evaldim)/1073741824.0;

    rbf.checkcapability(totalweight,"lyapunovfunctions");

    sdvector.set_size(ode_dimension,ode_dimension,evaldim);
    sdvector.zeros();
    
    int chunk = int(floor(evaldim/glovar::OMP_NUM_THREADS));
#pragma omp parallel num_threads(glovar::OMP_NUM_THREADS) shared(sdvector,chunk) private(i,j,k)
    {
        rowvec resultk(ode_dimension), saving(ode_dimension), savingdomain(ode_dimension), diffpoints(ode_dimension);
        int maxite=(int)betaod.n_rows;
        double twopointsdistance=0.0;
        double wdlfvalue2=0.0;
        double wdlfvalue3=0.0;
        double kdelta=0;
        double product=0.0;
        mat deltak=eye(ode_dimension,ode_dimension);
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk) nowait
        for(int p=0; p<evaldim; ++p)
        {
            savingdomain=evalcoordinates(p,All);//equis
            for(i=0; i<ode_dimension; ++i)
            {
                for(j=0; j<ode_dimension; ++j)
                {
                    kdelta=0.0;
                    if(i==j)
                    {
                        kdelta=1.0;
                    }
                    for(k=0; k<maxite; ++k)
                    {
                        saving=rbf.collocationpoints(k, All);//equisk
                        odesystem(normal,saving,resultk);
                        diffpoints=savingdomain-saving;
                        twopointsdistance=sqrt(dot(diffpoints,diffpoints));
                        product=dot(diffpoints,resultk);
                        checking=1.0-c*twopointsdistance;
                        
                        if(checking>0.0)
                        {
                            wdlfvalue2=wendland.evawdlfn(twopointsdistance,wendland.wdlf2);
                            wdlfvalue3=wendland.evawdlfn(twopointsdistance,wendland.wdlf3);
                            sdvector(i,j,p)+=betaod(k)*(
                                        -diffpoints(j)*resultk(i)*wdlfvalue2
                                        -kdelta*product*wdlfvalue2
                                        -diffpoints(i)*resultk(j)*wdlfvalue2
                                        -diffpoints(i)*diffpoints(j)*product*wdlfvalue3
                                        );
                        }
                    }
                }
            }
        }
    }

    *outputf
    << left
    << setw(65)
    << "The whole procedure to compute the Hessian lasted "
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


void LYAPUNOV::findingeigenamount(int currentiteration, bool type_of_grid, RBFMETHOD &rbf)
{
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(glovar::OMP_NUM_THREADS);
#endif
    wall_clock timer;
    timer.tic();
    



    int maxlim=(int)sdvector.n_slices;
    int dimlim=(int)rbf.collocationpoints.n_cols;
    double totalweight=(double)sizeof(double)*(maxlim*dimlim+dimlim*dimlim*maxlim)/1073741824.0;

    rbf.checkcapability(totalweight,"lyapunovfunctions");

    mat x(dimlim,dimlim);
    TOTALEIGEN.set_size(maxlim,dimlim);
    TOTALEIGENV.set_size(dimlim,dimlim,maxlim);
    int i=0;
    vec eigval;
    mat eigvec;


    int chunk = int(floor(maxlim/glovar::OMP_NUM_THREADS));
    
#pragma omp parallel shared(TOTALEIGEN,chunk) private(i,x,eigval, eigvec)
    {
#pragma omp barrier
#pragma omp for schedule (dynamic, chunk) nowait
        for(i=0; i<maxlim; ++i)
        {
                        x=sdvector(All,All,span(i));
                        eig_sym(eigval, eigvec, x,"std");
                        TOTALEIGEN(i,All)=eigval.t();
                        TOTALEIGENV(span(0,dimlim-1),span(0,dimlim-1),span(i))=eigvec;
        }
    }
    
    if(printing)
    {        
        if(type_of_grid)
        {
            printall("eigendir",  currentiteration, TOTALEIGEN);
            printall("eigenvecdir",  currentiteration, TOTALEIGENV);
        }else{
            printall("eigencar",  currentiteration, TOTALEIGEN);
            printall("eigenveccar",  currentiteration, TOTALEIGENV);
        }
    }

    *outputf
    << left
    << setw(65)
    << "The whole procedure to compute the Eigenvalues lasted "
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


void LYAPUNOV::chainrecurrentset(int currentiteration, bool type_of_grid, bool with_orbder, mat &evalcoordinates)
{
    list<int> counterzero;
    int maxlength=(int)evalcoordinates.n_rows;
    int maxwidth=(int)evalcoordinates.n_cols;
    for(int j=0; j<maxlength; ++j)
    {
        if(with_orbder)
        {
            if(orbder(j)>critval)
            {
                counterzero.push_back(j);
            }
        }else{
            if(-normed(j)>-critval)
            {
                counterzero.push_back(j);
            }
        }
    }
    int faillength=(int)counterzero.size();
    crslyapun.set_size(faillength);
    crsorbder.set_size(faillength);
    failinggrid.set_size(faillength,maxwidth);
    failinglyapunov.set_size(faillength);
    failingorbder.set_size(faillength);
    int m=0;
    {
        for(list<int>::iterator ii=counterzero.begin(); ii!=counterzero.end(); ++ii)
        {
            crslyapun(m)=lyapfunc(*ii);
            crsorbder(m)=orbder(*ii);
            failinglyapunov(m)=lyapfunc((*ii));
            failingorbder(m)=orbder((*ii));
            failinggrid(m,All)=evalcoordinates(*ii,All);
            m++;
        }
    }
    counterzero.clear();
    if(printing)
    {
        if(type_of_grid)
        {
            printall("fdirecgrid",  currentiteration, failinggrid);
            printall("flfdirecgrid",  currentiteration, failinglyapunov);
            printall("flfpdirecgrid",  currentiteration, failingorbder);
        }else{
            printall("fcartesian",  currentiteration, failinggrid);
            printall("flfcartesian",  currentiteration, failinglyapunov);
            printall("flfpcartesian",  currentiteration, failingorbder);
        }
    }
    
}


void LYAPUNOV::getnewalpha(int currentiteration, RBFMETHOD &rbf)
{
    double summing=0.0;
    double normalizationfactor=0.0;
    rbf.alphavector.resize(rbf.alphasize);
    for(int iii=0; iii<rbf.alphasize; ++iii)
    {
        summing=0.0;
        for(int j=0; j<2*points_directional;++j)
        {
            summing+=orbder((2*points_directional)*(iii)+j);
        }
        if(summing>0.0)
        {
            summing=0.0;
        }
        rbf.alphavector(iii)=summing/((double)(2*points_directional));
        normalizationfactor+=rbf.alphavector(iii);
    }
    rbf.alphavector=abs(rbf.alphasize/normalizationfactor)*rbf.alphavector;
    if(printing)
    {
        printall("alphavector", currentiteration, rbf.alphavector);
    }
    
}

void LYAPUNOV::make_lyap_direcional(WENDLAND &wendland, RBFMETHOD &rbf)
{
    
    int i;
    wall_clock timer1;
    
    for(i=0; i<totaliterations; i++)
    {
        iteration(i);
        timer1.tic();
        
        
        lyapequation(i, rbf);
        lyapunovfunctions(i, true, rbf.directgrid, wendland, rbf);
        getnewalpha(i, rbf);
        chainrecurrentset(i, true, true, rbf.directgrid);
        firstderivative(i, true, rbf.directgrid, wendland, rbf);
        normed.clear();
        secondderivative(rbf.directgrid, wendland, rbf);
        findingeigenamount(i, true, rbf);
        
        *outputf
        << left
        << setw(50)
        << "The whole procedure to go through iteration "
        << left
        << setw(2)
        << i
        << left
        << setw(5)
        << "lasted "
        << left
        << setw(5)
        << setprecision(9)
        << std::fixed
        << timer1.toc()
        << left
        << setw(5)
        << "s"
        << left
        << setw(5)
        << printhour()
        << endl;


        switch (glovar::method_type){
        case glovar::choose_the_method::minusone:
            break;
        case glovar::choose_the_method::op:
        case glovar::choose_the_method::fpop:
            i=2*totaliterations;
            break;
        default:
            *outputf << "METHOD: THIS HAS NOT BEEN ASSIGNED" <<endl;
            exit(9);

        }

    }
    
}


void LYAPUNOV::make_lyap_direc_and_cart(WENDLAND &wendland, RBFMETHOD &rbf)
{
    
    int i;
    wall_clock timer1;
    
    rbf.makeevalblpn(true);

    bool decide;
    for(i=0; i<totaliterations; i++)
    {
        timer1.tic();
        iteration(i);
        *outputf  << " " << endl;
        lyapequation(i, rbf);
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Directional grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        decide=true;
        lyapunovfunctions(i,decide, rbf.directgrid, wendland, rbf);
        chainrecurrentset(i, decide, true,  rbf.directgrid);
        firstderivative(i, decide, rbf.directgrid, wendland, rbf);
        normed.clear();
        secondderivative(rbf.directgrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        getnewalpha(i, rbf);
        
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        decide=false;
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Cartesian grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        lyapunovfunctions(i,decide, rbf.cartesianevalgrid, wendland, rbf);
        chainrecurrentset(i, decide, true, rbf.cartesianevalgrid);
        firstderivative(i, decide, rbf.cartesianevalgrid, wendland, rbf);
        normed.clear();
        secondderivative(rbf.cartesianevalgrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        *outputf
        << "\n"
        << left
        << setw(30)
        <<"  "
        << left
        << setw(31)
        << "The whole procedure for iteration  "
        << left
        << setw(2)
        << i
        << left
        << setw(5)
        << "lasted "
        << left
        << setw(5)
        << setprecision(9)
        << std::fixed
        << timer1.toc()
        << left
        << setw(5)
        << "s."
        << endl;
        *outputf
        << left
        << setw(47)
        << " "
        << printhour()
        << endl;


        switch (glovar::method_type){
        case glovar::choose_the_method::minusone:
            break;
        case glovar::choose_the_method::op:
        case glovar::choose_the_method::fpop:
            i=2*totaliterations;
            break;
        default:
            *outputf << "METHOD: THIS HAS NOT BEEN ASSIGNED" <<endl;
            exit(9);

        }
    }
    
}


void LYAPUNOV::make_chainrecurrent_eigenvalues(WENDLAND &wendland, RBFMETHOD &rbf)
{
    
    int i;
    wall_clock timer1;
    
    rbf.makeevalblpn(true);


    bool decide;
    for(i=0; i<totaliterations; i++)
    {
        timer1.tic();
        iteration(i);
        *outputf  << " " << endl;
        lyapequation(i, rbf);
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Directional grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        decide=true;
        lyapunovfunctions(i,decide, rbf.directgrid, wendland, rbf);
        chainrecurrentset(i, decide, true, rbf.directgrid);
        firstderivative(i, decide, failinggrid, wendland, rbf);
        normed.clear();
        secondderivative(failinggrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        getnewalpha(i, rbf);

        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        decide=false;
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Cartesian grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        lyapunovfunctions(i,decide, rbf.cartesianevalgrid, wendland, rbf);
        chainrecurrentset(i, decide, true, rbf.cartesianevalgrid);
        firstderivative(i, decide, failinggrid, wendland, rbf);
        normed.clear();
        secondderivative(failinggrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);

        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        *outputf
        << "\n"
        << left
        << setw(30)
        <<"  "
        << left
        << setw(31)
        << "The whole procedure for iteration  "
        << left
        << setw(2)
        << i
        << left
        << setw(5)
        << "lasted "
        << left
        << setw(5)
        << setprecision(9)
        << std::fixed
        << timer1.toc()
        << left
        << setw(5)
        << "s."
        << endl;
        *outputf
        << left
        << setw(47)
        << " "
        << printhour()
        << endl;

        switch (glovar::method_type){
        case glovar::choose_the_method::minusone:
            break;
        case glovar::choose_the_method::op:
        case glovar::choose_the_method::fpop:
            i=2*totaliterations;
            break;
        default:
            *outputf << "METHOD: THIS HAS NOT BEEN ASSIGNED" <<endl;
            exit(9);

        }
    }
    
}



void LYAPUNOV::make_norm_chain_recurrent_sets(WENDLAND &wendland, RBFMETHOD &rbf)
{
    
    int i;
    wall_clock timer1;
    if(critval<0)
    {
        *outputf << "EXECUTION STOPPED " << endl;
        *outputf << "The critical value must be positive or zero, you are chosing to obtain the chain-recurrent set with the norm and the norm is positive or zero " << endl;
        finalization();
        exit(0);
    }

    rbf.makeevalblpn(true);

    bool decide;
    for(i=0; i<totaliterations; i++)
    {
        timer1.tic();
        iteration(i);
        *outputf  << " " << endl;
        lyapequation(i, rbf);
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Directional grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        decide=true;
        lyapunovfunctions(i, decide, rbf.directgrid, wendland, rbf);
        firstderivative(i, decide, rbf.directgrid, wendland, rbf);
        chainrecurrentset(i, decide, false, rbf.directgrid);
        secondderivative(failinggrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        getnewalpha(i, rbf);
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        normed.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        decide=false;
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Cartesian grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        lyapunovfunctions(i,decide, rbf.cartesianevalgrid, wendland, rbf);
        firstderivative(i, decide, rbf.cartesianevalgrid, wendland, rbf);
        chainrecurrentset(i, decide, false, rbf.cartesianevalgrid);
        secondderivative(failinggrid, wendland, rbf);
        findingeigenamount(i, decide, rbf);
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        normed.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        *outputf
        << "\n"
        << left
        << setw(30)
        <<"  "
        << left
        << setw(31)
        << "The whole procedure for iteration  "
        << left
        << setw(2)
        << i
        << left
        << setw(5)
        << "lasted "
        << left
        << setw(5)
        << setprecision(9)
        << std::fixed
        << timer1.toc()
        << left
        << setw(5)
        << "s."
        << endl;
        *outputf
        << left
        << setw(47)
        << " "
        << printhour()
        << endl;
        switch (glovar::method_type){
        case glovar::choose_the_method::minusone:
            break;
        case glovar::choose_the_method::op:
        case glovar::choose_the_method::fpop:
            i=2*totaliterations;
            break;
        default:
            *outputf << "METHOD: THIS HAS NOT BEEN ASSIGNED" <<endl;
            exit(9);

        }
    }
    
}


void LYAPUNOV::make_lyap_cartesian(WENDLAND &wendland, RBFMETHOD &rbf)
{

    int i;
    wall_clock timer1;
    if(critval>0)
    {
        *outputf << "EXECUTION STOPPED " << endl;
        *outputf << "The critical value must be negative or zero, you are chosing to obtain the chain-recurrent set with the orbital derivative and the orbital derivative is negative or zero " << endl;
        finalization();
        exit(0);
    }

    rbf.makeevalblpn(true);
    for(i=0; i<totaliterations; i++)
    {
        iteration(i);
        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Solving the Lyapuov equation:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        timer1.tic();
        lyapequation(i, rbf);

        *outputf  << " " << endl;
        *outputf  << " " << endl;
        *outputf  << "Cartesian grid:" << endl;
        *outputf  << " " << endl;
        *outputf  << " " << endl;

        lyapunovfunctions(i,false, rbf.cartesianevalgrid, wendland, rbf);
        chainrecurrentset(i, false, true, rbf.cartesianevalgrid);
        firstderivative(i, false, rbf.cartesianevalgrid, wendland, rbf);
        normed.clear();
        secondderivative(rbf.cartesianevalgrid, wendland, rbf);
        findingeigenamount(i, false, rbf);
        failinggrid.clear();
        failinglyapunov.clear();
        failingorbder.clear();
        direcnozero.clear();
        negcollocation.clear();
        crslyapun.clear();
        crsorbder.clear();
        lyapfunc.clear();
        orbder.clear();
        fdvector.clear();
        sdvector.clear();
        normed.clear();
        TOTALEIGEN.clear();
        TOTALEIGENV.clear();
        
        if(totaliterations>1)
        {
            *outputf  << " " << endl;
            *outputf  << " " << endl;
            *outputf  << "Computing the iterative part with the directional grid:" << endl;
            lyapunovfunctions(i,rbf.directgrid, wendland, rbf);
            getnewalpha(i, rbf);
            orbder.clear();
        }
        
        *outputf
        << "\n"
        << left
        << setw(30)
        <<"  "
        << left
        << setw(31)
        << "The whole procedure for iteration  "
        << left
        << setw(2)
        << i
        << left
        << setw(5)
        << "lasted "
        << left
        << setw(5)
        << setprecision(9)
        << std::fixed
        << timer1.toc()
        << left
        << setw(5)
        << "s."
        << endl;
        *outputf
        << left
        << setw(47)
        << " "
        << printhour()
        << endl;
        
        switch (glovar::method_type){
        case glovar::choose_the_method::minusone:
            break;
        case glovar::choose_the_method::op:
        case glovar::choose_the_method::fpop:
            i=2*totaliterations;
            break;
        default:
            *outputf << "METHOD: THIS HAS NOT BEEN ASSIGNED" <<endl;
            exit(9);
        }
    }

}


void LYAPUNOV::make_check_num_approximation(WENDLAND &wendland, RBFMETHOD &rbf)
{

    *outputf  << " " << endl;
    *outputf  << " " << endl;
    *outputf  << "Verifying the numerical approximation:" << endl;
    *outputf  << " " << endl;
    *outputf  << " " << endl;

    int i=0;
    wall_clock timer1;
    {
        iteration(i);
        timer1.tic();
        rbf.checkdim();
        rbf.checkcond();
        lyapequation(i, rbf);
        lyapunovfunctions(i,rbf.collocationpoints, wendland, rbf);
        
        glovar::outputf
        << left
        << setw(75)
        << "The orbital derivative mean value is "
        << left
        << setw(5)
        << mean(orbder)
        << endl;

        glovar::outputf
        << left
        << setw(75)
        << "The orbital derivative median value is "
        << left
        << setw(5)
        << median(orbder)
        << endl;

        glovar::outputf
        << left
        << setw(75)
        << "The orbital derivative standard deviation value is "
        << left
        << setw(5)
        << stddev(orbder)
        << endl;

        glovar::outputf
        << left
        << setw(75)
        << "The orbital derivative variance value is "
        << left
        << setw(5)
        << var(orbder)
        << endl;

        glovar::outputf
        << left
        << setw(75)
        << "The orbital derivative range value is "
        << left
        << setw(5)
        << range(orbder)
        << endl;

        
        orbder.clear();

        *outputf
        << "\n"
        << left
        << setw(30)
        <<"  "
        << left
        << setw(31)
        << "The whole procedure for iteration  "
        << left
        << setw(2)
        << i
        << left
        << setw(5)
        << "lasted "
        << left
        << setw(5)
        << setprecision(9)
        << std::fixed
        << timer1.toc()
        << left
        << setw(5)
        << "s."
        << endl;
        *outputf
        << left
        << setw(47)
        << " "
        << printhour()
        << endl;
        
    }
}

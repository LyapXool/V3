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
#include "generalities.hpp"
#include "odesystem.hpp"
#include <iomanip>

using namespace arma;
using namespace std;

arma::span const All=span::all;

void printinformation()
{
    ofstream datos;
    datos.open ("data.lpx", fstream::out);
    datos  << setfill(' ') << setw(51) << " " <<  " LyapXool-V3 " << setfill(' ') << setw(51) <<  " " << endl;
    datos  << setfill(' ') << setw(47) << " " <<  printhour() << setw(51) <<  " " << endl;
    datos  << "\n" << endl;
    datos  << setfill('=') << setw(52) << "=" <<  " PARAMETERS " << setfill('=') << setw(52) <<  "=" << endl;
    datos  << setfill(' ') << endl;
    datos << "Parameters needed to reproduce this computation." << endl;
    datos << " " << endl;
    datos << left
    << setw(15)
    << "Instruction for the computation type "
    << left
    << setw(5)
    << glovar::probnames[glovar::computation_type]
    << endl;
    
    
    datos << left
    << setw(38)
    << "Instruction for the method "
    << left
    << setw(5)
    << glovar::metnames[glovar::method_type]
    << endl;
    
    datos << left
    << setw(38)
    << "Normalisation method ";
    if(glovar::normal)
    {
        datos << "true";
    }else{
        datos << "false";
    }
    datos << endl;
    if(glovar::totaliterations>0)
    {
        datos << left
        <<setw(38)
        <<"Total iterations: "
        << glovar::totaliterations << endl;
    }else{
        datos << "Computations made for one single iteration.\n" << endl;
    }
    datos << "\n" << endl;
    datos <<  "Wendland function parameters " << endl;
    datos << left
    << setw(38)
    <<"l"
    << glovar::l
    << endl;
    
    datos << left
    << setw(38)
    << "k"
    << glovar::k
    << endl;
    
    datos << left
    << setw(38)
    << "c"
    << glovar::c
    << "\n"
    << endl;
    
    datos << left
    << setw(38)
    << "alpha: "
    << glovar::alpha
    << endl;
    datos
    << left
    << setw(37)
    << "Gamma value: "
    << left
    << setw(5)
    << glovar::critval
    << endl;
    datos << left
    << setw(38)
    << "The Cartesian scaling parameter is  "
    << glovar::cart_grid_scaling
    << "\n"
    << endl;
    
    datos << " " << endl;
    
    
    datos << "The minima geometric limits are " << endl;
    for(int jc=0; jc<glovar::ode_dimension; ++jc)
    {
        datos << left
        << setw(1)
        << "axis" <<jc
        << setw(33)
        << " "
        << left
        << setw(5)
        << glovar::min_geometric_limits[jc]
        << endl;
    }
    datos << endl;
    
    datos << "The maxima geometric limits are " << endl;
    for(int jc=0; jc<glovar::ode_dimension; ++jc)
    {
        datos << left
        << setw(1)
        << "axis" <<jc
        << setw(34)
        << " "
        << left
        << setw(5)
        << glovar::max_geometric_limits[jc]
        << endl;
    }
    datos << endl;
    
    datos << "For the iterative grid " << endl;
    datos << left
    << setw(38)
    << "The radius "
    << left
    << setw(5)
    << glovar::radius
    << endl;
    datos
    << left
    << setw(38)
    << "Points per side "
    << left
    << setw(5)
    << glovar::points_directional
    << endl;
    
    datos << " " <<endl;
    datos << left
    << setw(38)
    << "Condition number ";
    if(glovar::condnumber)
    {
        datos << "true";
    }else{
        datos << "false";
    }
    datos << endl;
    datos << "\n" << endl;
    datos << setfill(' ') << setw(51) << " " <<  " LyapXool-V3 " << setfill(' ') << setw(51) <<  " " << endl;
    datos << setfill(' ') << setw(45) << " " <<  " cargaezg@iingen.unam.mx " << setfill(' ') << setw(51) <<  " " << endl;
    
    datos.close();
}

void printall(const string nombre, int ordernum, mat &vectoraimprimmir)
{
    wall_clock timer;
    timer.tic();
    const std::string output_file_extension="m";
    int dim=(int)vectoraimprimmir.n_rows;
    int dim2=(int)vectoraimprimmir.n_cols;
    if((dim!=0)&&(dim2!=0))
    {
        //glovar::outputf << "Printing results..." << endl;
        if(dim2 >= dim)
        {
            for(int totalwidth=0; totalwidth<dim; ++totalwidth)
            {
                ostringstream fileName;
                fileName<<"s"<<nombre<<"var"<<totalwidth<<"."<<output_file_extension;
                ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "=[";
            
                for(int p=0; p<dim2; ++p)
                {
                    valuesdocument << std::fixed << std::setprecision(18) << vectoraimprimmir(totalwidth,p) << " " ;
                }
                valuesdocument << "];" << endl;
                valuesdocument.close();
            }
        }else{
            for(int totalwidth=0; totalwidth<dim2; ++totalwidth)
            {
                ostringstream fileName;
                fileName<<"s"<<nombre<<"var"<<totalwidth<<"."<<output_file_extension;
                ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "=[";
                
                for(int p=0; p<dim; ++p)
                {
                    valuesdocument << std::fixed << std::setprecision(18) << vectoraimprimmir(p,totalwidth) << " " ;
                }
                valuesdocument << "];" << endl;
                valuesdocument.close();
            }
        }
    }else{
        glovar::outputf << "WARNING: " << nombre << " does not contain values and its dimension is 0 " << endl;
        finalization();
        exit(0);
    }
}

void printall(const string nombre, int ordernum, cube &vectoraimprimmir)
{
    wall_clock timer;
    timer.tic();
    const std::string output_file_extension="m";
    int dim=(int)vectoraimprimmir.n_rows;
    int dim2=(int)vectoraimprimmir.n_cols;
    int dim3=(int)vectoraimprimmir.n_slices;
    if((dim!=0)&&(dim2!=0))
    {
        {
            for(int totalwidth=0; totalwidth<dim; ++totalwidth)
            {
                ostringstream fileName;
                fileName<<"s"<<nombre<<"var"<<totalwidth<<"."<<output_file_extension;
                ofstream valuesdocument(fileName.str(), fstream::out | fstream::app);
                valuesdocument << nombre<< "var"<<totalwidth << "ite" << ordernum << "=[";
                for(int nsl=0; nsl<dim3; ++nsl)
                {
                    for(int p=0; p<dim2; ++p)
                    {
                        valuesdocument << std::fixed << std::setprecision(18) << vectoraimprimmir(totalwidth,p,nsl) << " " ;
                    }
                }
                valuesdocument << "];" << endl;
                valuesdocument.close();
            }
        }
    }else{
        glovar::outputf << "WARNING: " << nombre << " does not contain values and its dimension is 0 " << endl;
        finalization();
        exit(0);
    }
}

void printhour(const int &definition)
{
    time_t tempus;
    struct tm * infotiempo;
    char datos[100];
    
    time (&tempus);
    infotiempo = localtime(&tempus);
    if(definition==0)
    {
        strftime(datos,sizeof(datos),"Computation started on %d-%m-%Y at %H:%M:%S",infotiempo);
    }
    if(definition==1)
    {
        strftime(datos,sizeof(datos),"Computation finished on %d-%m-%Y at %H:%M:%S",infotiempo);
    }
    string str(datos);
    
    glovar::outputf << "\n" << str << "\n" << endl;
    
}


string printhour()
{
    time_t tempus;
    struct tm * infotiempo;
    char datos[100];
    
    time (&tempus);
    infotiempo = localtime(&tempus);
    
    strftime(datos,sizeof(datos),"%d-%m-%Y at %H:%M:%S",infotiempo);
        
    string str(datos);
     
    return str;
}



void iteration(int i)
{
    glovar::outputf  << setfill('=') << setw(50) << "=" <<  " Iteration no. " << i << " " << setfill('=') << setw(50) <<  "=" << endl;
    glovar::outputf  << setfill(' ') << endl;
}

void finalization()
{
    glovar::outputf  << " " << endl;
    glovar::outputf  << setfill('=') << setw(51) << "=" <<  " FINALIZATION " << setfill('=') << setw(51) <<  "=" << endl;
    glovar::outputf  << setfill(' ') << endl;
    glovar::outputf  << " " << endl;
    
    glovar::outputf
    << left
    << setw(20)
    << "The ODE system was called "
    << left
    << setw(2)
    << glovar::functionodecalls
    << left
    << setw(5)
    << " times."
    << endl;
}

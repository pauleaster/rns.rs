# rns.rs

This is going to become a rust port of the RNS code. The following has been taken from the header at http://www.gravity.phys.uwm.edu/rns/
EoS files downloaded from the same url.

# Rapidly Rotating Neutron Star
RNS is a code written by Nikolaos Stergioulas which constructs models of rapidly rotating, relativistic, compact stars using tabulated equations of state which are supplied by the user. Please direct questions about this program to either [Sharon Morsink](mailto:morsink@phys.ualberta.ca) or [Nikolaos Stergioulas](mailto:niksterg@aei-potsdam.mpg.de).

The code was mainly developed while Stergioulas and Morsink worked at UWM, Nikolaos as a graduate student and Sharon as a postdoc. Currently Nikolaos Stergioulas is an assistant professor at the Aristotle University of Thessaloniki (Greece) and Sharon Morsink is an Associate Professor at University of Alberta (Canada).

For mre information, please refer to http://www.gravity.phys.uwm.edu/rns/ 

If you want any information on the rust port then contact me at https://github.com/pauleaster/rns.rs

# Rapidly Rotating Neutron Star

This code has been downloaded from http://www.gravity.phys.uwm.edu/rns/source/rns.v2.0.tar.gz

The main website is http://www.gravity.phys.uwm.edu/rns/

The following is the readme from RNS version 2.0

If time permits, I will attempt to port this to rust.

rns.v2.0

Index to files:

FILE		DESCRIPTION

main.c	 	An example program which uses the routines in
		the subsequent files. 

makefile	The makefile for the example. It will produce
		an executable named "kepler". 

sample.out	Sample output from the program kepler

equil.c 	The routines used to integrate a rapidly rotating 
		neutron star. 

equil_util.c	Some utility programs used by equil.c

nrutil.c	Numerical Recipes routines used by the program.

HEADER FILES:

consts.h	Constants used by the programs

equil.h,	Function prototypes 
equil_util.h, 
nrutil.h  

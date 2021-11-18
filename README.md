# rns.rs

This is a rust port of the RNS code. If you want information on this rust port then you can contact me at [Paul Easter](mailto:paul.john.easter@gmail.com).  The code structure, at this point in time, deliberately follows the original structure for the C code. The only difference is that XXX.c is now XXX.rs. However, some functions may have been moved.

So far this code achieves the same numerical result as the C code to around a fractional error of 1e-11, which I found really interesting. Though, as much as possible, I performed a literal translation from C to rust avoiding the temptation to optimise before confirming that the results were comparable. I did perform a couple of optimisations though. I profiled the rust code with flamegraph.rs and found that a significant amount of time was spent in the spin() function, specifically, f64::sin(). I found that a sin() loop was being executed for each run of spin() which generated fixed values each time. So I moved this out to the main() function and passed this data to spin() directly. Even without this optimisation, I found that the rust code was faster than the C code for the same data (when using the `cargo run --release` flag). After this optimisation, the rust code executed in 1.2 seconds and the c code in 2.4 seconds. I have added in movies that show the running of both codes with the same data. The rust code also generates html files showing contour plots of various system parameters they can be found in the ./html folder.

# Details of original C code

The original C code along with the equation of state files were downloaded from http://www.gravity.phys.uwm.edu/rns/. The LS220 equation of state file was downloaded from https://core-gitlfs.tpi.uni-jena.de/eos/ls220. These equation of state files were used for testing.

The description of the RNS code from http://www.gravity.phys.uwm.edu/rns/ follows here:

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

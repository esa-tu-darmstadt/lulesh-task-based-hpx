#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include "lulesh.h"

/* Helper function for converting strings to ints, with error checking */
template<typename IntT>
int StrToInt(const char *token, IntT *retVal)
{
   const char *c ;
   char *endptr ;
   const int decimal_base = 10 ;

   if (token == NULL)
      return 0 ;

   c = token ;
   *retVal = strtol(c, &endptr, decimal_base) ;
   if((endptr != c) && ((*endptr == ' ') || (*endptr == '\0')))
      return 1 ;
   else
      return 0 ;
}

static void ParseError(const char *message, int myRank)
{
   if (myRank == 0) {
      printf("%s\n", message);
      exit(-1);
   }
}

void ParseCommandLineOptions(hpx::program_options::variables_map &vm,
                             Int_t myRank, struct cmdLineOpts *opts)
{
  opts->its = vm["i"].as<Int_t>();
  opts->nx = vm["s"].as<Int_t>();
  opts->numReg = vm["r"].as<Int_t>();
  opts->numFiles = vm["f"].as<Int_t>();
  opts->balance = vm["b"].as<Int_t>();
  opts->cost = vm["c"].as<Int_t>();
  if (vm.count("p"))
    opts->showProg = 1;
  if (vm.count("q"))
    opts->quiet = 1;
  if (vm.count("v")) {
#if VIZ_MESH
    opts-viz = 1;
#else
    ParseError("Use of -v requires compiling with -DVIZ_MESH\n", myRank);
#endif
  }
}

/////////////////////////////////////////////////////////////////////

void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx,
                               Int_t numRanks)
{
   // GrindTime1 only takes a single domain into account, and is thus a good way to measure
   // processor speed indepdendent of MPI parallelism.
   // GrindTime2 takes into account speedups from MPI parallelism.
   // Cast to 64-bit integer to avoid overflows.
   Int8_t nx8 = nx;
   Real_t grindTime1 = ((elapsed_time*1e6)/locDom.cycle())/(nx8*nx8*nx8);
   Real_t grindTime2 = ((elapsed_time*1e6)/locDom.cycle())/(nx8*nx8*nx8*numRanks);

   Index_t ElemId = 0;
   std::cout << "Run completed:\n";
   std::cout << "   Problem size        =  " << nx       << "\n";
   std::cout << "   Iteration count     =  " << locDom.cycle() << "\n";
   std::cout << "   Final Origin Energy =  ";
   std::cout << std::scientific << std::setprecision(6);
   std::cout << std::setw(12) << locDom.e(ElemId) << "\n";

   Real_t   MaxAbsDiff = Real_t(0.0);
   Real_t TotalAbsDiff = Real_t(0.0);
   Real_t   MaxRelDiff = Real_t(0.0);

   for (Index_t j=0; j<nx; ++j) {
      for (Index_t k=j+1; k<nx; ++k) {
         Real_t AbsDiff = FABS(locDom.e(j*nx+k)-locDom.e(k*nx+j));
         TotalAbsDiff  += AbsDiff;

         if (MaxAbsDiff <AbsDiff) MaxAbsDiff = AbsDiff;

         // Real_t RelDiff = AbsDiff / locDom.e(k*nx+j);
         Real_t RelDiff = FABS(locDom.e(k*nx+j)) > 1e-8 ? AbsDiff / locDom.e(k*nx+j) : 0.0; 

         if (MaxRelDiff <RelDiff)  MaxRelDiff = RelDiff;
      }
   }

   // Quick symmetry check
   std::cout << "   Testing Plane 0 of Energy Array on rank 0:\n";
   std::cout << "        MaxAbsDiff   = " << std::setw(12) << MaxAbsDiff   << "\n";
   std::cout << "        TotalAbsDiff = " << std::setw(12) << TotalAbsDiff << "\n";
   std::cout << "        MaxRelDiff   = " << std::setw(12) << MaxRelDiff   << "\n";

   // Timing information
   std::cout.unsetf(std::ios_base::floatfield);
   std::cout << std::fixed << std::setprecision(2);
   std::cout << "\nElapsed time         = " << std::setw(10) << elapsed_time << " (s)\n";
   std::cout << std::setprecision(8);
   std::cout << "Grind time (us/z/c)  = "  << std::setw(10) << grindTime1 << " (per dom)  ("
             << std::setw(10) << elapsed_time << " overall)\n";
   std::cout << "FOM                  = " << std::setw(10) << 1000.0/grindTime2 << " (z/s)\n\n";

   return ;
}

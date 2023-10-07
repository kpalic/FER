#include <stdlib.h>
#include "engine.h" // ovo je include fajl za MATLABov engine

int main()

{
	Engine *ep; // pointer na MATLAB-ov engine
	mxArray *ulaz = NULL, *izlaz = NULL; // deklariramo dvije MATLAB varijable
	double rezultat;

	// pokretanje MATLAB engine-a, uz standardne provjere ispravnosti
	if (!(ep = engOpen("\0"))) 
	{	fprintf(stderr, "\nCan't start MATLAB engine\n");
		return EXIT_FAILURE;
	}

	ulaz = mxCreateDoubleScalar(7.34);  // definiramo vrijednost MATLAB varijabli
	
	engPutVariable(ep, "X", ulaz);  // saljemo je u MATLAB engine
	
	engEvalString(ep, "Y = X * 2"); // saljemo string na izracunavanje

    izlaz = engGetVariable(ep, "Y"); // dohvacamo varijablu iz MATLAB okoline
	
	rezultat = mxGetScalar(izlaz);  // vrijednost kopiramo u double varijablu
	
	printf("MATLAB je izracunao: %g\n", rezultat); // ispisujemo dobiveni rezultat

    fgetc(stdin);
	
	engClose(ep); // gasimo engine

    // na kraju: OBVEZATNO UNISTITI SVE MATLAB VARIJABLE! (OSLOBODITI MEMORIJU)
	mxDestroyArray(ulaz);
    mxDestroyArray(izlaz);
	
	return EXIT_SUCCESS;
}
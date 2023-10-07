/* Definiramo matricu te je saljemo MATLABu
   Pomocu MATLABa ucitamo matricu sa diska i izracunamo inverziju */
#include <stdlib.h>
#include "engine.h" // ovo je include fajl za MATLABov engine
#define  BUFSIZE 512

int main()
{
	Engine *ep; // pointer na MATLAB-ov engine
	mxArray *ulaz, *izlaz; // deklariramo MATLAB varijable
	double rezultat, *p;
	int i;
	char buffer[BUFSIZE];

	// pokretanje MATLAB engine-a, uz standardne provjere ispravnosti
	if (!(ep = engOpen("\0"))) 
	{	fprintf(stderr, "\nCan't start MATLAB engine\n");
		return EXIT_FAILURE;
	}

    engOutputBuffer(ep, buffer, BUFSIZE);

	ulaz = mxCreateDoubleMatrix(3, 3, mxREAL); // definiramo double matricu 3x3

    p = mxGetPr(ulaz); // dobavljamo pointer na (realne) vrijednosti matrice
	for(i=0; i<9; i++) // i punimo istu po redu
	    *(p + i) = (double) i+1;
	
	engPutVariable(ep, "X", ulaz);  // saljemo je u MATLAB engine

	engEvalString(ep, "X"); // MATLAB sprema matrice PO STUPCIMA, A NE PO RETCIMA KAO U C-U!
	
    printf("Kako MATLAB vidi matricu:\n%s",buffer);
    fgetc(stdin);
	
    engEvalString(ep, "load matrica.txt"); // ucitamo matricu pomocu MATLABa
    engEvalString(ep, "Y = inv(matrica)"); // i izracunamo njenu inverziju

    izlaz = engGetVariable(ep, "Y"); // dohvacamo rezultat iz MATLAB okoline
	
    printf("MATLABov output za inverziju:\n %s",buffer);
    printf("A evo sto smo mi dobili:\n");
    p = mxGetPr(izlaz); // pointer na izlaznu matricu
    for(i=0; i<9; i++)
    {   printf("%6.4f  ",*(p+i));
    }
    
    fgetc(stdin);
	
	engClose(ep); // gasimo engine

    // na kraju: OBVEZATNO UNISTITI SVE MATLAB VARIJABLE! (OSLOBODITI MEMORIJU)
	mxDestroyArray(ulaz);
    mxDestroyArray(izlaz);
	
	return EXIT_SUCCESS;
}
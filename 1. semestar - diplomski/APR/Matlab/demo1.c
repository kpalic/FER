#include <stdlib.h>
#include "engine.h" // ovo je include fajl za MATLABov engine
#define  BUFSIZE 256

int main()

{
	Engine *ep; // pointer na MATLAB-ov engine
	char buffer[BUFSIZE];

	// pokretanje MATLAB engine-a, uz standardne provjere ispravnosti
	if (!(ep = engOpen("\0"))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		return EXIT_FAILURE;
	}

	engOutputBuffer(ep, buffer, BUFSIZE); // definiramo string varijablu za MATLABov output
	
	engEvalString(ep, "X = 2 + 2"); // saljemo string na izracunavanje

	printf("%s", buffer+2); // ispisujemo dobiveni rezultat

    fgetc(stdin);
	
	engClose(ep); // gasimo engine

	return EXIT_SUCCESS;
}








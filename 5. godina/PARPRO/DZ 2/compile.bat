@echo off
gcc -o connect4 connect4.c -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)
echo Compilation successful!
echo Running the program with mpiexec...
mpiexec -n 8 connect4.exe
if %errorlevel% neq 0 (
    echo Program execution failed!
    exit /b %errorlevel%
)
echo Program executed successfully!


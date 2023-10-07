#!/bin/bash

python myLab.py init masterPassword
python myLab.py put masterPassword www.index.hr 123456
python myLab.py put masterPassword www.index.hr abcdef
python myLab.py put wrongMasterPassword www.index.hr 123456
python myLab.py get masterPassword www.index.hr
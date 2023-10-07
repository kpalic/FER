#!/bin/bash

echo "bad addUser - password missmatch"
echo "$ ./usermgmt add Kristo"
python usermgmt.py add Kristo << EOF
password1
password2
EOF
echo

echo "good addUser"
echo "$ ./usermgmt add Kristo"
python usermgmt.py add Kristo << EOF
password123
password123
EOF
echo

echo "$ ./usermgmt passwd Kristo"
python usermgmt.py passwd Kristo << EOF
new_password1
new_password2
EOF
echo

echo "$ ./usermgmt passwd Kristo"
python usermgmt.py passwd Kristo << EOF
new_password123
new_password123
EOF
echo

echo "$ ./usermgmt forcepass Kristo"
python usermgmt.py forcepass Kristo
echo

echo "$ ./usermgmt del Kristo"
python usermgmt.py del Kristo
echo

echo "$ ./login Kristo"
python login.py Kristo << EOF
password123
EOF
echo

echo "$ ./login Kristo"
python login.py Kristo << EOF
new_password123
even_newer_password
even_newer_password
EOF
echo

echo "$ ./login Kristo"
python login.py Kristo << EOF
wrong_password
EOF
echo

echo "$ ./login Kristo"
python login.py Kristo << EOF
another_wrong_password
EOF
echo

echo "$ ./login Kristo"
python login.py Kristo << EOF
yet_another_wrong_password
EOF
echo

#!/bin/bash

kazalo=$1
nastavak=$2

echo "kazalo : $kazalo"
echo "nastavak : $nastavak"
pom="$(find $kazalo -name "*.$nastavak" -exec cat {} \; | wc -l)"
echo "$pom"
#! /bin/sh

for i in {1..10}
do
	        dir=$(printf "%0*d\n" 2 $i)
		        echo "Test $dir"

			        res=`./lab3.exe < test/test$dir/Test.in | diff -w test/test$dir/Test.out -`
				        if [ "$res" != "" ]
						        then
								                echo "FAIL"
										                echo $res
												        else
														                echo "OK"
																        fi
																done 

#! /bin/sh

for i in {1..20}
do
	        dir=$(printf "%0*d\n" 2 $i)
		        echo "Test $dir"

			        res=`./lab2.exe < test/test$dir/test.in | diff -w test/test$dir/test.out -`
				        if [ "$res" != "" ]
						        then
								                echo "FAIL"
										                echo $res
												        else
														                echo "OK"
																        fi
																done 

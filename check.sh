#!/usr/bin/env bash
if [ $(which nvcc | wc -l) -ne 1 ]; then
	echo "nvcc not in path"
	exit -1
fi
if [ $(nvcc -V | grep "release 9.0" | wc -l) -ne 1 ]; then
	echo "nvcc is not of version 9.0, pleaese install nvcc of version 9.0 to build"
	exit -2
fi
exit 0

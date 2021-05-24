# PythonCUDASNN
a SNN simulator writen in Python using CUDA 

# import information
it appears that tensorflow only works with cuda 10.1, to install that on ubuntu I used: 
https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1810&target_type=runfilelocal

It gave an error because the version of gcc I was using was newer than supported, to fix this there is a flag --override that fixes that issue

then, I ran into issues with tensorflow not being able to find the libcudart.so.10.1 file
there was a stack overflow link https://stackoverflow.com/questions/64141446/how-to-install-cuda-10-1-on-linux-ubuntu
and the solution was running: 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

This seemed to fix the problems that I was running into.

There was also an issue that needed the Cudnn installation 
Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1
https://developer.nvidia.com/rdp/cudnn-archive
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

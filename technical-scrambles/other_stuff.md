# apt-get

to remove
sudo apt-get remove --purge \<package name\>
sudo apt-get autoremove
ref:  
https://askubuntu.com/questions/187888/what-is-the-correct-way-to-completely-remove-an-application

# subl

### key-binding
<pre>
[
	{
		"keys": ["shift+space"], "command": "move", "args": {"by": "characters", "forward": true}
	}
]
</pre>
can skip out of parentheses

(don't copy)
https://forum.sublimetext.com/t/dont-copy-when-no-text-is-highlighted/9806/3


## yolov3 on conda environment with no sudo
solved
conda search opencv   
conda install opencv=3.1.0 <-- after many trials, this one works  
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/yolofire/lib/

and change makefile:  
COMMON+= -DGPU -I/opt/cuda/include/ (the place where cuda/include/ directory really was)


ref:  
https://medium.com/@WhoYoung99/install-opencv-2-4-13-on-conda-virtual-env-cffdbde27bf4
http://answers.opencv.org/question/27114/error-while-loading-shared-libraries-libopencv_coreso30/

https://github.com/pjreddie/darknet/issues/553
https://kb.iu.edu/d/abbe

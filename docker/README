
Docker build of jupyterlab with Aero design software installed:

Xfoil -  https://web.mit.edu/drela/Public/web/xfoil/
   compiled with intel fortran compiler.  compiler docker image is 6gb so make sure you have space
   I tried the gnu fortran compiler but compiled program was unstable.
AVL -  https://web.mit.edu/drela/Public/web/avl/
   compiled with intel fortran compiler
AeroSandbox - latest released version
(more softwares can be added)

Building
--------
docker build -t aerodesign ./


Running
-------

Prerequisite: docker  For example:  https://docs.docker.com/docker-for-windows/install/

docker run -p 8888:8888 aerodesign

When running, you will see output on how to connect to the jupyerlab using a browser.  It will look something like:

http://127.0.0.1:8888/lab?token=c9118b1c6cc5453ca28c559f85a1aa1dc19b7998c41594b8

In the root directory, there is a jupyter notebook called aerodesign that shows some examples:
- Create and draw 3d geometry of a plane
- Draw a 2d airfoil
- calculate polars for a 2D airfoil using Xfoil and graph them

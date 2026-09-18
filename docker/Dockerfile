# install xfoil  (taken from https://github.com/thomaseizinger/docker-xfoil/)
# and modified to use the intel compiler
FROM intel/oneapi-hpckit:latest as xfoil
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y libx11-dev
RUN wget https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz -O /tmp/xfoil6.99.tgz
RUN tar -xzvf /tmp/xfoil6.99.tgz -C /
WORKDIR /Xfoil
COPY xfoil.patch .
RUN git apply xfoil.patch
WORKDIR /Xfoil/orrs/bin
RUN make osgen
RUN make osmap.o
WORKDIR /Xfoil/orrs
RUN bin/osgen osmaps_ns.lst
WORKDIR /Xfoil/plotlib
RUN make libPlt_gSP.a
WORKDIR /Xfoil/bin
RUN make xfoil
RUN make pplot
RUN make pxplot
WORKDIR /Xfoil

# install avl
FROM intel/oneapi-hpckit:latest as avl
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y libx11-dev
RUN wget https://web.mit.edu/drela/Public/web/avl/avl3.36.tgz -O /tmp/avl3.36.tgz
RUN tar -xzvf /tmp/avl3.36.tgz -C /
WORKDIR /Avl
RUN cp eispack/Makefile.ifort eispack/Makefile
RUN cp bin/Makefile.ifort bin/Makefile
WORKDIR /Avl/plotlib
RUN make ifort
WORKDIR /Avl/eispack
RUN make
WORKDIR /Avl/bin
RUN make


#
# base image
#
FROM continuumio/anaconda3
EXPOSE 8888

# copy xfoil binaries
COPY --from=xfoil /usr/local/bin/xfoil /usr/bin
COPY --from=xfoil /usr/local/bin/pplot /usr/bin
COPY --from=xfoil /usr/local/bin/pxplot /usr/bin

# copy avl binaries
COPY --from=avl /Avl/bin/avl /usr/bin

# set LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"


# install openscad
#RUN apt-get update && apt-get install -y \
#   openscad

# install AeroSandbox latest releaseed version.
RUN pip install AeroSandbox

# copy ipython notebook
COPY aerodesign.ipynb /

# add command
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]

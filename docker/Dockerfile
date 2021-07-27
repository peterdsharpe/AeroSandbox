# install xfoil  (taken from https://github.com/thomaseizinger/docker-xfoil/)
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

#
# base image
#
FROM continuumio/anaconda3
EXPOSE 8888
# copy xfoil binaries
COPY --from=xfoil /usr/local/bin/xfoil /usr/bin
COPY --from=xfoil /usr/local/bin/pplot /usr/bin
COPY --from=xfoil /usr/local/bin/pxplot /usr/bin
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"

# install AeroSandbox latest releaseed version.
RUN pip install AeroSandbox

# copy ipython notebook
COPY aerodesign.ipynb /

# add command
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]

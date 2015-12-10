MXNET_DIR = /home/antonio/git/mxnet

mxnet_predict: mxnet_predict.o
	g++ -O3 -o mxnet_predict mxnet_predict.o `pkg-config --libs opencv` $(MXNET_DIR)/lib/libmxnet.so -lopenblas

mxnet_predict.o: mxnet_predict.cc
	g++ -O3 -c mxnet_predict.cc -Wall -I$(MXNET_DIR)/include `pkg-config --cflags opencv` -std=c++11
	
clean: 
	rm mxnet_predict

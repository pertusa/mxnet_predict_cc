MXNET_DIR = /home/antonio/git/mxnet

mxnet_predict: mxnet_predict.o
	g++ -O3 -o mxnet_predict mxnet_predict.o `pkg-config --libs opencv` $(MXNET_DIR)/lib/libmxnet.so
#static:g++ -O3 -o mxnet_predict mxnet_predict.o `pkg-config --libs opencv` -Wl,--whole-archive $(MXNET_DIR)/lib/libmxnet.a $(MXNET_DIR)/dmlc-core/libdmlc.a -Wl,--no-whole-archive 

mxnet_predict.o: mxnet_predict.cc
	g++ -O3 -c mxnet_predict.cc -Wall -I$(MXNET_DIR)/include `pkg-config --cflags opencv` -std=c++0x

clean: 
	rm mxnet_predict.o
	rm mxnet_predict

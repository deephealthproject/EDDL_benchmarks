/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <eddl/apis/eddl.h>
#include <chrono>

using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic CNN for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {
	
	//Check memory use flags
	string mem("low_mem");
	if(argc>1){
		std::string flag(argv[1]);
		if(!flag.compare("-FULL")){
			cout << "Using full memory" << endl;
			mem = "full_mem";
		}
		else if(!flag.compare("-LOW")){
			cout << "Using low memory" << endl;
			mem = "low_mem";
		}
	}
	//Check device flags
	auto device = CS_GPU({1}, mem);
	if(argc>1){
		std::string flag(argv[1]);
		if(!flag.compare("-GPU")){
			cout << "Compiling for GPU" << endl;
			device = CS_GPU({1}, mem);
		}
		else if(!flag.compare("-CPU")){
			cout << "Compiling for CPU" << endl;
			device = CS_CPU(-1, mem);
		}
	}

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 1;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Reshape(l,{1,28,28});
    l = MaxPool(ReLu(Conv(l,32, {3,3},{1,1})),{3,3}, {1,1}, "same");
    l = MaxPool(ReLu(Conv(l,64, {3,3},{1,1})),{2,2}, {2,2}, "same");
    l = MaxPool(ReLu(Conv(l,128,{3,3},{1,1})),{3,3}, {2,2}, "none");
    l = MaxPool(ReLu(Conv(l,256,{3,3},{1,1})),{2,2}, {2,2}, "none");
    l = Reshape(l,{-1});

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}, "full_mem") // one GPU
          //CS_CPU(-1, "full_mem") // CPU with maximum threads availables
		  device
    );

    // View model
    summary(net);

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    auto start = std::chrono::high_resolution_clock::now();
    evaluate(net, {x_test}, {y_test});
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Inference elapsed time: " << elapsed.count() << " s\n";
}

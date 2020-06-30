Change the following lines:

	// Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");

	// Preprocessing
	eddlT::div_(x_train, 255.0);
    eddlT::div_(x_test, 255.0);

For:
	

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);


Also change in the headers:


#include "apis/eddl.h"
#include "apis/eddlT.h"

For:


#include <eddl/apis/eddl.h>

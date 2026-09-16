## Table of Contents
- [Background](#background)
- [Dependencies](#depedencies)
    - [Core](#core)
    - [Optional](#optional)
- [Building & Using](#building--using)
    - [Building](#building)
        - [Short Version](#short-version)
        - [Step by Step Guide](#step-by-step-guide)
    - [Using](#using)
- [Documentation](#documentation)
    - [Usage](#usage)
    - [Generation](#generation)
- [Notes](#notes)
    - [Compatible Image Types](#compatible-image-types)
    - [Enabling Generation of Test Files](#enabling-generation-of-test-files)
- [Additional Tools & Techniques Used](#additional-tools--techniques-used)

## Background
This is a multilayer perceptron classification neural network with an input layer, an output layer, and three hidden layers. It was trained on images from the MNIST (Modified National Institute Standards & Technology) database, which has 60,000 28 by 28 white text on black background handwritten images, each containing a number. The input layer contains a 784 by 128 matrix, the hidden layers contain 128 by 128 matrices, and the output layer contains a 10 by 128 matrix. It uses ReLU as the activation function between layers, softmax as the activation function for the output layer, a cross entropy loss function to determine loss, and stochastic gradient descent to descend the gradients to minimize loss. Learning rates from 1e-1 to 1e-3 were used to find minimums.

## Depedencies
### Core
- C++17 or newer
- CMake
- GNU Make
- Eigen
- stb-image and stb-image-resize2. It should be noted that these are included in the project already

### Optional
- Doxygen
- GoogleTest

## Building & Using
### Building
#### Short Version
```sh
mkdir build
cd build
cmake -S .. -B .
make
cd ../bin/interface
```

#### Step by Step Guide
1. Clone the project where you please.
2. `cd` into the project root.
3. Run `mkdir build`. This creates a `build` file in the project root directory.
4. Run `cd build`. This will let you enter the `build` directory.
5. Run `cmake -S .. -B .`. This will generate a make file in your current directory, which is `build`, while starting at the layer above, `..`, to search for source files.
6. Run `make` while still inside `build`. This will generate the binaries.
7. Run `cd ../bin/interface` or navigate at your own pace. There you will find the binaries.

### Using
The following assumes you are at `./bin/interface`. 

1. Place a compatible image type in the current directory. There is more information on compatible image types in a section below.
2. Run `./console`.
3. Follow the instructions. Input the path to the image relative to the current directory.
4. Observe the percentages.

## Documentation
### Usage
From root, run `open docs/html/index.html`.

### Generation
Clone the repository and from root, run:
```sh
cd docs
doxygen doxyfile
```

## Notes
### Compatible Image Types
The MNIST database features images with white text on black backgrounds. Most search engines show black images on white backgrounds by default. Please note this descrepency when using the model.

Furthermore, this neural network will automatically resize images to fit a 28x28 pixel square, so you may feed images larger or smaller than that as you wish.

### Enabling Generation of Test Files
Inclusion of the testing directory is disabled by default. This can be renabled at one's leisure by uncommenting the appropriate line ./CMakeLists.txt.

## Additional Tools & Techniques Used
- LLDB
- Test driven development
- Unit testing

## Table of Contents
- [[##Dependencies|Dependencies]]
    - [[###Core|Core]]
    - [[###Optional|Optional]]
- [[]]


## Depedencies
### Core
- C++17 or newer
- CMake
- GNU Make
- Eigen
- stb-image and stb-image-resize2. It should be noted that these are included in the project already

### Optional
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

## Notes
### Compatible Image Types
The MNIST database features images with white text on black backgrounds. Most search engines show black images on white backgrounds by default. Please note this descrepency when using the model.

Furthermore, this neural network will automatically resize images to fit a 28x28 pixel square, so you may feed images larger or smaller than that as you wish.

### Enabling Generation of Test Files
Inclusion of the testing directory is disabled by default. This can be renabled at one's leisure by uncommenting the appropriate line ./CMakeLists.txt.

## Additional Tools Used
- LLDB
- Test driven development


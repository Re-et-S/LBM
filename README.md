# Simple model for training on a corpus of MIDI files

This is a personal learning project parsing MIDI files and training a model with chained LSTM and transformer. The model was adapted from a different project taking floating numbers as input. I also took the opportunity to learn a bit of Rust. The MIDI parser, K-means clustering for quantizing the music notes onto a grid, and byte pair encoding are all implemented in Rust. Additionally I was playing with emacs org mode literate programming and its interaction with a coding agent, and as a result, the notebook.org file is treated as the single source of truth for the Rust code, and Rust code files are tangled from the org file. 

I download the midi files from [this great website](https://www.kunstderfuge.com/midi.htm) 

## Building/Running the project

The Rust portion can be run simply with in the LBM_rust folder, where the midi files should also reside. 

``` sh
# parse and run thee parsing and bpe pipeline 
cargo run --bin parse_midi_bpe

# decode a sample sequence with generated vocabulary
cargo run --bin decode -- <sequence.bin> [vocab.bin]
```

The C/C++ CUDA portion uses standard CMake workflows. In the LBM_rust folder

``` sh
mkdir build
cd build
# Configure 
cmake ..

# parallel compilation
make -j4
```

## Usage

To run tests and verify the configuration, model initialization and basic forward pass

``` sh
./build/lbm_test
```

To run the main training program

``` sh
./build/lbm_train

```

Additional AI-generated summary can be found in notebook_transformer.org 

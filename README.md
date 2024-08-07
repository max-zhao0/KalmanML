# MCTS Track Reconstruction

## Overview

This repository contains an implementation of a novel track reconstruction algorithm based on the [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) for the [TrackML detector](https://www.kaggle.com/competitions/trackml-particle-identification). Every script will have a block of code marked by `# BEGIN INPUT` and `# END INPUT`, which delineate the parts you need to modify, typically paths and parameters.

Johannes Wagner (UC Berkeley) \<johannes_wagner@berkeley.edu\>  
Max Zhao (UC Berkeley) \<zhaomax2@gmail.com\>

## Full Sequence

### 1. Generate a dataset

This project is designed to use data generated with [ACTS](https://acts.readthedocs.io/en/latest/). ACTS needs to be installed. Alternatively, you can skip this step and the following one and generate data with a method of your choosing in the same format. Modify `generate_pp.py`, putting your own ACTS installation in `actsdir` and the desired output directory where you want the datafiles to go in `datadir`.

## Appendix

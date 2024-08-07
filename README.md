# MCTS Track Reconstruction

## Overview

This repository contains an implementation of a novel track reconstruction algorithm based on the [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) for the [TrackML detector](https://www.kaggle.com/competitions/trackml-particle-identification). Every script will have a block of code marked by `# BEGIN INPUT` and `# END INPUT`, which delineate the parts you need to modify, typically paths and parameters.

### Authors

Johannes Wagner (UC Berkeley) \<johannes_wagner@berkeley.edu\>  
Max Zhao (UC Berkeley) \<zhaomax2@gmail.com\>

## Full Sequence

### 0. ACTS

This project is designed to use data generated with [ACTS](https://acts.readthedocs.io/en/latest/). ACTS needs to be installed or data should be created in the same format as those generated from scripts provided here.

### 1. Generate a dataset

Modify `generate_pp.py`, putting your own ACTS installation in `actsdir` and the desired output directory where you want the datafiles to go in `datadir`.

## Appendix

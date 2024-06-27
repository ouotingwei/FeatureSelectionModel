Author 1 : Ting-Wei, Ou @ NYCU M.S in Robotics
Author 2 : Jihn-Yang, Long @ NYCU M.S in Robotics

## Training

### Build the ORBSLAM2 features
```
cd extractors/orbslam2_features/
mkdir build
cd build
cmake -DPYTHON_LIBRARY=~/anaconda3/envs/featurebooster/lib/libpython3.8.so \-DPYTHON_INCLUDE_DIR=~/anaconda3/envs/featurebooster/include/python3.8 \-DPYTHON_EXECUTABLE=~/anaconda3/envs/featurebooster/bin/python3.8 ..
make -j
```

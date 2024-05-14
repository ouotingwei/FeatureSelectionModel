Author 1 : Tingweiou @ NYCU M.S in Robotics

## Training
- set up the env from -> https://github.com/SJTU-ViSYS/FeatureBooster.git

- activate the env
```
conda activate featurebooster
```

- shut down the env
```
conda deactivate featurebooster
```

### Build the ORBSLAM2 features
```
cd extractors/orbslam2_features/
mkdir build
cd build
cmake -DPYTHON_LIBRARY=~/anaconda3/envs/featurebooster/lib/libpython3.8.so \-DPYTHON_INCLUDE_DIR=~/anaconda3/envs/featurebooster/include/python3.8 \-DPYTHON_EXECUTABLE=~/anaconda3/envs/featurebooster/bin/python3.8 ..
make -j
```

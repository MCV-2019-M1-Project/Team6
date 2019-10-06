# Team 6

## How to run
First, the following python packages should be installed
- OpenCV
- Numpy
- Glob
- ml_metrics
- pyyaml

If you miss any of these packages, install it using pip. For example:

`$ pip install pyyaml`


The default configuration is sotred in the config.yml file. The following parameters can be tuned:

| Parameter |  Possible values |
| ------------ | ------------ |
| Number of histogram bins (NBINS) | [1, 256]  |
| Color space (colorspace)   | HSV, YUV, LAB  |
| Distance metric (dist)  | euclidean, chisq, hellinger  |
| Background removal method (bgrm) | 1 or 2  |
| Query set (queryset) | qsd1_w1, qst1_w1, ...|

Once the parameters are configurated, run the main script.

`$ python main.py`

###End

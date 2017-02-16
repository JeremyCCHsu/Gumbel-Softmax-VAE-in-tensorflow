# Categorical VAE (using Gumbel-Softmax approximation) in Tensorflow
Implementation (with modifications) of [*Categorical Reparameterization 
with Gumbel-Softmax*](https://arxiv.org/abs/1611.01144)  
Modifications:
  1. Batch Norm
  2. ConvNet specifications
  3. alpha value
  4. (more?)

<br/>
<br/>

## Semi-supervised learning for MNIST dataset
Classification results on the test set
Error rate: ~ 3 - 4%  
Confusion matrix:  

|       |   0|   1|   2|  3|   4|   5|   6|   7|   8|   9|
|-------|----|----|----|---|----|----|----|----|----|----|
| **0** | 971|   0|   2|  0|   0|   1|   1|   1|   4|   0|
| **1** |   0|1112|  12|  0|   0|   4|   3|   1|   3|   0|
| **2** |   1|   0|1005|  1|   3|   0|   0|   7|  14|   1|
| **3** |   1|   0|  23|930|   1|  36|   0|   4|  15|   0|
| **4** |   1|   0|   7|  0| 950|   0|   2|   0|   2|  20|
| **5** |   6|   0|   2| 13|   0| 853|   2|   0|  14|   2|
| **6** |  11|   1|   3|  0|   1|   7| 931|   0|   3|   1|
| **7** |   2|   2|  23|  0|   9|   1|   0| 980|   2|   9|
| **8** |   4|   0|   8|  2|   3|   2|   1|   2| 950|   2|
| **9** |   6|   0|   5| 20|   8|   1|   0|   4|   9| 956|
<br/>
<br/>


### 100 labeled images
<img src="imgs/x_labeled.png" />  
Ten images per class.  
<br/>
<br/>

### Style Change
<img src="imgs/Ep-200-conv.png" />  
Row: same style across 10 classes.  
<br/>
<br/>

### Reconstruction
<img src="imgs/Ep-200-reconst.png" />  
Reconstruction using the inferred class label.  
 - Left: Input (Ground-truth)
 - Middle: using inferred dense label.
 - Right: using inferred discrete label.
<br/>
<br/>

### Environment and Dependency
- Linux Ubuntu 16.04
- Python 3.5
- Tensorflow 0.12
- Matplotlib
- sklearn
- Numpy
- Json

<br/>
<br/>

### Usage
Git clone this repo.  
Download and unzip MNIST to a sub-folder `dataset`  
Specify your configurations in `architecture.json`, and execute  
```bash
python train.py
```

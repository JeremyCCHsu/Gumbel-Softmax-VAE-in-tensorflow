# Categorical VAE (using Gumbel-Softmax approximation) in Tensorflow
(Adapted version) Semi-supervised learning part of the [*Categorical Reparameterization 
with Gumbel-Softmax*](https://arxiv.org/abs/1611.01144)  
Modifications are list as follows:
  1. Batch Norm
  2. ConvNet specifications
  3. alpha value
  4. temperature:  
  	Eric's: tau = max(0.5, exp(-r\*t)), t is step, r = {1e-5, 1e-4}  
  	Mine: tau = tau0 + (1 - tau0) exp(-r*t), t is epoch, r ~ 2.7e-4  
  4. (more?)
<br/>
<br/>


## Semi-supervised learning for MNIST dataset
Classification results on the test set
Error rate: ~ 3 - 8% (depending on the configs)  
Confusion matrix:  

|       |   0|   1|   2|  3|   4|   5|   6|   7|   8|   9|
|-------|----|----|----|---|----|----|----|----|----|----|
| **0** | 969|   0|   1|  0|   0|   0|   3|   1|   5|   1|
| **1** |   0|1101|  10|  1|   1|   3|   2|   4|  12|   1|
| **2** |   1|   0| 991|  3|   3|   1|   2|  14|  14|   3|
| **3** |   0|   0|   3|957|   0|  34|   0|   4|  10|   2|
| **4** |   0|   0|   2|  0| 940|   0|   6|   1|   1|  32|
| **5** |   3|   0|   0|  1|   0| 869|   2|   0|  14|   3|
| **6** |   3|   1|   1|  0|   2|   7| 923|   0|  21|   0|
| **7** |   0|   0|   7|  1|   1|   0|   0| 997|   3|  19|
| **8** |   3|   0|   2|  4|   0|   2|   2|   3| 950|   8|
| **9** |   2|   2|   1|  8|   1|   3|   1|   5|  12| 974|  
Accuracy: 9671/10000 = **96.71%**
<br/>
<br/>


### Labeled images (from the Training set)
<img src="imgs/x_labeled.png" width=400 />  
Ten images per class.  
<br/>
<br/>

### Style Change (Testing set)
<img src="imgs/Ep-200-conv.png" width=400 />  
Row: same style across 10 classes.  
Style is obtained from the diagonal image.
<br/>
<br/>

### Reconstruction (Testing set)
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

### Usage
```bash
pip install -r requirements.txt
git clone https://github.com/JeremyCCHsu/Gumbel-Softmax-VAE-in-tensorflow.git
cd Gumbel-Softmax-VAE-in-tensorflow
mkdir dataset
cd dataset
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -o train-images-idx3-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -o train-labels-idx1-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -o t10k-images-idx3-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -o t10k-labels-idx1-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
cd ..
python train.py
```  
The outputs will be in `./tmp`  

Or equivalently, `git clone` this repo.  
Download and unzip MNIST to a sub-folder `dataset`  
Specify your configurations in `architecture.json`, and execute `python train.py`  
<br/>

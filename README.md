# Git Loss For Deep Face Recognition
This repository contains code for my paper "Git Loss for Deep Face Recognition". 
</br>The preprint of the paper can be found at: <a href="https://arxiv.org/pdf/1807.08512.pdf">HERE</a>

## Usage
Standard Gitloss methodology:
<img src="results/push-pull-distance-1.png"/>

Following dependencies are required for the code:

<ol>
  <li>Tensorflow >= 1.4 </li>
  <li>Numpy</li>
  <li>Matplotlib</li>
</ol>
This code is for MNIST (mentioned in paper). For VGGFace2, you may use the code provided by <a href="https://github.com/davidsandberg/facenet">facenet</a>. Use the function `get_git_loss()` from the `gitloss.py` in facenet to use VGGFace2. 

To run the code: `python3 gitloss.py`

## Results 
Following are the plots of Gitloss and Centerloss [1].
### Gitloss: 
<img src="results/git-loss-lc001-lg01.png" height="400" width="400"/>

### Centerloss: 
<img src="results/center-loss-lc001-lg0.png" height="400" width="400"/>

## VGGFace2 
TODO

## Citation
If you use the loss function described, please cite the paper. Please note that this is the preprint version, the published version will be out soon.

```
@article{calefati2018git,
  title={Git Loss for Deep Face Recognition},
  author={Calefati, Alessandro and Janjua, Muhammad Kamran and Nawaz, Shah and Gallo, Ignazio},
  journal={arXiv preprint arXiv:1807.08512},
  year={2018}
}
```

## Contact
If there is any problem with the code or if you may have any question, feel free to open an issue or reach out here: mjanjua.bscs16seecs@seecs.edu.pk

## References
[1] https://ydwen.github.io/papers/WenECCV16.pdf

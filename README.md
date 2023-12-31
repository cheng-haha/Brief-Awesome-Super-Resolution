# Brief-Awesome-Super-Resolution
AAAI, CVPR, ICCV, ICLR, ICML, NIPS, MM, ECCV (22-23)

## _This project is continuously being updated..._
## Datasets
### Super Resolution Dataset

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution#link-of-datasets).

|     Name     |   Usage    |                             Link                             |                        Comments                        |
| :----------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------: |
|  UHDSR4K     | Train/Test |     [website(ICCV2021)](https://github.com/HDCVLab/Benchmarking-Ultra-High-Definition-Image-Super-resolution)     |                       Ultra-High-Definition 4K dataset                        |
|  UHDSR8K     | Train/Test |     [website(ICCV2021)](https://github.com/HDCVLab/Benchmarking-Ultra-High-Definition-Image-Super-resolution)     |                       Ultra-High-Definition 8K dataset                        |
|     Set5     |    Test    | [download](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|    SET14     |    Test    | [download](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|    BSD100    |    Test    | [download](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|   Urban100   |    Test    | [download](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|   Manga109   |    Test    |       [website](http://www.manga109.org/ja/index.html)       |                                                        |
|   SunHay80   |    Test    | [download](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|    BSD300    | Train/Val  | [download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz) |                                                        |
|    BSD500    | Train/Val  | [download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) |                                                        |
|   91-Image   |   Train    | [download](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar) |                          Yang                          |
|  DIV2K2017   | Train/Val  |     [website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)     |                       NTIRE2017                        |
|  Flickr2K   |   Train  |     [download](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)     |     |
|   Real SR    | Train/Val  | [website](https://drive.google.com/file/d/1Iqx3AbUlLjR_JglsQIq9y9BEcrNLcOCU/view) |    NTIRE2019                        |
|   Waterloo   |   Train    |   [website](https://ece.uwaterloo.ca/~k29ma/exploration/)    |                                                        |
|     VID4     |    Test    | [download](https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip) |                        4 videos                        |
|    MCL-V     |   Train    |        [website](http://mcl.usc.edu/mcl-v-database/)         |                       12 videos                        |
|    GOPRO     | Train/Val  | [website](https://github.com/SeungjunNah/DeepDeblur_release) |                   33 videos, deblur                    |
|    CelebA    |   Train    | [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  |                      Human faces                       |
|    Sintel    | Train/Val  |       [website](http://sintel.is.tue.mpg.de/downloads)       |                      Optical flow                      |
| FlyingChairs |   Train    | [website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) |                      Optical flow                      |
|  Vimeo-90k   | Train/Test |           [website](http://toflow.csail.mit.edu/)            |                     90k HQ videos                      |
|  SR-RAW   | Train/Test |           [website](https://ceciliavision.github.io/project-pages/project-zoom.html)            |                      raw sensor image dataset                      |
|  W2S   | Train/Test |           [arxiv](https://arxiv.org/pdf/2003.05961.pdf)            |     A Joint Denoising and Super-Resolution Dataset                      |
|  PIPAL   | Test |           [ECCV 2020](https://arxiv.org/pdf/2007.12142.pdf)            |     Perceptual Image Quality Assessment dataset                  |
|  HQ-50K   | Train |           [website](https://huggingface.co/datasets/YangQiee/HQ-50K)            |  50,000 images  |

#### Dataset collections

[Benckmark and DIV2K](https://drive.google.com/drive/folders/1-99XFJs_fvQ2wFdxXrnJFcRRyPJYKN0K): Set5, Set14, B100, Urban100, Manga109, DIV2K2017 include bicubic downsamples with x2,3,4,8

[SR_testing_datasets](https://www.kaggle.com/msahebi/super-resolution#SR_testing_datasets.zip): Test: Set5, Set14, B100, Urban100, Manga109, Historical; Train: T91,General100, BSDS200

## Papers
### AAAI
#### 22
[Feature Distillation Interaction Weighting Network for Lightweight Image Super-resolution](https://ojs.aaai.org/index.php/AAAI/article/view/19946)<br />[Best-Buddy GANs for Highly Detailed Image Super-resolution](https://ojs.aaai.org/index.php/AAAI/article/view/20030)<br />[Detail-Preserving Transformer for Light Field Image Super-resolution](https://ojs.aaai.org/index.php/AAAI/article/view/20153)<br />[Efficient Non-local Contrastive Attention for Image Super-resolution](https://ojs.aaai.org/index.php/AAAI/article/view/20179)<br />[Text Gestalt: Stroke-Aware Scene Text Image Super-resolution](https://ojs.aaai.org/index.php/AAAI/article/view/19904)<br />[SFSRNet: Super-resolution for Single-Channel Audio Source Separation](https://ojs.aaai.org/index.php/AAAI/article/view/21372)<br />[Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-Based Super-resolution](https://ojs.aaai.org/index.php/AAAI/article/view/20180)<br />[SCSNet: An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-resolution](https://ojs.aaai.org/index.php/AAAI/article/view/20236)
### CVPR
#### 22
BasicVSR++: Improving Video Super-Resolution With Enhanced Propagation and Alignment<br />Discrete Cosine Transform Network for Guided Depth Map Super-Resolution<br />VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution<br />Spatial-Temporal Space Hand-in-Hand: Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning<br />Look Back and Forth: Video Super-Resolution With Explicit Temporal Difference Modeling<br />RSTT: Real-Time Spatial Temporal Transformer for Space-Time Video Super-Resolution<br />GCFSR: A Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors<br />Reference-Based Video Super-Resolution Using Multi-Camera Video Triplets<br />Learning Graph Regularisation for Guided Super-Resolution<br />Reflash Dropout in Image Super-Resolution<br />Learning Trajectory-Aware Transformer for Video Super-Resolution<br />Memory-Augmented Non-Local Attention for Video Super-Resolution<br />Transformer-Empowered Multi-Scale Contextual Matching and Aggregation for Multi-Contrast MRI Super-Resolution<br />Blind Image Super-Resolution With Elaborate Degradation Modeling on Noise and Kernel<br />Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites<br />A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-Resolution<br />Learning the Degradation Distribution for Blind Image Super-Resolution<br />SphereSR: 360deg Image Super-Resolution With Arbitrary Projection via Continuous Spherical Image Representation<br />Investigating Tradeoffs in Real-World Video Super-Resolution<br />MNSRNet: Multimodal Transformer Network for 3D Surface Super-Resolution<br />Stable Long-Term Recurrent Video Super-Resolution<br />Dual Adversarial Adaptation for Cross-Device Real-World Image Super-Resolution<br />Deep Constrained Least Squares for Blind Image Super-Resolution<br />Task Decoupled Framework for Reference-Based Super-Resolution<br />LAR-SR: A Local Autoregressive Model for Image Super-Resolution<br />Texture-Based Error Analysis for Image Super-Resolution<br />Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution
#### 23
Towards High-Quality and Efficient Video Super-Resolution via Spatial-Temporal Data Overfitting<br />Spectral Bayesian Uncertainty for Image Super-Resolution<br />RefSR-NeRF: Towards High Fidelity and Super Resolution View Synthesis<br />Cross-Guided Optimization of Radiance Fields With Multi-View Image Super-Resolution for High-Resolution Novel View Synthesis<br />Compression-Aware Video Super-Resolution<br />Guided Depth Super-Resolution by Deep Anisotropic Diffusion<br />CutMIB: Boosting Light Field Super-Resolution via Multi-View Image Blending<br />Image Super-Resolution Using T-Tetromino Pixels<br />Toward Accurate Post-Training Quantization for Image Super Resolution<br />Implicit Diffusion Models for Continuous Super-Resolution<br />Correspondence Transformers With Asymmetric Feature Learning and Matching Flow Super-Resolution<br />Rethinking Image Super Resolution From Long-Tailed Distribution Learning Perspective<br />Azimuth Super-Resolution for FMCW Radar in Autonomous Driving<br />Spatial-Frequency Mutual Learning for Face Super-Resolution<br />B-Spline Texture Coefficients Estimator for Screen Content Image Super-Resolution<br />CABM: Content-Aware Bit Mapping for Single Image Super-Resolution Network With Large Input<br />Memory-Friendly Scalable Super-Resolution via Rewinding Lottery Ticket Hypothesis<br />Structured Sparsity Learning for Efficient Video Super-Resolution<br />Omni Aggregation Networks for Lightweight Image Super-Resolution<br />OSRT: Omnidirectional Image Super-Resolution With Distortion-Aware Transformer<br />Human Guided Ground-Truth Generation for Realistic Image Super-Resolution<br />Learning Generative Structure Prior for Blind Text Image Super-Resolution<br />Super-Resolution Neural Operator<br />OPE-SR: Orthogonal Position Encoding for Designing a Parameter-Free Upsampling Module in Arbitrary-Scale Image Super-Resolution<br />Equivalent Transformation and Dual Stream Network Construction for Mobile Image Super-Resolution<br />Activating More Pixels in Image Super-Resolution Transformer<br />Better "CMOS" Produces Clearer Images: Learning Space-Variant Blur Estimation for Blind Image Super-Resolution<br />Cascaded Local Implicit Transformer for Arbitrary-Scale Super-Resolution<br />Deep Arbitrary-Scale Image Super-Resolution via Scale-Equivariance Pursuit<br />N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution<br />Perception-Oriented Single Image Super-Resolution Using Optimal Objective Estimation<br />Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution<br />Toward Stable, Interpretable, and Lightweight Hyperspectral Super-Resolution<br />CiaoSR: Continuous Implicit Attention-in-Attention Network for Arbitrary-Scale Image Super-Resolution<br />Consistent Direct Time-of-Flight Video Depth Super-Resolution<br />Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution<br />Zero-Shot Dual-Lens Super-Resolution

### MM
#### 22
Cross-Modality High-Frequency Transformer for MR Image Super-Resolution<br />RepSR: Training Efficient VGG-style Super-Resolution Networks with Structural Re-Parameterization and Batch Normalization<br />Quality Assessment of Image Super-Resolution: Balancing Deterministic and Statistical Fidelity<br />You Only Align Once: Bidirectional Interaction for Spatial-Temporal Video Super-Resolution<br />Real-World Blind Super-Resolution via Feature Matching with Implicit High-Resolution Priors<br />Joint Learning Content and Degradation Aware Feature for Blind Super-Resolution<br />Model-Guided Multi-Contrast Deep Unfolding Network for MRI Super-resolution Reconstruction<br />Learning Generalizable Latent Representations for Novel Degradations in Super-Resolution<br />Adjustable Memory-efficient Image Super-resolution via Individual Kernel Sparsity<br />OISSR: Optical Image Stabilization Based Super Resolution on Smartphone Cameras<br />Sophon: Super-Resolution Enhanced 360° Video Streaming with Visual Saliency-aware Prefetch<br />Rethinking Super-Resolution as Text-Guided Details Generation<br />Flexible Hybrid Lenses Light Field Super-Resolution using Layered Refinement<br />Geometry-Aware Reference Synthesis for Multi-View Image Super-Resolution<br />ICNet: Joint Alignment and Reconstruction via Iterative Collaboration for Video Super-Resolution
### ICCV
#### 23
On the Effectiveness of Spectral Discriminators for Perceptual Quality Improvement<br />SRFormer: Permuted Self-Attention for Single Image Super-Resolution<br />Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution<br />DLGSANet: Lightweight Dynamic Local and Global Self-Attention Network for Image Super-Resolution<br />Boosting Single Image Super-Resolution via Partial Channel Shifting<br />Dual Aggregation Transformer for Image Super-Resolution<br />Feature Modulation Transformer: Cross-Refinement of Global Representation via High-Frequency Prior for Image Super-Resolution<br />MetaF2N: Blind Image Super-Resolution by Learning Efficient Model Adaptation from Faces<br />Spherical Space Feature Decomposition for Guided Depth Map Super-Resolution<br />Real-CE: A Benchmark for Chinese-English Scene Text Image Super-resolution<br />MoTIF: Learning Motion Trajectories with Local Implicit Neural Functions for Continuous Space-Time Video Super-Resolution

### ECCV
#### 22
Efficient Long-Range Attention Network for Image Super-Resolution<br />A Codec Information Assisted Framework for Efficient Compressed Video Super-Resolution<br />Metric Learning Based Interactive Modulation for Real-World Super-Resolution<br />Learning Series-Parallel Lookup Tables for Efficient Image Super-Resolution

### ICLR
#### 22
Learning Efficient Image Super-Resolution Networks via Structure-Regularized Pruning
#### 23
Forward Super-Resolution: How Can GANs Learn Hierarchical Generative Models for Real-World Distributions<br />DySR: Adaptive Super-Resolution via Algorithm and System Co-design<br />Knowledge Distillation based Degradation Estimation for Blind Super-Resolution

### ICML
#### 22
MemSR: Training Memory-efficient Lightweight Model for Image Super-Resolution
#### 23
Learning Controllable Degradation for Real-World Super-Resolution via Constrained Flows <br />Crafting Training Degradation Distribution for the Accuracy-Generalization Trade-off in Real-World Super-Resolution <br />DeSRA: Detect and Delete the Artifacts of GAN-based Real-World Super-Resolution Models

### NIPS
#### 22
ShuffleMixer: An Efficient ConvNet for Image Super-Resolution<br />AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos<br />Rethinking Alignment in Video Super-Resolution Transformers


### TPAMI
#### 22
Densely Residual Laplacian Super-Resolution<br />A Progressive Fusion Generative Adversarial Network for Realistic and Consistent Video Super-Resolution<br />Deep Spatial-Angular Regularization for Light Field Imaging, Denoising, and Super-Resolution<br />Exploiting Raw Images for Real-Scene Super-Resolution<br />RankSRGAN: Super Resolution Generative Adversarial Networks With Learning to Rank<br />Erratum to "Deep Back-Projection Networks for Single Image Super-Resolution"


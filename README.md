# Universal-description-of-the-NS-surface-using-ML

Neutron stars provide an ideal theoretical framework for exploring fundamental physics when nuclear matter surpasses densities encountered within atomic nuclei. Despite their paramount importance, uncertainties in the equation of state (EoS) have shrouded their internal structure. For rotating neutron stars, the shape of their surface is contingent upon the EoS and the rotational dynamics. This work proposes new universal relations regarding the star's surface, employing machine-learning techniques for regression. More specifically, we developed highly accurate universal relations for a neutron star's eccentricity, the star's ratio of the polar to the equatorial radius, and the effective gravitational acceleration at both the pole and the equator. Furthermore, we propose an accurate theoretical formula for $(d\log R(\mu)/d\theta)_{\max}$. This research addresses key astronomical aspects by utilizing these global parameters as features for the training phase of a neural network. Along the way, we introduce new effective parameterizations for each star's global surface characteristics. Our regression methodology enables accurate estimations of the star's surface $R(\mu)$, its corresponding logarithmic derivative $d\log R(\mu)/d\theta$, and its effective acceleration due to gravity $g(\mu)$ with accuracy better than 1 %. The analysis is performed for an extended sample of rotating configurations constructed using a large ensemble of 70 tabulated hadronic, hyperonic, and hybrid EoS models that obey the current multimessenger constraints and cover a wide range of stiffnesses. Above all, the suggested relations could provide an accurate framework for the star's surface estimation using data acquired from the NICER X-ray telescope or future missions, and constrain the EoS of nuclear matter when measurements of the relevant observables become available.

### Summary of the EoS-insensitive relations investigated in this work

| Universal Relation | Parameters and their respective ranges | Equation | Max % Error |
|:--------:|:---------------------------:|:---------------------------:|:-------------:|
| $e(C,\sigma)$ | $C\in[0.0876, 0.3075], \sigma \in [0.0328,0.9612]$| Improved Fit | 4.57  |
| $g_{\mathrm{pole}}(C,\sigma)$ | $C\in[0.0876,0.3095], \sigma \in [0.0000,0.9612]$ | Improved Fit | 3.07  |
| $\mathcal{R(C,\sigma)}$ | $C\in[0.0876,0.3095], \sigma \in [0.0000,0.9612]$  | New Fit | 2.79  |
| $(d \log R(\mu) / d \theta)_{\mathrm{max}} (C,\sigma,\mathcal{R})$ | $C\in[0.0876, 0.3075], \sigma \in [0.0328,0.9612],$ <br> $\mathcal{R}\in[0.626,0.981]$ | New Fit | 3.21  |
| $g_{\mathrm{eq}}(C,\sigma,e)$ | $C\in[0.0876,0.3095], \sigma \in [0.0000,0.9612],$  <br> $e\in[0.000,0.780]$| New Fit | 4.26  |
| $R(\mu;R_{\mathrm{pole}}, R_{\mathrm{eq}},C,\sigma,e)$ | $R_{\mathrm{pole}} \in [8.618,14.161]  \mathrm{km}, R_{\mathrm{eq}} \in [9.683,19.413] \mathrm{km},$ <br> $C\in[0.0876,0.3095], \sigma \in [0.0000,0.9612],$  <br> $e\in[0.000,0.780]$| New Fit | 0.25  |
| $g(\mu;g_{\mathrm{pole}}, g_{\mathrm{eq}},C,\sigma,e)$ | $g_{\mathrm{pole}}/g_0 \in[0.987,2.107], g_{\mathrm{eq}}/g_0 \in [0.069,1.000],$  <br> $C\in[0.0876,0.3095],  \sigma \in [0.0000,0.9612],$  <br> $e\in[0.000,0.780]$ | New Fit | 0.91  |

| Universal Relation | Parameters and their respective ranges | Equation | Max Residual |
|:--------:|:---------------------------:|:---------------------------:|:-------------:|
| $\left(\frac{d \log R(\mu)}{d\theta} \right)\left(\mu;(d \log R(\mu) / d \theta)_{\mathrm{max}},C,\sigma,\mathcal{R}\right)$ | $(d \log R(\mu) / d \theta)_{\mathrm{max}} \in [0.019,0.503],$ <br> $C\in[0.0876, 0.3075], \sigma \in [0.0328,0.9612],$ <br> $\mathcal{R}\in[0.626,0.981]$ | New Fit | $8.36\times 10^{-3}$ |


### Universal relations of the Neutron Star’s surface key global EoS-independent parameterizations investigated in this work are defined by the following analytical expressions:
| Rp/Req: $\mathcal{R(C,\sigma)}$| Star's eccentricity: $e(C,\sigma)$| Max of Logaritmic derivative: $\left(\frac{d \log R(\mu)}{d\theta}\right)_{\mathrm{max}}$|
|:--------:|:---------------------------:|:-------------:|
| ![Formula 1](https://latex.codecogs.com/svg.latex?\mathcal{R}(C,\sigma)=\sum_{n=0}^{4}\sum_{m=0}^{4-n}\hat{\mathcal{A}}_{nm}%20C^n%20\sigma^m) | ![Formula 2](https://latex.codecogs.com/svg.latex?e(C,\sigma)=\sum_{n=0}^{5}\sum_{m=0}^{5-n}\hat{\mathcal{B}}_{nm}%20C^n%20\sigma^m) | ![Formula 3](https://latex.codecogs.com/svg.latex?\left(\frac{d%20\log%20R(\mu)}{d%20\theta}\right)_{\mathrm{max}}=%20\sum_{n=0}^{3}%20\sum_{m=0}^{3-n}%20\sum_{q=0}^{3-(n+m)}\hat{\mathcal{C}}_{nmq}%20C^n%20\sigma^m%20\mathcal{R}^q) ||



<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Universal Relations/R_C_sigma_2.png" alt="Figure 1" width="350"></td>
      <td><img src="Figures/Universal Relations/e_C_sigma.png" alt="Figure 2" width="350"></td>
      <td><img src="Figures/Universal Relations/log_der_max.png" alt="Figure 3" width="350"></td>
    </tr>
    </table>
</div>

|Effective gravity at pole:  $g_{\mathrm{pole}}(C,\sigma)$| Effective gravity at equator: $g_{\mathrm{eq}}(C,\sigma,e)$| 
|:--------:|:---------------------------:|
| ![Formula 4](https://latex.codecogs.com/svg.latex?g_{\mathrm{pole}}(C,\sigma)=g_0\sum_{n=0}^{4}\sum_{m=0}^{4-n}\hat{\mathcal{D}}_{nm}%20C^n%20\sigma^m) | ![Formula 5](https://latex.codecogs.com/svg.latex?g_{\mathrm{eq}}(C,\sigma,e)=g_0\sum_{n=0}^{3}\sum_{m=0}^{3-n}\sum_{q=0}^{3-(n+m)}\hat{\mathcal{E}}_{nmq}%20C^n%20\sigma^m%20e^q) | 

<div style="text-align: center;">
  <table>        
    <tr>
      <td><img src="Figures/Universal Relations/g_p_C_sigma.png" alt="Figure 4" width="350"></td>
      <td><img src="Figures/Universal Relations/g_e_C_segma_e.png" alt="Figure 5" width="350"></td>
    </tr>
  </table>
</div>


### Effective universal normalization for the star's surface
<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Surface/R(mu)_min-max_universal_representation.png" alt="Figure 6" width="500"></td>
      <td><img src="Figures/Surface/R(mu)_min-max_universal_representation_2.png" alt="Figure 7" width="500"></td>
    </tr>
  </table>
</div>

The investigated new EoS-insensitive relation for the star's surface is given by: 

![Formula](https://latex.codecogs.com/svg.latex?\Large%20R(\mu)%20=%20R_{\mathrm{pole}}%20+%20(R_{\mathrm{eq}}-R_{\mathrm{pole}})%20\hat{F}_{\theta^{\star}}(|\mu|,%20C,%20\sigma,%20e))


### Indicative surfaces for a catalog of NS benchmark models presented in the paper
<img src="Figures/Surface/fits_panel_1.png" alt="Figure 8" width="1000">

<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Surface/surf_model_1_fit.png" alt="Figure 9" width="350"></td>
      <td><img src="Figures/Surface/surf_model_2_fit.png" alt="Figure 10" width="350"></td>
      <td><img src="Figures/Surface/surf_model_3_fit.png" alt="Figure 11" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Surface/surf_model_4_fit.png" alt="Figure 12" width="350"></td>
      <td><img src="Figures/Surface/surf_model_5_fit.png" alt="Figure 13" width="350"></td>
      <td><img src="Figures/Surface/surf_model_6_fit.png" alt="Figure 14" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Surface/surf_model_7_fit.png" alt="Figure 15" width="350"></td>
      <td><img src="Figures/Surface/surf_model_8_fit.png" alt="Figure 16" width="350"></td>
      <td><img src="Figures/Surface/surf_model_9_fit.png" alt="Figure 17" width="350"></td>
    </tr>
  </table>
</div>

### Effective universal normalization for the surface's angular derivative
<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Logarithmic Derivative/dlogR(mu)_min-max_universal_representation.png" alt="Figure 18" width="500"></td>
      <td><img src="Figures/Logarithmic Derivative/dlogR(mu)_min-max_universal_representation_2.png" alt="Figure 19" width="500"></td>
    </tr>
  </table>
</div>

The investigated new EoS-insensitive relation for the logarithmic derivative at the star's surface is given by: 

![Formula](https://latex.codecogs.com/svg.latex?\Large%20\left(\frac{d\log%20R(\mu)}{d\theta}\right)%20=%20\left(\frac{d\log%20R(\mu)}{d\theta}\right)_{\mathrm{max}}%20\hat{\mathcal{F}}_{\theta^{\star}}(\mu,%20C,%20\sigma,%20\mathcal{R}))



### Indicative curves associated with the surface's logarithmic derivative for a catalog of NS benchmark models presented in the paper
<img src="Figures/Logarithmic Derivative/fits_panel_2b.png" alt="Figure 20" width="1000">

<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Logarithmic Derivative/der_model_2_fit.png" alt="Figure 21" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_3_fit.png" alt="Figure 22" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_4_fit.png" alt="Figure 23" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Logarithmic Derivative/der_model_5_fit.png" alt="Figure 24" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_6_fit.png" alt="Figure 25" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_7_fit.png" alt="Figure 26" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Logarithmic Derivative/der_model_8_fit.png" alt="Figure 27" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_9_fit.png" alt="Figure 28" width="350"></td>
    </tr>
  </table>
</div>

### Universal normalization for the NS surface's effective acceleration due to gravity
<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Effective Gravity/g(mu)_min-max_universal_representation.png" alt="Figure 29" width="500"></td>
      <td><img src="Figures/Effective Gravity/g(mu)_min-max_universal_representation_2.png" alt="Figure 30" width="500"></td>
    </tr>
  </table>
</div>

The investigated new EoS-insensitive relation for the star's effective gravity on the surface is given by: 

![Formula](https://latex.codecogs.com/svg.latex?\Large%20g(\mu)%20=%20g_{\mathrm{pole}}%20+%20(g_{\mathrm{eq}}%20-%20g_{\mathrm{pole}})%20\hat{\mathbb{F}}_{\theta^{\star}}(|\mu|,%20C,%20\sigma,%20e))


### Indicative curves associated with the surface's effective acceleration due to gravity for a catalog of NS benchmark models presented in the paper
<img src="Figures/Effective Gravity/fits_panel_3.png" alt="Figure 31" width="1000">

<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Effective Gravity/g_mu_model_1_fit.png" alt="Figure 32" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_2_fit.png" alt="Figure 33" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_3_fit.png" alt="Figure 34" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Effective Gravity/g_mu_model_4_fit.png" alt="Figure 35" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_5_fit.png" alt="Figure 36" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_6_fit.png" alt="Figure 37" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Effective Gravity/g_mu_model_7_fit.png" alt="Figure 38" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_8_fit.png" alt="Figure 39" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_9_fit.png" alt="Figure 40" width="350"></td>
    </tr>
  </table>
</div>

### Repository Overview
This repository contains analytic expressions, the ANN model architecture, pre-trained model weights, and example notebooks that demonstrate how to use our fitting functions for predicting the neutron star surface properties discussed in the paper.

### Contents
* **Model Architecture**: The machine learning models developed for regression tasks.
* **Pre-trained Weights**: Ready-to-use model weights for reproducing our results.
* **Demos**: Sample scripts illustrating how to load the models and perform predictions on the data.

### Installation
To set up the necessary environment for running the code, ensure the following dependencies are installed:

```bash
pip install numpy pandas scikit-learn torch
```

### Dependencies
* `numpy`
* `pandas`
* `sklearn`
* `pytorch`

### Usage
1. **Loading the Pre-trained Model**: You can load the provided pre-trained model weights and use them to predict surface parameters of neutron stars. Check the demos folder for example usage.
```python
import torch
model = torch.load('path_to_model_weights.pth')
```
2. **Running a Demo**:These scripts demonstrate how estimate the neutron star properties related to the star's surface using universal relations.

### Contact
For any questions, feel free to contact us via email at gpapigki@auth.gr, g.vardakas@uoi.gr

### Citation
If you find this work useful in your research, please cite our paper as:

@article{papigkiotis2025universaldescriptionneutronstars,
      title={Universal description of the Neutron Star's surface and its key global properties: A Machine Learning Approach for nonrotating and rapidly rotating stellar models}, 
      author={Grigorios Papigkiotis and Georgios Vardakas and Aristidis Likas and Nikolaos Stergioulas},
      year={2025},
      eprint={2501.18544},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE},
      url={https://arxiv.org/abs/2501.18544}, 
}





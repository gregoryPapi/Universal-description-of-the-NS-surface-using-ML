# Universal-description-of-the-NS-surface-using-ML
* Neutron stars provide an ideal theoretical framework for exploring fundamental physics when nuclear matter surpasses densities encountered within atomic nuclei. Despite their paramount importance, uncertainties in the equation of state (EoS) have shrouded their internal structure. The star's shape morphology is contingent upon the EoS and the rotational dynamics. This work proposes new universal relations regarding the star's surface, employing advanced machine-learning techniques for regression. More specifically, we developed highly accurate universal relations for a neutron star's eccentricity, the star's ratio of the polar to the equatorial radius, and the effective gravitational acceleration at both the pole and the equator. Furthermore, we proposed a precise theoretical formula for $(d\log R(\mu)/d\theta)_{\max}$. This research addresses key astronomical aspects by utilizing these global parameters as features for the neural network training phase. Along the way, new effective parameterizations for each star's global surface characteristics have been introduced. The evaluated regression methodology enabled precise estimations of the star's surface $R(\mu)$, its corresponding logarithmic derivative $d\log R(\mu)/d\theta$, and its effective acceleration due to gravity $g(\mu)$ with accuracy better than $1 \%$. Thus, offering an adequate formulation that transcends the EoS dependencies. In addition, this study aligns with the burgeoning field of multimessenger astronomy, contributing to recent advancements in gravitational wave astronomy by incorporating machine learning methodologies. The analysis is performed for an extended sample of rotating configurations constructed using a complete ensemble of 70 tabulated hadronic, hyperonic, and hybrid EoS models that obey the multimessenger constraints and cover a wide range of stiffnesses. Above all, the suggested relations could provide an accurate framework for the star's surface estimation using data acquired from the NICER X-ray telescope or future missions, and constrain the EoS of nuclear matter when measurements of the relevant observables become available.

# Universal relations of the Neutron Star’s surface key global properties

# Effective universal normalization for the star's surface
<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Surface/R(mu)_min-max_universal_representation.png" alt="Figure 1" width="500"></td>
      <td><img src="Figures/Surface/R(mu)_min-max_universal_representation_2.png" alt="Figure 2" width="500"></td>
    </tr>
  </table>
</div>

The investigated new EoS-insensitive relation for the star's surface is given by: 

![Formula](https://latex.codecogs.com/svg.latex?\Large%20R(\mu)%20=%20R_{\mathrm{pole}}%20+%20(R_{\mathrm{eq}}-R_{\mathrm{pole}})%20\hat{F}_{\theta^{\star}}(|\mu|,%20C,%20\sigma,%20e))


# Indicative surfaces for a set of benchmark NS models
<img src="Figures/Surface/fits_panel_1.png" alt="Figure 3" width="1000">

<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Surface/surf_model_1_fit.png" alt="Figure 4" width="350"></td>
      <td><img src="Figures/Surface/surf_model_2_fit.png" alt="Figure 5" width="350"></td>
      <td><img src="Figures/Surface/surf_model_3_fit.png" alt="Figure 6" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Surface/surf_model_4_fit.png" alt="Figure 7" width="350"></td>
      <td><img src="Figures/Surface/surf_model_5_fit.png" alt="Figure 8" width="350"></td>
      <td><img src="Figures/Surface/surf_model_6_fit.png" alt="Figure 9" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Surface/surf_model_7_fit.png" alt="Figure 10" width="350"></td>
      <td><img src="Figures/Surface/surf_model_8_fit.png" alt="Figure 11" width="350"></td>
      <td><img src="Figures/Surface/surf_model_9_fit.png" alt="Figure 12" width="350"></td>
    </tr>
  </table>
</div>

# Effective universal normalization for the surface's angular derivative
<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Logarithmic Derivative/dlogR(mu)_min-max_universal_representation.png" alt="Figure 13" width="500"></td>
      <td><img src="Figures/Logarithmic Derivative/dlogR(mu)_min-max_universal_representation_2.png" alt="Figure 14" width="500"></td>
    </tr>
  </table>
</div>

The investigated new EoS-insensitive relation for the star's surface logarithmic derivative is given by: 

![Formula](https://latex.codecogs.com/svg.latex?\Large%20\left(\frac{d\log%20R(\mu)}{d\theta}\right)%20=%20\left(\frac{d\log%20R(\mu)}{d\theta}\right)_{\mathrm{max}}%20\hat{\mathcal{F}}_{\theta^{\star}}(\mu,%20C,%20\sigma,%20\mathcal{R}))



# Indicative curves associated with the surface's logarithmic derivative for a set of NS benchmark models 
<img src="Figures/Logarithmic Derivative/fits_panel_2b.png" alt="Figure 15" width="1000">

<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Logarithmic Derivative/der_model_2_fit.png" alt="Figure 16" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_3_fit.png" alt="Figure 17" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_4_fit.png" alt="Figure 18" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Logarithmic Derivative/der_model_5_fit.png" alt="Figure 19" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_6_fit.png" alt="Figure 20" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_7_fit.png" alt="Figure 21" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Logarithmic Derivative/der_model_8_fit.png" alt="Figure 22" width="350"></td>
      <td><img src="Figures/Logarithmic Derivative/der_model_9_fit.png" alt="Figure 23" width="350"></td>
    </tr>
  </table>
</div>

# Universal normalization for the NS surface's effective acceleration due to gravity
<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Effective Gravity/g(mu)_min-max_universal_representation.png" alt="Figure 24" width="500"></td>
      <td><img src="Figures/Effective Gravity/g(mu)_min-max_universal_representation_2.png" alt="Figure 25" width="500"></td>
    </tr>
  </table>
</div>

The investigated new EoS-insensitive relation for the star's effective gravity on surface is given by: 

![Formula](https://latex.codecogs.com/svg.latex?\Large%20g(\mu)%20=%20g_{\mathrm{pole}}%20+%20(g_{\mathrm{eq}}%20-%20g_{\mathrm{pole}})%20\hat{\mathbb{F}}_{\theta^{\star}}(|\mu|,%20C,%20\sigma,%20e))


# Indicative curves associated with the surface's effective acceleration due to gravity for a set of NS benchmark models 
<img src="Figures/Effective Gravity/fits_panel_3.png" alt="Figure 26" width="1000">

<div style="text-align: center;">
  <table>
    <tr>
      <td><img src="Figures/Effective Gravity/g_mu_model_1_fit.png" alt="Figure 27" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_2_fit.png" alt="Figure 28" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_3_fit.png" alt="Figure 29" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Effective Gravity/g_mu_model_4_fit.png" alt="Figure 30" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_5_fit.png" alt="Figure 31" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_6_fit.png" alt="Figure 32" width="350"></td>
    </tr>
    <tr>
      <td><img src="Figures/Effective Gravity/g_mu_model_7_fit.png" alt="Figure 33" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_8_fit.png" alt="Figure 34" width="350"></td>
      <td><img src="Figures/Effective Gravity/g_mu_model_9_fit.png" alt="Figure 35" width="350"></td>
    </tr>
  </table>
</div>

# Repository Overview
This repository contains the model architecture, pre-trained model weights, and example demos that demonstrate the usage of our code for predicting neutron star surface properties using machine learning.

# Contents
* **Model Architecture**: The machine learning models developed for the regression tasks.
* **Pre-trained Weights**: Ready-to-use model weights for reproducing our results.
* **Demos**: Sample scripts illustrating how to load the models and perform predictions on new data.

# Installation
To set up the necessary environment for running the code, ensure the following dependencies are installed:

```bash
pip install numpy pandas scikit-learn torch
```

# Dependencies
* `numpy`
* `pandas`
* `sklearn`
* `pytorch`

# Usage
1. **Loading the Pre-trained Model**: You can load the provided pre-trained model weights and use them to predict surface parameters of neutron stars. Check the demos folder for example usage.
```python
import torch
model = torch.load('path_to_model_weights.pth')
```
2. **Running a Demo**: Example demo scripts can be found in the ... directory. These scripts demonstrate how to predict neutron star properties, such as eccentricity and gravitational acceleration.

# Contact
For any questions, data access, or further collaborations, feel free to contact us via email at ....

# Citation
If you find this work useful in your research, please cite our paper: ...


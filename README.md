<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />

  <h1 align="center">Cross-information Neural Network for Low-velocity Impact 
Damage Prediction in Laminated Composite</h1>

  <p align="center">
    Dazhi Zhao<sup>1§</sup>, Yinglong Liu<sup>1§</sup>, Yujie Xiang<sup>1</sup>, Peng Zhang<sup>1</sup>, Ning Liu1<sup>*</sup>, Keke Tang<sup>1,2*</sup> <br>
<sup>1</sup>School of Aerospace Engineering and Applied Mechanics, Tongji University, Shanghai, 200092, China <br>
<sup>2</sup>Key Laboratory of AI-aided Airworthiness of Civil Aircraft Structures, Civil Aviation Administration of China, Tongji University, Shanghai, 200092, China <br>
§  Dazhi Zhao and Yinglong Liu contributed equally to the work <br>

<!-- ABOUT THE PROJECT -->
## Abstract

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Laminated composite plates exhibit complex failure mechanisms under low-velocity impacts that are computationally demanding to analyze. To overcome these limitations while accurately evaluating the damage morphology, we generate a comprehensive computational dataset (10,240 samples) through Abaqus/Explicit simulations, incorporating variations in material properties and ply configurations to characterize damage area evolution under different impact conditions. Building on this foundation, we propose the Cross-information Low-velocity Impact Damage Prediction Network (CliDPN), a neural network-based surrogate model that integrates external impact dynamics (impact velocity direction) and internal laminate characteristics (ply angles, material types) for efficient prediction. The framework's dual-branch architecture consists of: (1) a Damage Image Reconstructor (DIR) for high-fidelity damage visualization, and (2) a Parameter-to-Image Predictor (PIP) based on an improved VQ-VAE algorithm with hybrid MLP-CNN modules. By leveraging deep convolutional networks, this ensembled system reaches up to RMSE 0.0393, LPIPS 0.0325 and SSIM 0.9682 across the full damage severity spectrum while maintaining 5ms/inference computational efficiency, demonstrating superior performance over conventional FE-based methods. Results demonstrate consistent performance in both intra-material laminates through intra-material generalization and diverse material systems via inter-material adaptability. Ablation studies confirm the CNN-Based Information Enhancer's critical role in enhancing predictive accuracy, achieving a maximum accuracy improvement of 56.5% with this module. Additionally, the cross-information fusion mechanism proves essential for predicting subtle damage features. Our proposed CliDPN enables accurate and efficient impact damage assessment in laminated composites, demonstrating robust inter-material adaptability while facilitating maintenance decision-support and impact-resistant design optimization.



<!-- GETTING STARTED -->
## Environment Setup

Clone this repository to your local machine, and install the dependencies.
  ```sh
  git clone git@github.com:dazhizhao/CliDPN.git 
  pip install -r requirements.txt
  ```
## Dataset & Checkpoints
You can find the all the dataset and checkpoints we adopted in this paper from [Google Drive](https://drive.google.com/file/d/1Y51FmLvJPXxGFEZudxF8U85QQQTxxpRG/view?usp=drive_link).

## Usage
### Training Stage
To train the Damage Image Reconstructor (DIR), you can run this code:
```sh
  python dir.py
  ```
To train the Parameter-to-Image Predictor (PIP), you can run this code:
```sh
  python pip.py
  ```
### Inference Stage
For inference, it must be pointed that you should pay attention to the checkpoints path and the corresponding dataset format. Meanwhile, the impact information should be input in the `eval.py`.
After inputting the information, you will get an image of damage in milliseconds, which is far faster than the Finite Element simulation.</br>
To leverge the whole ensembled deep learning model CliDPN for inference, you can run this code:
```sh
  python eval.py
  ```

## License
This project is licensed under the MIT License, see the LICENSE file for details.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/fig1.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

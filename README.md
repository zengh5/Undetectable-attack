The code is reproducing for our paper "Towards undetectable adversarial examples: a steganographic perspective"

## Function
We borrow the embedding suitability map from steganography to modulate the adversarial perturbation. The obtained AEs are hard to detect against statistical detectors.

## Usage 
Run Attack_IFGSM_u (or Attack_TDI_u) to compare AEs with (left)/without (right) the proposed scheme.

![image](https://github.com/zengh5/Undetectable-attack/blob/main/advimgs_un/ifgsm_UNIWARD/0c7ac4a8c9dfa802.png)
![image](https://github.com/zengh5/Undetectable-attack/blob/main/advimgs_un/ifgsm/0c7ac4a8c9dfa802.png)

## Extention
Here we use S-UNIWARD [1] to generate the embedding suitability maps. You are welcomed to try other steganography methods. This could be done easily by changing the dir to the alternative suitability maps:
                                               mat_path = './SuitabilityMap/S_UNIWARD/'

Here we use IFGSM, TDI-FGSM as baselines. You are welcomed to extend it to other advanced attacks: C&W, NI, PID, ...

[1] Holub, V., Fridrich, J. & Denemark, T. Universal distortion function for steganography in an arbitrary domain. EURASIP J. on Info. Security 2014

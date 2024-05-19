# Equivariant Diffusion for Molecule Generation in 3D

## Introduction

Generating 3D molecular structures that are both chemically valid and physically accurate is a critical challenge in computational chemistry and drug discovery. Traditional methods for molecular generation have relied on heuristic and rule-based approaches, which often lack the flexibility ad generalizability required to handle the vast diversity of molecular structures. 

One of the most promising developments in this filed is the application of diffusion models to molecule generation. DIffusion models (DMs) have shown remarkable success in generating high-quality data across various domains, including image synthesis and audio generation. These models operate by iteratively denoising a sample starting from pure noise, guided by a learned probability distribution that captures the underlying structure of the target data. Applying diffusion models to the generation of 3D molecular structures presents some unique challenges, primarily due to the need for the generated molecules to be invariant to Euclidean transformations such as translations, rotations and reflections.

The paper “Equivariant Diffusion for Molecule Generation in 3D” by Emiel Hoogeboom et al. introduces a novel approach to address these challenges. The authors propose the E(3) Equivariant Diffusion Model (EDM), which integrates E(3) equivariant neural networks into the diffusion framework, maintaining the physical properties and chemical validity of the generated molecules. 

## Related Work

## **Recap on Diffusion Models**

## Key Contributions

## Model Architecture

The architecture employs a U-Net-like structure, a common choice in diffusion models due to its effective handling of high-dimensional data. The U-Net consists of an encoder and a decoder, connected by a bottleneck layer that processes the data at multiple resolutions. The encoder progressively downsamples the input data, extracting hierarchical features that capture various levels of molecular detail. The decoder then upsamples these features, reconstructing the 3D molecular structure while ensuring that the output remains invariant to Euclidean transformations.

A key innovation in EDM is the integration of E(3) equivariant convolutional layers within the U-Net. These layers are designed to operate on the 3D coordinates and atom types simultaneously, ensuring that the learned features respect the symmetries of the underlying physical space. The equivariant layers perform convolutions that are invariant to rotations and translations, allowing the network to learn representations that are robust to such transformations.

## Strengths and Weaknesses

## Our Contribution

Our primary contribution is the re-implementation of the original EDM in the JAX/FLAX framework. JAX, with its ability to automatically differentiate through native Python and Numpy functions, and FLAX, which provides a high-level interface for neural network building, collectively offer significant advantages in terms of performance and flexibility.

By porting the model to JAX/FLAX, we aim to:

1. **Improve Computational Efficiency**: JAX’s just-in-time compilation and automatic vectorization capabilities can significantly speed up the training and inference processes.
2. **Enhance Scalability**: The new implementation can leverage distributed computing resources more effectively, allowing for training on larger datasets and more complex models.
3. **Facilitate Research and Development**: The modular and flexible nature of FLAX makes it easier for researchers to experiment with different model architectures and training regimes.

## Results

We are still in the process of running experiments to compare the performance of our JAX/FLAX implementation against the original PyTorch-based model. We plan to provide detailed evaluations in an accompanying notebook, which will include:

- **Comparative FID Scores**: Assessing the quality of generated molecular structures.
- **Training TIme Analysis**: Evaluating the efficiency gains achieved through the new implementation.
- **Scalability Tests**: Demonstrating the model’s performance on larger datasets and more complex molecules (?)

## Conclusion

In conclusion, the E(3) Equivariant Diffusion Model represents a significant advancement in the field of molecular generation, providing a robust framework for generating 3D molecules with high fidelity.

Our re-implementation in JAX/FLAX aims to further enhance the model’s efficiency and scalability, making it more accessible and practical for broader use in molecular sciences.

## Contributions

- **Harold**:
- **Marina**:
- **Ric**:
- **Alkis**:
#  Equivariant Diffusion for Molecule Generation in 3D 

*Date: 27/05/2024*      |      *Authors: 
Marina Orozco González, Harold Ruiter, Robin Reitsma*

---




## Section 1: Introduction

> Discovering 3D molecular structures for medicine or new materials has conventionally been done through human expertise. However, modern deep learning techniques have proven to be a more efficient approach.

Generating 3D molecular structures that are both chemically valid and physically accurate is a critical challenge in computational chemistry and drug discovery. Traditional methods for molecular generation have relied on heuristic and rule-based approaches, which often lack the flexibility ad generalizability required to handle the vast diversity of molecular structures. In the paper we based our project on, the use of a **3D Equivariant Difussion Model (EDM)** that has **Graph Neural Networks**, **Diffusion Models** and **E(3) equivariant** as **key components**, is introduced in order to leverage the symmetries to the Eucliden group in 3 dimensions that molecules have, outperforming previous 3D molecular generative models.

![Generated molecule from the original model](imgs_blogpost\gen_molecule.png)

### Section 1.2: Motivation
The discovery of novel molecules is a crucial step in the development of new products that can be used for drug discovery for the treatment of complex diseases, developing material science and other molecular design related fields.  But finding new chemical compounds with desired properties is a challenging task, as the space of synthesizable molecules is vast and sear in it proves to be very difficult, mostly owing to its discrete nature (Cao and Kipf, 2022). Although conventional molecular design involves the use of human expertise to propose, synthesize and test new molecules, this process can be cost and time intensive, limiting the number and diversity of molecules that can be reasonably explored (Bilodeau et al., 2022). In deed, it has been estimated that the number of compounds that have ever been synthesized lies around $10^8$ while the total number of theoretically feasible compounds lies between $10^{23}$ and $10^{60}$ (Polishchuk et al., 2013).

In light of this situation, modern deep learning techniques provides an alternative and more efficient approach to achieve this through generative models. There are two characteristics of deep learning that makes it particularly promising when applied to molecules: First, they can cope with “unstructured” data representations such as text sequences, speech signals, images and graphs. Second, deep learning can perform feature extraction from the input data, that is, produce data-driven features from the input data without the need for manual intervention ([Atz et al., 2021](https://arxiv.org/pdf/2107.12375)).

## Section 2: Keypoints
### Section 2.1: Introduction to Graphs
Concerning the use of deep learning techniques for molecules generation, according with Maziarz et al. (2022), early approaches relied on the textual SMILES representation and on the reuse of architectures from natural language processing. 

However, since 2018 we can observe the use of **Graph Neural Networks (GNN)** for this goal, as stated in Cao and Kipf (2022), due to the great improvements that were made on graphs in the area of deep learning during 2017 (Bronstein et al., 2017). Using graphs is convenient for this goal as this representation is able to encapsulate crucial structural information for molecules generation. Moreover, is has the benefit that all generated outputs are valid graphs (but not necessarily valid molecules).

![Exemplary molecular representations for a selected molecule a. Two-dimensional depiction b. Molecular graph (2D) composed of vertices (atoms) and edges (bonds). (Atz et al., 2021)](imgs_blogpost\graph_mol.png)

### Section 2.2: Introduction to  Diffusion Models
As we said, the paper we used as baseline, Hoogeboom et al. (2022), combines the use of **GNN** with ****Diffusion Models (DMs)**,** standing as the first Diffusion Model that generates molecules in 3D space. 

The use of Diffusion Models have gained more attention since the introduction of the Diffusion Probabilistic Models (DPMs) framework by Dhariwal and Nichol (2021) in 2020. They have shown remarkable success in generating high-quality data across various domains, including image synthesis and audio generation. 

These models operate by iteratively denoising a sample starting from pure noise, guided by a learned probability distribution that captures the underlying structure of the target data. 

In contrast to other generative models, in diffusion models the generative process is defined with respect to the _true denoising process_, which is known after modelling the reverse process of diffusion to a certain given data point **$x_0$**.

![Picture from](imgs_blogpost\diffusion_cat.png)

### Section 2.3: Introduction to Equivariance transformations

A function f is said to be equivariant to the action of a group
G if:

$$T_g(f(\mathbf{x}))=f(S_g(\mathbf{x})), \forall g \in G$$

Where $S_g, T_g$ are linear representations related to the group element $g$ ([Serre, 1977](https://link.springer.com/book/10.1007/978-1-4684-9458-7)). In this work, we only consider the Eucliden group E(3) generated by translations, rotations and reflections. In this case, become a translation $\mathbf{t}$ and an orthogonal matrix $\mathbf{R}$ that rotates or reflects coordinates. We will have then that the function $f$ is equivariant to a rotatio or reflection $\mathbf{R}$ if:

$$\mathbf{R} f(\mathbf{x})=f(\mathbf{Rx})$$

The reason why using an equivariance approach for molecule generation is convenient lies on the inherent symmetries of molecular structures to those transformations. By leveraging symmetries, equivariant models can reduce redundant computations. For instance, once a particular arrangement is learned, its symmetric counterparts are automatically accounted for, which can lead to more efficient training and inference processes.

### 2.4: EDM: E(3) Equivariance Diffusion Model for molecules generation

In this work interactions between all atoms are considered and model through a fully connected graph $\mathcal{G}$ with nodes $v_i \in \mathcal{V}$. Each node $v_i$ has associated a coordinate representation $\mathbf{x}_i \in \mathbb{R}^3$ and an attribute vector $\mathbf{h}_i \in \mathbb{R}^d$.

The EGNN architecture is composed of L EGCL layers wich applies the non-linear transformation:

$$ \hat{\mathbf{x}}_h, \hat{\mathbf{h}} = EGNN[\mathbf{x}^0, \mathbf{h}^0 ]$$

Which satisfies the required equivariant property we saw before in \ref{nosé}.

### Section 2.5: Introduction to JAX
#### Section 2.5.1: JAX
Hola buenas 

#### Section 2.5.2: JIT paradigm
Hola buenas 

#### Section 2.5.3: XLA
Hola buenas 

## Section 3: Out contribution

Our primary contribution is the re-implementation of the original EDM from [_link to original paper_] in the JAX/FLAX framework. JAX, with its ability to automatically differentiate through native Python and Numpy functions, and FLAX, which provides a high-level interface for neural network building, collectively offer significant advantages in terms of performance and flexibility.

By porting the model to JAX/FLAX, we aim to:

1. **Improve Computational Efficiency**: JAX’s just-in-time compilation and automatic vectorization capabilities can significantly speed up the training and inference processes.
2. **Enhance Scalability**: The new implementation can leverage distributed computing resources more effectively, allowing for training on larger datasets and more complex models.
3. **Facilitate Research and Development**: The modular and flexible nature of FLAX makes it easier for researchers to experiment with different model architectures and training regimes.

## Section 4: Results

We are still in the process of running experiments to compare the performance of our JAX/FLAX implementation against the original PyTorch-based model. We plan to provide detailed evaluations in an accompanying notebook, which will include:

- **Comparative FID Scores**: Assessing the quality of generated molecular structures.
- **Training TIme Analysis**: Evaluating the efficiency gains achieved through the new implementation.
- **Scalability Tests**: Demonstrating the model’s performance on larger datasets and more complex molecules (?)

## Section 5: Conclusion

In conclusion, the E(3) Equivariant Diffusion Model represents a significant advancement in the field of molecular generation, providing a robust framework for generating 3D molecules with high fidelity.

Our re-implementation in JAX/FLAX aims to further enhance the model’s efficiency and scalability, making it more accessible and practical for broader use in molecular sciences.

## Contributions

- **Harold**:
- **Marina**:
- **Ric**:
- **Alkis**:
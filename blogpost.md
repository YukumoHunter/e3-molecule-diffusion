#  Equivariant Diffusion for Molecule Generation in 3D 

*Date: 27/05/2024*      |      *Authors: 
Marina Orozco González, Harold Ruiter, Robin Reitsma, Ricardo Chávez Torres*

---




## Introduction

> Discovering 3D molecular structures for medicine or new materials has conventionally been done through human expertise. However, modern deep learning techniques have proven to be a more efficient approach.

Generating 3D molecular structures that are both chemically valid and physically accurate is a critical challenge in computational chemistry and drug discovery. Traditional methods for molecular generation have relied on heuristic and rule-based approaches, which often lack the flexibility ad generalizability required to handle the vast diversity of molecular structures. In the paper we based our project on, the use of a **3D Equivariant Diffusion Model (EDM)** that has **Graph Neural Networks**, **Diffusion Models** and **E(3) equivariant** as **key components**, is introduced in order to leverage the symmetries to the Euclidean group in 3 dimensions that molecules have, outperforming previous 3D molecular generative models.

![Generated molecule from the original model](img/gen_molecule.gif) 

### Motivation
The discovery of novel molecules is a crucial step in the development of new products that can be used for drug discovery for the treatment of complex diseases, developing material science and other molecular design related fields.  But finding new chemical compounds with desired properties is a challenging task, as the space of synthesizable molecules is vast and sear in it proves to be very difficult, mostly owing to its discrete nature [(Cao and Kipf, 2022)$ ^{[1]}$](#1--cao-n-d-and-kipf-t-2022-molgan-an-implicit-generative-model-for-small-molecular-graphs). Although conventional molecular design involves the use of human expertise to propose, synthesize and test new molecules, this process can be cost and time intensive, limiting the number and diversity of molecules that can be reasonably explored [(Bilodeau et al., 2022)$ ^{[2]}$](#2--bilodeau-c-jin-w-jaakkola-t-barzilay-r-and-jensen-k-f-2022-generative-models-for-molecular-discovery-recent-advances-and-challenges-wires-computational-molecular-science-125e1608). Indeed, it has been estimated that the number of compounds that have ever been synthesized lies around $10^8$ while the total number of theoretically feasible compounds lies between $10^{23}$ and $10^{60}$ (Polishchuk et al., 2013).

In light of this situation, modern deep learning techniques provides an alternative and more efficient approach to achieve this through generative models. There are two characteristics of deep learning that makes it particularly promising when applied to molecules: First, they can cope with “unstructured” data representations such as text sequences, speech signals, images and graphs. Second, deep learning can perform feature extraction from the input data, that is, produce data-driven features from the input data without the need for manual intervention [(Atz et al., 2021)$ ^{[3]}$](#3--kenneth-atz-and-francesca-grisoni-and-gisbert-schneider-2021-ggeometric-deep-learning-on-molecular-representations)

## Key points
### Introduction to Graphs
Concerning the use of deep learning techniques for molecules generation, according with [(Maziarz et al., 2022)$ ^{[4]}$](#4--maziarz-k-jackson-flux-h-cameron-p-sirockin-f-schneider-n-stiefl-n-segler-m-and-brockschmidt-m-2022-learning-to-extend-molecular-scaffolds-with-structural-motifs), early approaches relied on the textual SMILES representation and on the reuse of architectures from natural language processing. 

However, since 2018 we can observe the use of **Graph Neural Networks (GNN)** for this goal, as stated in [(Cao and Kipf, 2022)$ ^{[1]}$](#1--cao-n-d-and-kipf-t-2022-molgan-an-implicit-generative-model-for-small-molecular-graphs), due to the great improvements that were made on graphs in the area of deep learning during 2017 [(Bronstein et al., 2017)$^{[5]}$](#5-bronstein-m-m-bruna-j-lecun-y-szlam-a-and-vandergheynst-p-2017-geometric-deep-learning-going-beyond-euclidean-data-ieee-signal-processing-magazine-3441842). Using graphs is convenient for this goal as this representation is able to encapsulate crucial structural information for molecules generation. Moreover, is has the benefit that all generated outputs are valid graphs (but not necessarily valid molecules).

![Exemplary molecular representations for a selected molecule a. Two-dimensional depiction b. Molecular graph (2D) composed of vertices (atoms) and edges (bonds). (Atz et al., 2021)](img/graph_mol.png)

### Introduction to  Diffusion Models
As we said, the paper we used as baseline, [Hoogeboom et al. (2022)$^{[6]}$](#6-hoogeboom-e-satorras-v-g-vignac-c-and-welling-m-2022-equivariant-diffusion-for-molecule-generation-in-3d), combines the use of **GNN** with **Diffusion Models (DMs)**, standing as the first Diffusion Model that generates molecules in 3D space. 

The use of Diffusion Models have gained more attention since the introduction of the Diffusion Probabilistic Models (DPMs) framework by [Dhariwal and Nichol (2021)$^{[7]}$](#7-dhariwal-p-and-nichol-a-2021-diffusion-models-beat-gans-on-image-synthesis) in 2020. They have shown remarkable success in generating high-quality data across various domains, including image synthesis and audio generation. 

These models operate by iteratively denoising a sample starting from pure noise, guided by a learned probability distribution that captures the underlying structure of the target data. 

In contrast to other generative models, in diffusion models the generative process is defined with respect to the _true denoising process_, which is known after modelling the reverse process of diffusion to a certain given data point **$x_0$**.

![Picture from](img/diffusion_cat.png)

### Introduction to Equivariance transformations

A function f is said to be equivariant to the action of a group
G if:

$$T_g(f(\mathbf{x}))=f(S_g(\mathbf{x})), \forall g \in G$$

Where $S_g, T_g$ are linear representations related to the group element $g$ ([Serre (1977) $^{[8]}$](#8-j-p-serre-1977-linear-representations-of-finite-groups)). In this work, we only consider the Euclidean group E(3) generated by translations, rotations and reflections. In this case, become a translation $\mathbf{t}$ and an orthogonal matrix $\mathbf{R}$ that rotates or reflects coordinates. We will have then that the function $f$ is equivariant to a rotatio or reflection $\mathbf{R}$ if:

$$\mathbf{R} f(\mathbf{x})=f(\mathbf{Rx})$$

The reason why using an equivariance approach for molecule generation is convenient lies on the inherent symmetries of molecular structures to those transformations. By leveraging symmetries, equivariant models can reduce redundant computations. For instance, once a particular arrangement is learned, its symmetric counterparts are automatically accounted for, which can lead to more efficient training and inference processes.

### EDM: E(3) Equivariance Diffusion Model for molecules generation

In this work interactions between all atoms are considered and model through a fully connected graph $\mathcal{G}$ with nodes $v_i \in \mathcal{V}$. Each node $v_i$ has associated a coordinate representation $\mathbf{x}_i \in \mathbb{R}^3$ and an attribute vector $ \mathbf{h}_i \in \mathbb{R}^d$ .

The EGNN architecture is composed of L EGCL layers wich applies the non-linear transformation:

$$ \hat{\mathbf{x}}_h, \hat{\mathbf{h}} = EGNN[\mathbf{x}^0, \mathbf{h}^0 ] $$  

Which satisfies the required equivariant property we saw before in \ref{nosé}.

### Introduction to JAX

JAX is an open-source library for numerical computing that combines automatic differentiation with the capability to run code on GPUs and TPUs. Created by Google Research, JAX serves as a high-performance alternative to PyTorch and NumPy, enhanced by gradient-based optimization and just-in-time compilation.

JAX's key features make it particularly effective for machine learning:

1. **NumPy Compatibility:** JAX’s API is very similar to NumPy’s, making it easy for users to switch. Functions in JAX have the same names and signatures as those in NumPy.
2. **Automatic Differentiation:** JAX includes powerful tools for automatic differentiation. The 'grad' function helps compute gradients of scalar-valued functions, and the 'vjp' and 'jvp' functions handle Jacobian-vector and vector-Jacobian products.
3. **Accelerator Support:** JAX code can run on CPUs, GPUs, and TPUs with few changes, offering great versatility.
4. **Composable Function Transformations:** JAX supports composable transformations like 'grad' for gradients, 'jit' for just-in-time compilation, 'vmap' for vectorization, and 'pmap' for parallelization across devices.

#### JIT paradigm
Just-In-Time (JIT) compilation is a standout feature in JAX. It optimizes code by converting Python functions into efficient machine code, which can be run on accelerators.

- **JIT Usage:** In JAX, the '@jit' decorator marks functions for JIT compilation. When used, JAX employs the XLA compiler to turn the function into machine code. This involves tracing the function to create a computation graph, which XLA then compiles.
- **Performance Impact:** JIT compilation can greatly improve performance, especially for intensive tasks. By optimizing the computation graph and fusing operations, XLA cuts down on the overhead from Python’s dynamic nature and speeds up execution.

#### XLA
XLA (Accelerated Linear Algebra) is a specialized compiler that optimizes machine learning computations. It's crucial to JAX’s performance and offers several benefits:

- **Compilation Process:** XLA processes the computation graph from JAX and optimizes it through operation fusion, constant folding, and kernel generation. These optimizations reduce memory use and speed up execution.
- **Operation Fusion:** XLA’s operation fusion combines multiple operations into a single kernel, reducing memory accesses and boosting computational efficiency.

### Handling Graphs in JAX

JAX handles graphs efficiently despite their irregular data structures using several techniques:

1. **Padding Nodes and Edges:** To batch process graphs, they are padded to a uniform size by adding dummy nodes and edges, ensuring all graphs in a batch have the same number of nodes and edges. This creates consistent tensor shapes for GPU processing.
2. **Masking:** Masks, binary arrays indicating valid elements of padded tensors, ensure that padding doesn’t affect computation. Masks are applied to ignore padded elements during computation.
3. **Batch Processing:** With padding and masking, graphs can be processed in batches for efficient GPU computation. JAX’s 'vmap' operations enhance this by automatically applying operations across batches.

## Implementation

- **Generic Forward Pass:**
    - JAX:
        
        ```python
        # Define Model Parameters
        def init_params(layer_sizes, key):
            keys = random.split(key, len(layer_sizes))
            params = [(random.normal(k, (m, n)) * jnp.sqrt(2.0/m), jnp.zeros(n))
                      for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
            return params
        
        # Define Model Functions
        def relu(x):
            return jnp.maximum(0, x)
        
        def forward(params, x):
            activations = x
            for W, b in params[:-1]:
                activations = relu(jnp.dot(activations, W) + b)
            final_W, final_b = params[-1]
            logits = jnp.dot(activations, final_W) + final_b
            return logits
        
        # Compute Forward Pass
        x = random.normal(key, (1, 784))  # Example input: batch size of 1
        logits = forward(params, x)
        ```
        
    - PyTorch:
        
        ```python
        # Define Model Class
        	class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
        
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
         
        # Initialize Model and compute Forward Pass     
        model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
        
        x = torch.randn(1, 784)  # Example input: batch size of 1
        logits = model(x)
        ```
        
        JAX uses a functional programming style with pure functions, while PyTorch opts for a more object-oriented approach where models are instances of a nn.Module.
        
- **Backward pass:**
    - JAX:
        
        ```python
        # Define Loss Function
        def loss(params, x, y):
            logits = forward(params, x)
            return jnp.mean(jnp.square(logits - y))  # Mean squared error loss
        
        # Compute Gradients
        x = random.normal(key, (1, 784))  # Example input
        y = random.normal(key, (1, 10))   # Example target output
        
        grad_loss = grad(loss)
        gradients = grad_loss(params, x, y)
        ```
        
    - PyTorch:
        
        ```python
        # Defines Loss Function
        criterion = nn.MSELoss()  # Mean squared error loss
        
        # Compute Forward pass and Loss
        x = torch.randn(1, 784)  # Example input
        y = torch.randn(1, 10)   # Example target output
        
        logits = model(x)
        loss = criterion(logits, y)
        
        # Compute Gradients
        loss.backward()
        ```
        
        Both approaches automatically handle differentiation. Again JAX uses a more functional approach, as opposed to the OOP style of PyTorch
        
- **Graph Padding:**
    - JAX:
        
        ```python
        x_padded = jnp.pad(x, padding, mode='constant', constant_values=0)
        ```
        
    - PyTorch:
        
        ```python
        x_padded = F.pad(x, padding, mode='constant', value=0)
        ```
        
        Both approaches are similar and straightforward.
        
- **Noise Schedule:**
    - JAX:
        
        ```python
        # Add noise to the inputs during training
        noise_level = noise_schedule(epoch, total_epochs)
        x = random.normal(key, (1, 784))  # Example input
        x_noisy = add_noise(x, noise_level, key)
        logits = forward(params, x_noisy)
        ```
        
    - PyTorch:
        
        ```python
        # Add noise to the inputs during training
        noise_level = noise_schedule(epoch, total_epochs)
        x = torch.randn(1, 784)  # Example input
        model.train()
        x_noisy = add_noise(x, noise_level)
        optimizer.zero_grad()
        logits = model(x_noisy)
        ```
        
        Adding noise is the exact same, except for the fact that in PyTorch the model needs to be put into training mode.
        

## Comparison

- **Runtime:**
    - Forward pass:
    - Backward pass:
        - Compare Jitting, 1, 2 multiple passes
- **Comp time vs JIT backward time**
- **Runtime vs JIT backward time**
- **Compare NLL for PyTorch & JAX (Necessary)**

## Visualization

### Model Architecture

The architecture employs a U-Net-like structure, a common choice in diffusion models due to its effective handling of high-dimensional data. The U-Net consists of an encoder and a decoder, connected by a bottleneck layer that processes the data at multiple resolutions. The encoder progressively downsamples the input data, extracting hierarchical features that capture various levels of molecular detail. The decoder then upsamples these features, reconstructing the 3D molecular structure while ensuring that the output remains invariant to Euclidean transformations.

A key innovation in EDM is the integration of E(3) equivariant convolutional layers within the U-Net. These layers are designed to operate on the 3D coordinates and atom types simultaneously, ensuring that the learned features respect the symmetries of the underlying physical space. The equivariant layers perform convolutions that are invariant to rotations and translations, allowing the network to learn representations that are robust to such transformations.

<aside>
<img src="https://www.notion.so/icons/arrow-right-basic_blue.svg" alt="https://www.notion.so/icons/arrow-right-basic_blue.svg" width="40px" /> They use:
Diffusion:

Equivariant:
- As they consider interactions between all atoms, the assume a fully connected graph G with nodes vi, where each node represents a coordinate and the corresponding feautres.
- They use an Equivariant Graph NN composed of L Equiv Graph Conv Layers that apply the following non linear transformation: x,h = EGNN(x0,h0).

Puting them together: EQUIVARIANT DIFFUSION MODEL
- Consists on defining a noising process on node positions and features, and learning the generative denoising process using an equivariant NN. They also define the log-likelihood computation.
      - Noising:
             -Features: as they are invariant to E(n) transformations, the noise distribution for them will be the conventional normal distribution. However, depending if we are in categorical or ordinal we will have different representations of the features and different LIKELIHOOD.

</aside>

## Section 3: Our contribution

Our primary contribution is the re-implementation of the original EDM from ([Hogeboom (2022)$^{[6]}$](#6-hoogeboom-e-satorras-v-g-vignac-c-and-welling-m-2022-equivariant-diffusion-for-molecule-generation-in-3d)) in the JAX/FLAX framework. JAX, with its ability to automatically differentiate through native Python and Numpy functions, and FLAX, which provides a high-level interface for neural network building, collectively offer significant advantages in terms of performance and flexibility.

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
- **Ricardo**:
- **Robin**:

## References
#### [1]  Cao, N. D. and Kipf, T. (2022). Molgan: An implicit generative model for small molecular graphs.

#### [2]  Bilodeau, C., Jin, W., Jaakkola, T., Barzilay, R., and Jensen, K. F. (2022). Generative models for molecular discovery: Recent advances and challenges.


#### [3]  Kenneth Atz and Francesca Grisoni and Gisbert Schneider (2021). Geometric Deep Learning on Molecular Representations. 


#### [4]  Maziarz, K., Jackson-Flux, H., Cameron, P., Sirockin, F., Schneider, N., Stiefl, N., Segler, M., and Brockschmidt, M. (2022). Learning to extend molecular scaffolds with structural motifs.

#### [5] Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., and Vandergheynst, P. (2017). Geometric deep learning: Going beyond euclidean data. IEEE Signal Processing Magazine, 34(4):18–42.

#### [6] Hoogeboom, E., Satorras, V. G., Vignac, C., and Welling, M. (2022). Equivariant diffusion for molecule generation in 3d.

#### [7] Dhariwal, P. and Nichol, A. (2021). Diffusion models beat gans on image synthesis

#### [8] J.-P. Serre (1977). Linear Representations of Finite Groups
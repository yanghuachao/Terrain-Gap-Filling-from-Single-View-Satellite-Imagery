# Terrain-Gap-Filling-from-Single-View-Satellite-Imagery
Supplementary Materials for the Paper “Terrain Gap Filling from Single-View Satellite Imagery based on Prior Geographical Knowledge Constraints”

Terrain Gap Filling from Single-View Satellite Imagery Supplementary Document

Jianguo Pan*    Huachao Yang*    Zihao Lian*    Peichi Zhou*    Yuan Yang*    Chen Li$

Affiliations
*The College of Information, Mechanical and Electrical Engineering, Shanghai Normal University, China

$School of Computer Science and Engineering, Tianjin University of Technology, China

Related Work
To address the problem of missing data in digital elevation models (DEMs), conventional approaches include:

​​Manual reconstruction​​ - Time and labor intensive

​​Integration and fusion with other data sources​​ - Often results in quality inconsistencies

​​Interpolation techniques​​ - Frequently fails in complex terrain regions

These limitations have led to the emergence of deep learning techniques as the mainstream approach for DEM completion. Unlike traditional methods, deep learning approaches can:

Learn local terrain features

Capture global information and contextual characteristics

Extract latent information from consistent terrain patterns

Facilitate effective reconstruction of complete DEM

Convolutional Generative Adversarial Networks (CGANs)
CGANs have been widely applied to missing data reconstruction in DEM through:

​​Adversarial learning​​ - Generates height maps resembling real data

​​Effective restoration​​ of missing regions

Notable implementations include:

​​EdgeConnect​​ (Nazeri et al.) - Performs inpainting guided by image edges in missing regions

Achieves remarkable results in pixel-level tasks

However, pixel-level loss functions neglect semantic terrain constraints, leading to:

Violation of terrain continuity principles

Artifacts such as distorted shadow distributions

Disrupted runoff patterns

Inconsistency with real-world physical rules

Terrain Information Integration
Recent studies have combined CGAN frameworks with terrain-specific information:

​​Shadow-based supervision​​ (Dong) - Uses automatically extracted projected shadow maps with known solar directions as supervisory signals

​​Elevation-related features​​ (Qiu) - Joint training on global mountainous SRTM data with terrain features like relief

​​Constrained terrain knowledge​​ (Li) - TKCGAN model transfers terrain feature knowledge into training process

​​Multi-scale feature fusion​​ (Zhou) - Employs multi-attention refinement network with channel-spatial pruning mechanism

Remaining Challenges
Despite progress, several challenges persist:

​​Optimal selection​​ of terrain information types

​​Effective integration​​ of terrain information into deep learning models

​​Enhancement​​ of model generalization ability

​​Improvement​​ of refinement performance

Summary
Inpainting plays a crucial role in addressing missing DEM data, and CGAN-based deep learning methods have achieved remarkable results. However, key challenges remain in optimizing terrain information integration, feature learning, and enhancing model generalization and refinement performance.

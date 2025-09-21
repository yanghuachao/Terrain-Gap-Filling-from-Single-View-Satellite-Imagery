# Terrain-Gap-Filling-from-Single-View-Satellite-Imagery
Supplementary Materials for the Paper “Terrain Gap Filling from Single-View Satellite Imagery based on Prior Geographical Knowledge Constraints”

Terrain Gap Filling from Single-View Satellite Imagery Supplementary Document

Jianguo Pan*    Huachao Yang*    Zihao Lian*    Peichi Zhou*    Yuan Yang*    Chen Li$

Affiliations
*The College of Information, Mechanical and Electrical Engineering, Shanghai Normal University, China

$School of Computer Science and Engineering, Tianjin University of Technology, China

## Related Work

To address the problem of missing data in digital elevation models (DEMs), conventional approaches include manual reconstruction, integration and fusion with other data sources, as well as various interpolation techniques. However, these methods face significant limitations. Manual reconstruction is often costly in terms of time and labor, the integration of DEM from different sources frequently results in quality inconsistencies, and interpolation in complex terrain regions often fails to achieve satisfactory results [^1,^2,^3]. Therefore, deep learning techniques, owing to their powerful data learning and pattern recognition capabilities, have gradually become the mainstream approach for addressing missing data in DEM. Unlike traditional methods, deep learning approaches are able not only to learn local terrain features but also to capture global information and contextual characteristics. This enables the model to extract latent information from consistent terrain patterns, thereby facilitating the effective reconstruction of complete DEM [4,5].

Convolutional Generative Adversarial Networks (CGANs), as a representative deep learning architecture, have been widely applied to the task of missing data reconstruction in DEM. Through adversarial learning, this model can not only generate height maps that resemble real data but also effectively restore missing regions. Numerous inpainting models based on generative adversarial networks have demonstrated outstanding performance in practical applications. For example, Nazeri et al. proposed **EdgeConnect**, which performs inpainting guided by image edges in the missing regions and has achieved remarkable results in pixel-level tasks [6]. However, inpainting methods based on pixel-level loss functions neglect the semantic constraints inherent in terrain, thereby violating the principle of terrain continuity. As a result, the completed terrain often exhibits artifacts such as distorted shadow distributions and disrupted runoff patterns, which are inconsistent with real-world physical rules [7,8].

To address these issues, many studies have leveraged the CGAN framework in combination with specific terrain information in an attempt to improve the effectiveness of the refinement models. For example, Dong employed automatically extracted projected shadow maps together with known solar directions as shadow-based supervisory signals, in conjunction with the conventional supervision derived directly from DEM, thereby successfully enhancing the quality of the refinement results [7]. Qiu, on the other hand, jointly trained on global mountainous SRTM data together with elevation-related terrain features such as relief, effectively capturing more fine-grained topographic information and thereby enhancing the model’s capability to complete missing regions [9].

Although these methods have made certain progress in the integration of terrain information, several challenges remain, particularly regarding how to optimize the selection and integration of terrain information to further enhance the quality of the reconstructed DEM. For example, identifying which specific types of terrain information can effectively improve refinement performance and how to accurately incorporate such information into deep learning models remain unresolved issues. To address these challenges, Li proposed a constrained terrain knowledge-based CGAN model (TKCGAN), which effectively transfers knowledge of terrain features into the training process, thereby enhancing the model’s ability to recover critical terrain characteristics [10]. Zhou proposed a multi-scale feature fusion CGAN approach, which, after performing preliminary inpainting, employs a multi-attention refinement network to further recover details in the missing regions, while introducing a channel-spatial pruning mechanism to enhance network performance [11].

In summary, inpainting plays a crucial role in addressing missing data in DEM, and deep learning methods represented by CGANs have achieved remarkable results in terrain inpainting tasks. However, key challenges remain in further optimizing the integration and feature learning of terrain information, as well as in enhancing the model's generalization ability and refinement performance.

---

### References
[^1] F. Hallo, G. Falorni, and R. L. Bras, *Characterization and quantification of data voids in the shuttle radar topography mission data*, IEEE Geosci. Remote Sens. Lett., vol. 2, no. 2, pp. 177–181, 2005.  
[^2] S. J. Boulton and M. Stokes, *Which DEM is best for analyzing fluvial landscape development in mountainous terrains?*, Geomorphology, vol. 310, pp. 168–187, 2018.  
[^3] H. I. Reuter, A. Nelson, and A. Jarvis, *An evaluation of void-filling interpolation methods for SRTM data*, Int. J. Geogr. Inf. Sci., vol. 21, no. 9, pp. 983–1008, 2007.  
[4] G. Dong, F. Chen, and P. Ren, *Filling SRTM void data via conditional adversarial networks*, in Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS), 2018, pp. 7441–7443.  
[5] W. Li and C. Hsu, *Automated terrain feature identification from remote sensing imagery: A deep learning approach*, Int. J. Geogr. Inf. Sci., vol. 34, no. 4, 2020.  
[6] K. Nazeri, E. Ng, T. Joseph, et al., *EdgeConnect: Structure guided image inpainting using edge prediction*, in Proc. IEEE Int. Conf. Comput. Vis. Workshops (ICCVW), 2019, pp. 3265–3274.  
[7] G. Dong, W. Huang, W. A. P. Smith, and P. Ren, *A shadow-constrained conditional generative adversarial net for SRTM data restoration*, Remote Sens. Environ., vol. 237, pp. 111602, 2020.  
[8] P. Zhou, D. Lu, C. Li, et al., *Unsupervised textured terrain generation via differentiable rendering*, in Proc. ACM Int. Conf. Multimedia (ACM MM), 2022, pp. 2654–2662.  
[9] Z. Qiu, L. Yue, and X. Liu, *Void-filling of digital elevation models with a terrain texture learning model based on generative adversarial networks*, Remote Sens., vol. 11, no. 23, pp. 2829, 2019.  
[10] S. Li, G. Hu, X. Cheng, et al., *Integrating topographic knowledge into deep learning for the void-filling of digital elevation models*, Remote Sens. Environ., vol. 269, pp. 112818, 2022.  
[11] G. Zhou, B. Song, P. Liang, et al., *Voids filling of DEM with multi-attention generative adversarial network model*, Remote Sens., vol. 14, no. 5, pp. 1206, 2022.  


**References**  

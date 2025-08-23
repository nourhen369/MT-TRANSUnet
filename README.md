## Multi-Task TRANSU-Net: Segmentation & Classification

Ce projet implémente un modèle de type **TransUNet** pour la segmentation et la classification d'images médicales d’embryons humains (blastocystes).  
Il se base sur deux références principales :  

- **TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation**  
  [Lien vers l’article](https://arxiv.org/abs/2102.04306)

- **Automatic Identification of Human Blastocyst Components via Texture** (IEEE)  
  [Lien vers l’article](https://ieeexplore.ieee.org/document/8059868)

---

## Dataset

Le dataset introduit dans l’article IEEE se compose d’images d’embryons annotées par des experts, utilisées pour **la segmentation et la classification**.  
Il présente un double intérêt :  

- **Segmentation** : localisation et délimitation des structures principales telles que *Zona Pellucida (ZP)*, *Inner Cell Mass (ICM)* et *Trophectoderme (TE)*.  
- **Classification** : identification et catégorisation des différents composants pour une analyse plus fine.

Voici un exemple de **masques Ground Truth (GT)** utilisés pour la segmentation des composants embryonnaires :

<img src="assets/segmentation-mask.png" alt="Masques GT" width="400"/>

---

## Challenges rencontrés

Durant le développement, plusieurs défis ont été identifiés :  

1. **Gestion des tâches multiples (Segmentation + Classification)**  
   - Combiner les pertes pour les deux tâches tout en maintenant un entraînement stable a nécessité des expérimentations avec les **loss weights** et des callbacks personnalisés.

2. **Overfitting sur la classification**  
   - Le modèle montre une forte précision sur l’entraînement mais une faible performance sur la validation, **à cause de la taille limitée du dataset**.  
   - Solutions explorées : ajustement du learning rate, callbacks, régularisation et **cross-validation**.

3. **Prétraitement et qualité des images**  
   - Variations de tailles et de luminosité dans les images.  
   - Mise en place d’une pipeline de **prétraitement et normalisation** pour garantir une meilleure cohérence des données.

4. **Limitation du dataset**  
   - Peu d’exemples annotés pour certaines classes, compliquant l’entraînement d’un modèle robuste.  
   - Solutions : undersampling/oversampling et data augmentation.

5. **Temps d’entraînement élevé**  
   - La combinaison TransUNet + tâches multiples est gourmande en GPU.  
   - Solutions : batch size adapté, callbacks pour early stopping et réduction automatique du learning rate.


---

## Projet et fichiers

- **Frontend** : `ui` en **React**.  
- **Entraînement** : `model/train.ipynb` contient le prétraitement, l’entraînement, les métriques et callbacks.  
- **API** : `app.py` fournit un endpoint **FastAPI** pour les prédictions.  
- **Dépendances** : listées dans `requirements.txt`.

---

## Demo

Voici un aperçu de notre modèle en action :  
<img src="assets/demo.gif" alt="Demo" width="500"/>
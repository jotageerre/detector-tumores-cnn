**Langue :** [Español](../01_historia_del_proyecto.md) · [English](../en/01_project_history.md) · **Français**

# 1. Historique du projet : ce qui a mal tourné et comment cela a été corrigé

Ce TFG n'a pas suivi un plan linéaire. Son apport le plus utile pour d'autres personnes, c'est justement le chemin parcouru : trois décisions méthodologiques qui ont changé le projet et que toute personne travaillant avec des images médicales devrait connaître.

Source : chapitres 1, 4 et 6 du [mémoire](../TFG_Joaquin_Gonzalez_Rodriguez_memoria.pdf) (en espagnol).

<p align="center"><img src="../img/evolucion_datos.jpg" width="60%"></p>

## Étape 1 — Prototype avec des données Kaggle (sept. 2025 – juil. 2026)

Les premières versions du classifieur ont été entraînées sur des collections d'IRM agrégées et redistribuées sur Kaggle, organisées en un dossier par classe. Elles ont servi à comprendre le problème, à monter l'infrastructure d'entraînement et à tester une première interface (le prototype *ScanIA*, voir [`legacy/scania`](../../legacy/scania)).

**Le problème :** aucune de ces sources ne conservait l'identifiant du patient. Le pipeline utilisait un `group_id` tiré du nom de fichier comme s'il s'agissait du patient, mais il n'y avait aucun moyen de le vérifier. Avec cette organisation, il était **impossible de garantir** que des coupes d'un même patient n'apparaissaient pas à la fois en entraînement et en test.

> Aucune fuite d'information n'a été démontrée ; ce qui a été détecté est un **risque structurel** qui empêchait de l'écarter. Cette différence est importante, et c'est elle qui a conduit à reconstruire les données plutôt qu'à les rafistoler.

## Étape 2 — Reconstruction avec de vrais identifiants de patient (août 2026)

Le jeu de données a été reconstruit à partir des **sources primaires**, qui publient bien l'identité de chaque patient :

| Classe | Source | Identifiant de patient | Licence |
|---|---|---|---|
| tumeur | Cheng et al. — figshare | champ `cjdata.PID` de chaque fichier `.mat` | CC BY 4.0 |
| sans tumeur | IXI — Imperial College London | ID du sujet dans le nom du fichier NIfTI (`IXI002-Guys-0828-T1`) | CC BY-SA 3.0 |

Résultat : **467 patients réels** (233 Cheng + 234 IXI) et 5 872 images, avec un manifeste (`manifest_final.csv`) qui enregistre pour chaque image son patient, sa source, son SHA‑256 et son sous-ensemble. Des contrôles automatiques ont été ajoutés : aucun patient dans deux sous-ensembles, SHA‑256 uniques et détection des quasi-doublons (pHash + SSIM). L'exécution complète a été automatisée sur le cluster HPC de l'Université de Séville (job 70436, 28 min sur une NVIDIA A30).

## Étape 3 — Le biais de provenance (et pourquoi la tâche binaire a été abandonnée)

Sur le jeu de données reconstruit, la tâche binaire « tumeur / pas de tumeur » a atteint une balanced accuracy de **98,57 %**. Cela ressemblait à un succès. Ce n'en était pas un :

- Toutes les tumeurs venaient de Cheng (T1 **avec** contraste, deux hôpitaux chinois).
- Tous les sujets sains venaient d'IXI (T1 **sans** contraste, hôpitaux de Londres).
- **L'étiquette coïncidait exactement avec le jeu de données d'origine.**

Pour le vérifier, un classifieur de contrôle a été entraîné avec **seulement 10 descripteurs statistiques et de texture, sans aucun réseau de neurones** : il a obtenu **98,16 %** d'accuracy et 99,55 % d'AUC. Si quelque chose d'aussi simple sépare les classes, le réseau n'a pas besoin de « voir » la tumeur : reconnaître l'hôpital suffit. Un autre contrôle l'a confirmé : un classifieur a su identifier le **centre d'acquisition au sein d'IXI** (uniquement des sujets sains) avec 89,32 % d'accuracy.

> **Leçon :** si vos classes positive et négative proviennent de jeux de données différents, votre modèle apprend peut-être le scanner, pas la maladie. Construisez toujours un classifieur de contrôle « naïf » avant de vous réjouir.

## Étape 4 — Réorientation vers la classification multiclasse au sein d'une seule cohorte

Le cœur du TFG est devenu la distinction entre **gliome, méningiome et tumeur hypophysaire** au sein de Cheng. Les trois classes venant de la même cohorte, le lien déterministe entre classe et jeu de données disparaît. Tous les raccourcis possibles ne disparaissent *pas* pour autant : le même contrôle de bas niveau obtient 69,22 % sur cette tâche (au-dessus du hasard, 33 %), donc il existe aussi un signal de bas niveau au sein de Cheng. C'est documenté comme une limite, pas caché.

Une partition par patient **163 / 35 / 35** (entraînement / validation / test) a été figée avec le hash `fda7e2daeb9ec2de…`. Les 35 patients de test sont restés **verrouillés** jusqu'à la fin.

## Étape 5 — Sélection de l'architecture, évaluation unique et validation 5CV (sept. 2026)

1. **Benchmark** de 5 architectures pré-entraînées (VGG16, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3) avec une validation croisée groupée par patient à 3 plis sur les 198 patients d'entraînement+validation.
2. **Ajustement fin** des deux meilleures (ResNet50, InceptionV3) et **ensemble** 50/50, chaque étape avec son protocole rédigé *avant* de l'exécuter ([`protocolos/`](../protocolos), en espagnol).
3. **Protocole d'évaluation finale** rédigé et hashé → entraînement final → **test évalué une seule fois** : 93,89 % de balanced accuracy.
4. **Validation croisée à 5 plis** sur les 233 patients pour mesurer la variabilité : 89,44 % ± 4,64 pts. Le résultat du test se situe près du haut de la fourchette, donc le chiffre de la 5CV est l'estimation la plus prudente.

## Étape 6 — Service dans le cloud et application de bureau (sept. 2026)

Le modèle final est servi par un service **Cloud Run** (`run-model`), et une application de bureau Tkinter permet d'envoyer une image ou un volume NIfTI et de voir la prédiction, la carte Grad-CAM et une visionneuse 3D. Une vraie panne de production (erreurs 503 dues au manque de mémoire lors du chargement de plusieurs copies du modèle) a été diagnostiquée et corrigée ; elle est documentée dans [06_deploiement_google_cloud.md](06_deploiement_google_cloud.md#problème-connu--erreur-503).

## Effort

306 h enregistrées sur 101 sessions (Clockify). Le lot de travail qui s'est le plus écarté de l'estimation initiale est **données et traçabilité** (63 h contre 25 h prévues, +152 %) : la reconstruction du jeu de données n'était pas prévue, et c'est la partie la plus précieuse du travail.

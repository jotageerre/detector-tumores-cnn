**Langue :** [Español](../07_limitaciones_y_etica.md) · [English](../en/07_limitations_and_ethics.md) · **Français**

# 7. Limites et considérations éthiques

Source : chapitres 11 et 13 du mémoire (en espagnol).

## Ce que ce travail ne démontre PAS

- **Il n'y a pas de validation externe.** Tout a été évalué au sein de la cohorte Cheng (233 patients, deux hôpitaux chinois, T1 avec contraste). On ne sait pas comment le modèle se comporte avec d'autres scanners, protocoles, hôpitaux ou populations.
- **Il n'y a pas de validation clinique.** Ce n'est ni un dispositif médical ni un outil de diagnostic.
- **Le test est petit.** 35 patients : un seul patient mal classé fait varier la balanced accuracy de plusieurs points. D'où l'intervalle de confiance très large [84,26 %, 100 %] et l'importance de la validation à 5 plis.
- **La 5CV n'est pas imbriquée.** L'architecture a été choisie auparavant avec une autre partition ; la 5CV mesure la variabilité du modèle choisi, pas celle de tout le processus de sélection.
- **Grad-CAM est qualitatif.** Il n'a pas été comparé aux masques tumoraux, il ne prouve donc pas que le modèle regarde la tumeur.
- **Raccourcis possibles au sein de Cheng.** Un classifieur de bas niveau obtient 69,22 % sur la tâche multiclasse (le hasard donnerait 33 %). Il n'a pas été démontré que le modèle s'appuie dessus, mais cela n'a pas été écarté non plus.
- **Pas de déterminisme strict du GPU.** Les réexécutions ne donneront pas exactement les mêmes chiffres.
- **Une partie du code de sélection de l'architecture n'a pas été conservée** (la version exacte du notebook avec l'ajustement fin, l'ensemble et l'évaluation finale ; annexe A.3).

## Données

| Aspect | Cheng | IXI |
|---|---|---|
| Anonymisation | Déclarée explicitement par les auteurs | Non vérifiée avec le même niveau de détail |
| Approbation éthique | Déclarée (comités du Nanfang Hospital et du General Hospital de la Tianjin Medical University) | Non vérifiée avec le même niveau de détail |
| Consentement éclairé | Aucune mention explicite trouvée | Non vérifié |
| Licence | CC BY 4.0 (vérifiée avec réserve) | CC BY-SA 3.0, citation obligatoire |

Ce dépôt **ne redistribue aucune image** : seulement des identifiants de patient issus du jeu de données public lui-même (p. ex. `CHENG-100360`), nécessaires pour reproduire la partition.

## L'application de bureau

- Les mots de passe des utilisateurs locaux sont stockés **en clair** dans `credentials.json` (vraie limite du démonstrateur, pas une fonctionnalité de sécurité).
- Les résultats (Grad-CAM, visionneuse 3D) sont publiés dans des buckets lisibles publiquement.
- Il n'existe aucun journal persistant indiquant quelle image, quel modèle et quel résultat a produit chaque inférence.

Pour toutes ces raisons, **n'utilisez pas l'application avec des données de vrais patients**.

## Travaux futurs proposés

1. Validation externe sur une cohorte indépendante (autre centre, autre scanner).
2. Validation croisée imbriquée (sélection du modèle à l'intérieur de l'estimation).
3. Évaluation quantitative de Grad-CAM par rapport aux masques tumoraux de Cheng.
4. Rechercher des signaux résiduels de centre/scanner au sein de la cohorte et essayer d'autres règles d'agrégation par patient.
5. Déterminisme GPU, environnement conteneurisé reproductible et empaquetage complet.
6. Dans l'application : le même prétraitement qu'à l'entraînement, la traçabilité de chaque inférence et un stockage privé.

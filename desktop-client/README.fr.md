**Langue :** [Español](README.md) · [English](README.en.md) · **Français**

# desktop-client — application de bureau

Application Tkinter qui :

1. connecte un utilisateur local ;
2. permet de choisir le modèle (**Cheng multiclasse** ou binaire historique) ;
3. envoie une image (JPG/PNG) ou un volume NIfTI vers Google Cloud Storage ;
4. appelle `POST /predict` du service Cloud Run ;
5. affiche la classe, les probabilités et la carte Grad-CAM, ou ouvre la visionneuse 3D dans le navigateur.

<p align="center"><img src="../docs/img/app_resultado.jpg" width="60%"></p>

## Prérequis

- Python 3.10+ (Tkinter est fourni avec l'installateur officiel Windows/macOS ; sous Linux : `sudo apt install python3-tk`).
- Un service Cloud Run déployé et un compte de service autorisé à envoyer des fichiers dans les buckets ([`docs/fr/06_deploiement_google_cloud.md`](../docs/fr/06_deploiement_google_cloud.md)).

## Installation

```bash
python -m venv myenv
myenv\Scripts\activate            # Windows  (Linux/macOS : source myenv/bin/activate)
pip install -r requirements.txt
```

## Configuration (rien de tout cela n'est envoyé sur GitHub)

| Quoi | Comment |
|---|---|
| Clé du compte de service | Copiez le JSON dans ce dossier sous le nom **`gcp-service-account.json`**, ou définissez `GOOGLE_APPLICATION_CREDENTIALS` |
| URL de l'API | Variable `SCANIA_API_URL`, ou *Settings* dans l'application, ou `config.json` (copie de `config.example.json`) |
| Buckets | Variables `SCANIA_BUCKET_2D` et `SCANIA_BUCKET_3D` |

`credentials.json` (utilisateurs de l'application) et `config.json` sont créés automatiquement à l'usage. Les mots de passe sont stockés en clair : c'est une démo, n'utilisez pas de vrais mots de passe.

## Lancer

```bash
python main.py
```

> Outil expérimental. Le résultat ne constitue pas un diagnostic médical.

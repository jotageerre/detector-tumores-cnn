**Langue :** [Español](README.md) · [English](README.en.md) · **Français**

# legacy/scania — prototype historique d'interface

Artefacts du premier prototype d'interface (*ScanIA*), datés du 11/07/2026, avant la reconstruction du jeu de données. Ils sont conservés **uniquement comme référence historique** (annexe F des annexes techniques).

| Fichier | État |
|---|---|
| `brain_viewer.html` | Modèle d'interface avec des emplacements pour le CSS, le JS et les données du volume |
| `brain_viewer_1.css` | Feuille de style complète |
| `Dockerfile` | Conteneur prévu pour Cloud Run (`gunicorn … main:app`) |

Aucun backend n'est associé à ce Dockerfile et le fichier de logique JavaScript original était vide. Le prototype n'a **jamais** été connecté au modèle multiclasse final. L'application qui fonctionne réellement est [`desktop-client/`](../../desktop-client) + [`cloud-run-backend/`](../../cloud-run-backend), dont le `main.py` intègre déjà sa propre visionneuse 3D.

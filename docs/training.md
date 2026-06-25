# Entrainement de S-JEPA
Le modele S-JEPA est utilise pour modeliser le son. Precisement la parole humaine et autre. Il s'agira d'implementer
le programme d'entrainement et d'evaluation complete pour entrainer le modele **S-JEPA** pour une utilisation
en production.

On doit pouvoir entrainer le modele avec une dataset composer de fichier sons venant de n'importe quel format
(.wav, .mp3, .mp2, etc...). Ces fichiers seront soite dans un dossier zippé (.zip, .tar), soit dans un dossier non
compressé. Les fichiers peuvent etre a la racine du dossier et/ou dans des sous dossiers. Il faut juste que la fonction
de chargement des fichiers de donnees audio soit assez futté pour reperer et charger tous les fichiers audio
de façon recursifs depuis le dossier (zippé ou non). Donc, si l'utilisateur donne un fichier zippé, sans décompresser
on doit pouvoir lire tous les fichiers audios qu'il contient de façon recursifs. Pareil si le dossier n'est pas
compressé.

## Dataset
Les deux datasets de meme structure (`train_path` et `test_path`) sont fournies separement dans les fichiers
de configurations.
Ensuite dans les fichiers de configs, l'utilisateur de definir le nombre maximal
(`max_train_samples`, `max_test_samples`) d'exemples de donnees qu'il souhaite utiliser dans chacune des deux datasets
fournies. La dataset de validation, quant à elle, est constituee uniquement à partie de la dataset de `test`.
Ainsi, l'utilisateur peut specifier dans les configs la fraction `val_prob` des donnees de teste à prendre
pour constituer la dataset de validation. Par defaut, `val_prob` vaut $0.5$ (soit $50%$ des donnees de teste).

Le paramettrage et l'activations des algorithmes de transformation et d'augmentation de donnees doivent etre disponibles
dans les configurations pour permettre un controle total.

### Filtrage Des Donnees
Avant de commencer a faire quoi que ce soit sur les donnees de la dataset, il faut a tout pris passer par l'etape
de nettoyage. Cela consiste a scanner et a filtrer par toutes les techniques les donnees de chacune des datasets
(train, comme teste) -- par exemple : verification de format, fichier corrompu, fichier inneccibles, etc,
et autre technique de verification avancees et adaptees a notre context actuel. Une fois filtrer, il faut simplement
retenir les echantillons correctes et les sauvegarder dans des fichiers cache, un par dataset
(`train.cache.json`, `test.cache.json`) dans le meme dossier que les datasets respectives. Voici un exemple
de structure :

```
path/to/dataset/directory/
  train.zip
  train.cache.json
```

```
path/to/dataset/directory/
  test.zip
  test.cache.json
```

Comme ca, au prochain chargement du programme d'entrainement ou du programme d'evaluation, il faut tout simplement
charger les echantillons retenues depuis les `.cache.json` pour chacune des datasets afin d'eviter de rescanner
et refiltrer a chaque fois. Ce qui peut devenir chronophage.

### Dataset HDF5
La possibilite de charger, transformer et augmenter toute la dataset dans des fichiers HDF5 (`train.h5` et `test.h5`).
Ce qui sera dans les fichier HDF5 sera les donnees d'exemples deja transformees et si l'utilisateur
demande d'augmenter directement alors il y aura aussi les exemples de donnees augmentees. C'est pour permettre de faire
moins de calcul pendant l'entrainement et l'evaluation. Ensuite, dans les fichiers de configuration, on doit pouvoir
swicher entre la dataset bruite et la dataset deja buildee sous forme HDF5.

Exemples de configuration de la dataset dans les fichiers de configuration :

```yaml
dataset:
  use_hdf5: false              # read from zip on the fly
  validate: true               # pre-flight: drop corrupt/invalid zip entries up front
  train_path: path/to/dataset/directory/train.zip
  test_path: path/to/dataset/directory/test.zip
  train_h5: data/train.h5
  test_h5: data/test.h5
  max_train_samples: null      # null = use all; or an integer cap
  max_test_samples: null
  val_prob: 0.5                # fraction of test used for per-epoch validation
  augment:
    enabled: true             # on-the-fly detector-style noise on the train split
    body_jitter_std: 0.01
    hand_jitter_std: 0.02
    joint_drop_prob: 0.03
    hand_drop_prob: 0.1
    lr_swap_prob: 0.01
```


## Accumulation De Gradient
Accumulation de gradient pour simuler de large lot de données.
Et n'oublit de rappeler l'optimizer à la fin de l'epoch
car il peut y avoir d'accumulation restant qui n'ont pas été optimisé.
Tu dois prendre ce cas en charge en verifiant si on a eu une accumulation qui n'a pas ete optimisé.
Pour cela tu peux essaiyer d'utiliser une variables booleen pour le savoir et reappeler l'optimization
si besoin à la fin de la boucle.

## Checkpoint
Un système de checkpoint permettant de sauvegarder les poids du model, de l'optimiseur, et de scheduler.
La fonction de checkpoint doit sauvegarder tout l'etat de l'entrainement y compris les poids du models,
les optimiseurs, scheduler, etc, à la fin de chaque epoch. Tu feras le checkpoint dans le dossier checkpoint
que l'utilisateur indiquera dans le fichier de configuration `train.yaml`. A chaque checkpoint, tu verifiras
si le nombre de fichier de checkpoint est superieur a un nombre `max_checkpoint`. Si le nombre de fichier
de checkpoints est supperieur a un certain seuil (`max_checkpoint`), alors tu supprime le ou les plus vieux fichiers
de checkpoint afin d'optimizer l'espace disk. `max_checkpoint` est aussi à specifier dans le fichier de configuration.
Enfin, integrer la fonction de reprise de l'entrainement au dernier point de sauvegarde (checkpoint).
Cette fonction de reprise est prioritaire sur la fonction de demarrage d'entrainement a partie d'un modele existant.

## Historique d'Entrainement
A chaque fin de tour de boucle d'epoch, il y a la génération de la courbe de l'historique d'entrainement/
Validation pour chaque fonction d'erreur afin permettre a l'utilisateur d'avoir un visuelle graphique.
Ces graphiques doivent nous permettre de savoir si le modele est en train d'overfiter ou non.

## Demarrage A Partir De Poids Existant
On doit pouvoir démarrer l'entraînement à partir des poinds d'un modèle existant. Lorsque l'utilisateur
fournit dans le fichier de configuration `train.yaml` l'emplacement vers le fichier de poids (`.pt`)
d'un modèle existant sauvegardé, le programme doit pouvoir le charger et l'entrainer.
Cette fonctionnalité doit pouvoir immédiatement s'exécuté après vérification d'un checkpoint. En effet,
si aucun checkpoint n'existe, alors on procède au chargement d'un modèle existant éventuellement specifié
dans le fichier de configuration. L'ordre des priorités est donc: checkpoint -> load model.
Si un checkpoint existe, alors on ignore le chargement du modèle existant et on se concentre sur les données chargées
du dernier checkpoint.

## Sauvegarde Du Meilleur Modele
Il s'agit de tracker le modele de meilleur precision apres chaque etape de validations apres chaque boucle
d'entrainement. Seule une des metriques de validation est utilisee.
On doit sauvegarder des poids du modèle de meilleur performance : Il s'agira de sauvegarder à la fin de chaque
étape de validation le modèle qui a atteint plus de performance à la validation. L'emplacement du meilleur modèle
doit être configurable depuis le fichier de configs. Il faut aussi laisser le choix à l'utilisateur de pouvoir
spécifier entre tous les metriques observees, le métrique qui sera utilisé comme critère pour retenir le modele
de meilleur performance.

## Rendu
Tout les scriptes qui seront developpes doivent produire de jolie rendu terminal (texte) dans le style des geeks
prensentant les logs et les barres de progressions et autre avec tous les metrics affichees bien jolie sans icones,
ni emoji, ni caracteres speciaux impossibles à taper sur un clavier Anglais.
Par contre les ANSI char doivent etre utilises pour la coloration sur le terminal.

> NOTE: Le module `loguru` doit est utilise pour la journalisation et dois etre configure pour qu'il stocke les logs
dans des fichiers de journalisation.

Au demarrage de l'entrainement ou de n'importe quel programme qui doit agir sur le modele, il faut toujours faire
un summary complete de l'architecture du modele en utilisant par exemple `torchinfo` afin de nous permettre d'avoir
connaissance de l'architecture global, de l'etat des paramettres et de la tailles du modele en memoire.

### Progression D'entrainement
Dans le programme d'entrainement, il y aura une grande barre de progression pour materialiser la progression au file
des epoches de l'entrainement du modele. On doit savoir par exemple combien de temps total il nous reste pour terminer
toutes les epoches de la boucle d'entrainement. Ensuite, il y aura une sous barre de la grande barre de progression
qui donne la progression de chaque etape d'une epoche de la boucle d'entrainement (training, validation, etc).
C'est pour connaitre l'avancement de chaque etape d'une epoche de la boucle d'entrainement.

Afin de pouvoir logger avec `loguru` sans endommager le rendu de la grande barre de progression de `tqdm`,
on doit toujour instancier `tqdm` avec le paramettre ` leave=True` comme suit :

```python
pbar = tqdm(dataloader, leave=True, ...)
```

Par contre pour les petites barres de progression, on doit les instancier avec ` leave=False`.

> NOTE: Les deux instances de barre doivent etre separee en `epoch_bar` et `step_bar`

On doit afficher de facon périodique des métriques pendant le deroulement une epoch d'entrainement.
Il s'agit des steps de l'etape d'entrainement du modele et non de l'etape de validation du modele.
L'affichage des steps d'entrainement (`epoch 1/10 | step 400/553824 |...`) doit etre fait
avec le logger (`loguru`). Voici des exemples de rendu qu'on doit avoir sur le terminal :

```
2026-06-15 21:51:04 INFO    |     validate: true
2026-06-15 21:51:04 INFO    |     train_path: /home/mokira3d48/datasets/mocap/train.zip
2026-06-15 21:51:04 INFO    |     test_path: /home/mokira3d48/datasets/mocap/test.zip
2026-06-15 21:51:04 INFO    |     train_h5: data/train.h5
2026-06-15 21:51:04 INFO    |     test_h5: data/test.h5
2026-06-15 21:51:04 INFO    |     max_train_samples: null
2026-06-15 21:51:04 INFO    |     max_test_samples: null
2026-06-15 21:51:04 INFO    |     val_prob: 0.5
2026-06-15 21:51:04 INFO    |     augment:
2026-06-15 21:51:04 INFO    |       enabled: true
2026-06-15 21:51:04 INFO    |       body_jitter_std: 0.01
2026-06-15 21:51:04 INFO    |       hand_jitter_std: 0.02
2026-06-15 21:51:04 INFO    |       joint_drop_prob: 0.03
2026-06-15 21:51:04 INFO    |       hand_drop_prob: 0.1
2026-06-15 21:51:04 INFO    |       lr_swap_prob: 0.01
validating dataset:  14%|██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 8570/59832 [00:11<00:59, 865.66it/s]
```

```
2026-06-15 21:53:51 INFO    | Starting training pipeline
2026-06-15 21:53:51 INFO    | ===== Run summary =====
2026-06-15 21:53:51 INFO    |   device                 = cpu
2026-06-15 21:53:51 INFO    |   epochs                 = 10 (start_epoch=0)
2026-06-15 21:53:51 INFO    |   batch_size             = 16 x grad_accum=4 -> effective=64
2026-06-15 21:53:51 INFO    |   batches/epoch          = 3740 | optimizer_steps/epoch=935
2026-06-15 21:53:51 INFO    |   total optimizer steps  = 9350
2026-06-15 21:53:51 INFO    |   grad_clip_norm         = 1.0
2026-06-15 21:53:51 INFO    |   optimizer              = adamw lr=3.000e-04 wd=0.01
2026-06-15 21:53:51 INFO    |   scheduler              = cosine
2026-06-15 21:53:51 INFO    |   best criterion         = mpjae_deg (mode=min)
2026-06-15 21:53:51 INFO    |   learnable loss weights = False
2026-06-15 21:53:51 INFO    |   loss weights           = LossWeights(reprojection=1.0, joints3d=1.0, rotation=1.0, velocity=0.1, acceleration=0.1, foot_sliding=0.5, contact_bce=0.1, bone_length=0.001, transl_reg=0.01, betas=0.1, motion_prior=0.1)
2026-06-15 21:53:51 INFO    |   model parameters       = total 2,847,618 | trainable 2,847,618
2026-06-15 21:53:51 INFO    |   data                   = train 59832 | val 4428 | test 8856
2026-06-15 21:53:51 INFO    |   checkpoint dir         = runs/smplx_mocap/train4/checkpoints (max=5)
2026-06-15 21:53:51 INFO    |   outputs                = runs/smplx_mocap/train4
2026-06-15 21:53:51 INFO    | ===== Starting epoch 1/10 =====
```

```
2026-06-15 22:01:50 INFO    |   model parameters       = total 2,847,618 | trainable 2,847,618
2026-06-15 22:01:50 INFO    |   data                   = train 59832 | val 4428 | test 8856
2026-06-15 22:01:50 INFO    |   checkpoint dir         = runs/smplx_mocap/train5/checkpoints (max=5)
2026-06-15 22:01:50 INFO    |   outputs                = runs/smplx_mocap/train5
2026-06-15 22:01:50 INFO    | ===== Starting epoch 1/10 =====
2026-06-15 22:01:57 DEBUG   | step 16/3740 grad_norm=6.099 loss=4.861 reprojection=2.955 joints3d=0.439 rotation=1.080 bone_length=0.000 betas=0.387 transl_reg=0.000
2026-06-15 22:02:04 DEBUG   | step 32/3740 grad_norm=4.941 loss=4.729 reprojection=2.835 joints3d=0.433 rotation=1.085 bone_length=0.000 betas=0.376 transl_reg=0.000
2026-06-15 22:02:11 DEBUG   | step 48/3740 grad_norm=5.584 loss=4.626 reprojection=2.777 joints3d=0.420 rotation=1.066 bone_length=0.000 betas=0.362 transl_reg=0.000
2026-06-15 22:02:17 DEBUG   | step 64/3740 grad_norm=6.436 loss=4.539 reprojection=2.733 joints3d=0.407 rotation=1.051 bone_length=0.000 betas=0.349 transl_reg=0.000
2026-06-15 22:02:26 DEBUG   | step 80/3740 grad_norm=6.113 loss=4.466 reprojection=2.697 joints3d=0.396 rotation=1.038 bone_length=0.000 betas=0.336 transl_reg=0.000
2026-06-15 22:02:34 DEBUG   | step 96/3740 grad_norm=7.752 loss=4.368 reprojection=2.637 joints3d=0.387 rotation=1.021 bone_length=0.000 betas=0.323 transl_reg=0.000
2026-06-15 22:02:41 DEBUG   | step 112/3740 grad_norm=4.512 loss=4.299 reprojection=2.601 joints3d=0.378 rotation=1.008 bone_length=0.000 betas=0.311 transl_reg=0.000
2026-06-15 22:02:48 DEBUG   | step 128/3740 grad_norm=8.971 loss=4.225 reprojection=2.553 joints3d=0.372 rotation=0.999 bone_length=0.000 betas=0.300 transl_reg=0.000
2026-06-15 22:02:57 DEBUG   | step 144/3740 grad_norm=12.049 loss=4.165 reprojection=2.521 joints3d=0.367 rotation=0.987 bone_length=0.000 betas=0.289 transl_reg=0.000
2026-06-15 22:03:07 DEBUG   | step 160/3740 grad_norm=6.124 loss=4.091 reprojection=2.473 joints3d=0.362 rotation=0.978 bone_length=0.000 betas=0.278 transl_reg=0.000
TRAINING:   0%|░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 0/10 [01:16<?, ?it/s, best_mpjae_deg=n/a lr=3.00e-04]
    train e1:   4%|█░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 165/3740 [01:18<31:24,  1.90it/s, loss=4.074 reprojection=2.463 joints3d=0.360 rotation=0.976 bone_length=0.000 betas=0.274 transl_reg=0.000]
```

Le caractere de progression qui sera utilise sur la barre de progression de `tqdm` est : `█`. Le caractere d'arriere
plan utilise sur la barre de progression sera `░`.

Sur la grande barre de progression, on doit afficher en prefix les informations suivantes :

- le temps ecoule depuis le demarrage de la boucle;
- le temps restant pour terminer la boucle d'entrainement;
- la durree d'une epoche en moyenne;
- le meilleur score (le valeur la plus elevee du metrique designe comme critere de meilleur performance);
- la valeur actuelle du learning rate.

Sur la petite barre de progression, on doit juste afficher les valeurs des metriques pertinantes liees a l'etape
(train, validation) actuelle.

<!-- ### Progression De La Validation -->


### Sortie
Les entrainements de modele produisent de fichiers comme les checkpoints, le `best.pt`, les plotes
de training vs validation, etc ...
Ces fichiers sont ranges dans un dossier appele `runs`. Ce dossier comporte tous les resultats de chacun
de tous les runs. Ce dossier suit la structure suivante :

```
runs/
  run_name/  # Le nom donne au run dans les fichiers de configurations.
    train/   # Le premier `run` d'entrainement.
    train2/
    train3/
      history.csv
      config_used.yaml  # Fichier de configuration utilise pour initialiser l'entrainement.
      weights/
        best.pt         # Les poids du modele aillant les meilleurs performances.
        last.pt         # Les poids de la version du modele sauvegarde a la derniere epoch.
      checkpoints/
        epoch_000.pth
        epoch_001.pth
            .
            .
            .
        epoch_n.pth
      plotes/
        training_history.jpg  # La courbe d'entrainement VS validation pour detecter d'eventuel overfiting.
            .
            .
            .
        *.jpg
      logs/  # Le dossier comportant toutes les journalisations.
        train_2026-06-06_13-42-11.log
        train_2026-06-07_16-27-51.log
        train_2026-06-07_16-30-03.log
        train_2026-06-07_18-08-02.log
        ...
    eval/
    eval2/
    eval3/
      .
      .
      .
```

A chaque fois qu'on demarre un run d'entrainement avec le meme nom de run dans les configs, alors un nouveau dossier
$traini$ (par exemple: train2, train10) est cree. S'il s'agit du demarrage d'un run d'evaluation portant
le meme nom du run, alors un nouveau dossier $evali$ (par exemple: eval19, eval38) est cree.
Si c'est le premier entrainement ou evaluation alors le dossier `train` ou `eval` ne porte aucun numero.
Au premier run il n'est pas question d'avoir un train0 ou eval0 ou meme un train1 ou eval1.

> NOTE IMPORTANTE: Lorsque dans les configs, resume: true et qu'un checkpoint réutilisable existe,
  réutiliser le dossier i de run existant au lieu d'en créer un nouveau. Si n'y a pas un seul dossier, on charge
  le checkpoint du dossier dont le numero est le plus **élevé**. Ex: entre `train`, `train1` et `train2`, il faut
  charger le checkpoint de `train2`.
  Les checkpoints, l'historique et les poids continuent au même endroit, et la reprise devient déterministe.

Concernant la structure du dossier d'evaluation, on aura :

```
eval/
  plotes/      # Dossier comportant les graphiques.
  renders/     # Des exemples de rendus de predictions effectues par le modele en question.
  results.csv  # Fichier comportant les valeurs de metriques d'evaluation.
  config_used.yaml  # Les configs d'utilises pour l'evaluation du modele (`eval.yaml`).
  logs/             # Le dossier comportant toutes les journalisations.
    eval_2026-06-06_13-42-11.log
    eval_2026-06-07_16-27-51.log
    eval_2026-06-07_16-30-03.log
    eval_2026-06-07_18-08-02.log

```

## Style De Codage
- Toutes les regles du principe SOLID doivent etre respectees.
- Commentaire de code et les logs sur le terminal doivent être en **Anglais**. L'anglais doit etre simple et facile
  a comprendre pour un debutant qui ne parle pas correctement l'anglais. Pas de mots compliques. Necessite de rediger
  des phrases simple de niveau A.
- Afficher plus de logs resumant les hyperparamettres du programme et de toute les etapes d'execution. Le programme
  doit favoriser une tracabilite dans la journalisation.
- Dans les commentaires de code et docstring, il est interdiction d'utiliser des symboles, emoji et caracteres
  speciaux qui n'existent pas sur les **touches du clavier anglais**. L'anglais doit etre simple et facile
  a comprendre pour un debutant qui ne parle pas correctement l'anglais. Pas de mots compliques. Necessite de rediger
  des phrases simple de niveau A.
- Le nombre d'instruction dans une fonction ou methode ne doit pas depasser **16**. Il faut donc etre tres modulaire
  dans l'ecriture de la logique du code afin de facilite la maintenabilité.
- Utilisation de classe pour programmer chaque composent et utilisation de la programmation modulaire.
- Chaque classe doit avoir une et une seule responsabilite : une classes doit realiser une et une seule tâche
  (operation) et doit tres bien le realiser -- Single Responsability.
- Pas de boucle `while` sans un compteur pour permettre de limiter et stopper en cas de boucle infinie.
- Apres l'appel de toute fonction ou methode de classe, chaque sortie de fonction doit etre verifiee a l'aide
  de condition necessaires avant de passer a la prochaine instruction.
- Rediger un Readme complete et bien detaille pour l'installation et la prise en main du projet.
- Rediger un document pour debutant expliquant dans les moindres detailles avec beaucoup d'exemples les conceptes
  et le fonctionnement du modele et ses composents dans le style pédago-bavard de Mateo21 — analogies, fausses et naives
  questions du lecteur, encadrés, et beaucoup d'explications dans deux langues differentes :
  Anglaise Et Francais. Ces deux documents doivent se trouver dans le dossier `docs/` a la racine du depot de projet.
- Rediger les testes unitaires pour tester chaque fonction et methode de classe.
- Rediger les testes unitaires pour tester la precision de tous les metriques de mesure
  (perte, evaluation de perf, etc).
- Rediger les testes unitaires pour testes la vitesse d'execution et les formes de tenseur retournes par le modele
  et par chacune de ses composents implementes dans `modules/`.
- Generer deux configurations pour les trois architectures de materiel (device): CPU, GPU CUDA NVIDIA et GPU ROCm AMD.
- Architecture general des fichiers de code :
  ```
  README.md
  docs/
    en_concepts.md
    fr_concepts.md
  cpu/
    configs/
      hdf5.yaml
      train.yaml
      eval.yaml
      export.yaml
  gpu/   # configuration pour GPU NVIDIA et GPU AMD
    configs/
      hdf5.yaml
      train.yaml
      eval.yaml
      export.yaml
  tests/
  src/
    nom_model/
      entrypoints/
        buildds.py       # Permet de builder une dataset existant en un fichier HDF5. Cela permettra d'accelerer
                         # le processus d'entrainement.
        train.py         # Programme d'entrainement, comportant les trois etapes : entrainement, validation
                         # (sur une fraction des donnees de teste), et evaluation finale sur l'integralite
                         # des donnees de teste.
        evaluate.py      # Programme d'evaluation du modele sur l'integralite de la dataset de teste.
        exportmodel.py   # Exporter en modele ONNX.
        inference.py     # Inference dont le modele ONNX est charge. Il s'agit ici d'un scripte totalement isole
                         # et autonome. On doit pouvoir copier et colle ce scripte ailleur dans un autre programme
                         # et le faire fonctionner.

      dataset/          # Module comportant l'implementation des composents qui interviennent dans la construction
                        # de la dataset : chargement des donnees (features, target si possible),
                        # transformation des donnees (normalization, redimentionnement, etc),
                        # augmentation des donnees (application de bruits, effet de bord, etc).
                        # Ou charge directement des donnees ready to train et deja transformees et si possible, deja
                        # augmentees deux fichiers HDF5 (train.h5 et test.h5).

      modules/           # Module comportant l'implementation des composents du modele.
      model.py           # L'implementation complete du ou des modeles.

      lossfn.py         # Implementation complete de ou des fonction d'erreurs.

      metrics/          # Module comportant l'implementation de chacune des metriques de validation et d'evaluation.

      optimizers.py     # Definition, implementation et construction avancee d'instance d'optimiseur necessaire
                        # a l'entrainement du modele. Ici on doit pouvoir construire des groupes d'optimisation
                        # pour n'importe quelle algorithme d'optimisation selectionne (Adam, AdamW, SGD, etc...).

      lr_shedulers.py   # Definition, implementation et construction avancee d'instance de LR scheduler adapte
                        # et necessaire pour entrainer le modele en question et avoir une performance
                        # de qualite production.

      logging.py        # Configuration des loggings et feedbacks.
      plotting.py       # Definition de fonctionnalite de trace de graphique.
      onnx_export.py    # Definition de fonctionnalite d'exportation du modele en ONNX.
  ```

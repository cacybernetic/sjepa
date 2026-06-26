# Comprendre S-JEPA, pas a pas (Francais)

Bonjour et bienvenue ! Ce guide explique S-JEPA depuis zero. Pas de maths
lourdes, beaucoup de petites etapes et des analogies. Si tu n'as jamais entraine
de modele de parole, ce guide est pour toi. Prends ton temps et lis doucement.

> **Ce que tu vas apprendre.** Quel probleme S-JEPA resout, ce que fait chaque
> partie du modele, et comment l'entrainement marche vraiment a l'interieur.

---

## 1. La vue d'ensemble

Imagine que tu veux qu'un ordinateur "comprenne" la parole. La methode classique
a besoin de beaucoup de donnees annotees : quelqu'un doit ecrire ce qui est dit
dans chaque clip. C'est lent et cher.

**L'apprentissage auto-supervise** est une astuce pour apprendre *sans
annotation*. Le modele joue a un jeu avec lui-meme : on cache une partie du son
et on demande au modele de deviner ce qui est cache. Pour gagner, le modele doit
apprendre la structure de la parole.

> *Question naive : "Deviner la partie cachee ? Comme remplir un mot manquant
> dans une phrase ?"*
>
> Oui, exactement. Mais ici les "mots" sont de petites trames de son de 20
> millisecondes, et le modele ne devine pas le son brut. Il devine une
> **etiquette douce** pour chaque trame. On verra tres vite ce que c'est.

---

## 2. Les trames : couper le son en petits morceaux

Le son est une longue onde. On la coupe en petites trames de 20 ms chacune. A 16
kHz, une trame fait 320 echantillons. Un clip de 10 secondes donne environ 500
trames.

Un petit **frontal CNN** fait cette decoupe et transforme chaque trame en un
vecteur de nombres. Vois-le comme une machine qui lit l'onde et ecrit une "fiche
resume" par trame.

```
onde brute  ->  [frontal CNN]  ->  trame 1, trame 2, trame 3, ... trame T
```

---

## 3. Les trois parties entrainees

S-JEPA a trois parties qui apprennent ensemble :

| Partie         | Symbole    | Travail                                            |
|----------------|------------|----------------------------------------------------|
| Encodeur       | `f_phi`    | Lire les trames et construire des representations. |
| Predicteur     | `h_psi`    | Remplir les trames cachees a partir des visibles.  |
| Tete de cluster| `g_omega`  | Transformer une trame en `K` scores de cluster.    |

> **Analogie.** L'encodeur est un *lecteur* qui comprend le contexte. Le
> predicteur est un *devineur* qui remplit les trous. La tete de cluster est un
> *traducteur* qui transforme une trame en un vote sur `K` boites.

Apres l'entrainement, on **garde seulement l'encodeur**. Le predicteur et la
tete de cluster etaient juste des aides. On les jette, comme un echafaudage une
fois la maison construite.

---

## 4. Le masquage : cacher des trames volontairement

Avant que l'encodeur tourne, on choisit des trames a cacher. On les cache par
**blocs** (de petites suites de trames voisines), jusqu'a cacher environ 65% des
trames. L'encodeur voit des zeros aux trames cachees.

Ensuite le predicteur met un **jeton de masque** appris aux endroits caches,
ajoute l'information de position, et essaie de les remplir.

> *Question naive : "Pourquoi cacher autant de trames ? 65%, c'est enorme !"*
>
> Si on cache trop peu, le modele peut tricher en copiant les voisins. Cacher
> beaucoup l'oblige a vraiment comprendre la parole pour combler les trous.

---

## 5. Les etiquettes douces : le coeur de S-JEPA

Voici l'idee cle. Quand on demande "qu'y a-t-il dans cette trame cachee ?", les
anciennes methodes (comme HuBERT) repondent avec **un seul** numero de cluster,
par exemple "cluster 42". C'est une **etiquette dure**.

Mais la parole est floue aux frontieres. Entre deux sons, une trame peut etre "a
moitie cluster 42, a moitie cluster 87". Une etiquette dure doit en choisir un
seul et jeter le doute.

S-JEPA utilise plutot une **etiquette douce** : une probabilite complete sur tous
les `K` clusters, par exemple `42 -> 0.55, 87 -> 0.40, autres -> petit`. Cela
garde le doute.

> **Encadre : d'ou viennent les etiquettes douces ?**
>
> D'un **modele de melange gaussien (GMM)**. Un GMM est un ensemble de `K` taches
> douces dans l'espace des features. Pour une trame, il dit a quel point la trame
> appartient a chaque tache. Ces `K` nombres (leur somme fait 1) sont la cible
> douce.

Le predicteur sort ses propres `K` nombres (un softmax). L'entrainement pousse
les nombres du predicteur a coller a ceux du GMM. La distance entre deux listes
de probabilites est la **divergence KL**. Cette seule distance est toute la
perte.

```
perte = KL( cible douce GMM  ||  softmax du predicteur )   aux trames cachees
```

---

## 6. Deux phases (un seul entrainement continu)

HuBERT fait deux passes de clustering : d'abord sur des features simples, puis
sur de meilleures features. S-JEPA garde cette idee mais la fait en douceur, en
deux phases.

### Phase 1 : le GMM MFCC fige

- On calcule des features **MFCC** (un resume classique de 39 nombres du son)
  pour beaucoup de trames.
- On ajuste un GMM avec `K = 100` taches sur ces features. Il est **fige** : il
  ne change jamais pendant la Phase 1.
- On entraine l'encodeur et le predicteur a coller a ce GMM aux trames cachees.

### Phase 2 : le GMM en ligne sur l'encodeur

- Maintenant le GMM travaille sur les **features de l'encodeur**, plus riches que
  les MFCC. On utilise `K = 500` taches.
- Le GMM est **en ligne** : apres chaque lot, il bouge lentement pour suivre
  l'encodeur. Pas besoin de tout re-clusteriser.
- Une seconde copie lente de l'encodeur (l'**encodeur EMA**) nourrit le GMM. Elle
  change lentement, donc les cibles restent stables.
- La **couche** de l'encodeur utilisee par le GMM est choisie automatiquement,
  grace a un signal appele **rang effectif** (a quel point une couche est riche).

> *Question naive : "Pourquoi une copie lente de l'encodeur ?"*
>
> Si la cible et l'eleve changent a la meme vitesse, l'entrainement devient un
> chien qui court apres sa queue. La copie lente donne une cible plus stable.

---

## 7. Comment marche une etape d'entrainement

Pour chaque lot d'audio :

1. Faire une copie bruitee de l'onde (augmentation) pour l'entree de l'encodeur.
   Garder l'onde propre pour la cible.
2. Construire la cible douce a partir de l'onde propre (GMM MFCC en Phase 1,
   encodeur EMA + GMM en ligne en Phase 2).
3. Choisir le masque par blocs et faire tourner l'encodeur sur l'onde bruitee.
4. Le predicteur remplit les trames cachees.
5. La tete de cluster transforme les trames du predicteur en `K` logits.
6. Calculer la perte KL aux trames cachees et faire une etape de gradient.
7. (Phase 2) Mettre a jour un peu l'encodeur EMA lent et le GMM en ligne.

---

## 8. Les astuces d'entrainement dans la config

- **Accumulation de gradient.** Les petits GPU ne tiennent pas un grand lot. On
  additionne les gradients de plusieurs petits lots et on fait une seule etape.
  Le lot effectif est `batch_size x grad_accum`. On fait aussi une etape a la fin
  d'une epoque s'il reste des gradients en attente.
- **Clipping du gradient.** On limite la taille du gradient pour qu'un mauvais
  lot ne fasse pas exploser l'entrainement.
- **Planning du learning rate.** Le taux monte pendant un court *warmup*, puis
  descend selon une courbe *cosinus*. Debut doux, fin douce.
- **Checkpoints.** A chaque epoque on sauve tout l'etat (modele, optimiseur,
  scheduler, GMM, EMA). On garde seulement les plus recents et on peut
  **reprendre** exactement la ou on s'est arrete.
- **Meilleur modele.** On surveille une metrique de validation (par exemple
  `kl`, plus c'est bas mieux c'est) et on sauve `best.pt` des qu'elle s'ameliore.
- **Courbes d'historique.** Apres chaque epoque on trace les courbes train vs
  validation. Si la courbe de validation monte pendant que celle d'entrainement
  descend, le modele sur-apprend (overfitting).

---

## 9. Ce que veulent dire les metriques

- **kl** : a quel point la prediction est loin de la cible douce. Plus bas est
  mieux. C'est le score principal.
- **top1** : a quelle frequence le meilleur cluster predit egale le meilleur
  cluster cible.
- **entropy_bits** : a quel point la prediction est incertaine, en bits. `1 bit`
  veut dire une egalite parfaite entre deux clusters. L'article montre que la
  parole a beaucoup de telles egalites aux frontieres des sons, ce que les
  etiquettes douces gardent justement.
- **effective_rank** : combien de directions une couche utilise vraiment. Sert a
  choisir la couche du GMM en Phase 2.

---

## 10. Apres l'entrainement : l'inference

L'entrainement garde seulement l'encodeur. On l'exporte en **ONNX**, un format
portable qui tourne vite sans Python. Ensuite `infersjepa` charge le fichier
ONNX, lit un clip audio, et affiche les features par trame. Ces features sont ce
que tu donnes a une petite tete de tache (reconnaissance de parole, emotion,
etc.).

> **Bravo, tu as fini !** Tu connais maintenant toute l'histoire de S-JEPA :
> trames, masquage, cibles douces GMM, perte KL, deux phases, et les astuces
> d'entrainement. Ouvre `cpu/configs/train.yaml`, change quelques valeurs, et
> lance `trainsjepa -c cpu/configs/train.yaml` pour le voir en vrai.

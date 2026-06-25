# Style De Codage
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
  dans l'ecriture de la logique du code afin de facilite la **maintenabilité**.
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
- Rediger les testes unitaires pour tester la precision et temps d'execution des modules importants
  avec tous les metriques de mesure (perte, evaluation de perf, etc).

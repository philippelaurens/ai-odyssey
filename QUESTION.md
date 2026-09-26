# Réflexion sur l'évolution de l'IA : Rupture conceptuelle ou illusion du Scaling ?

## Question

Quand je songe à l'effervescence autour du Machine Learning des années 90-début 2000, celui du Deep Learning des années 2006-2017, avec les conséquences sur le NLP qui conduisent aux transformers, je me pose une question.

L'époque actuelle (2017-2026) a vu la découverte de l'IA par le grand public : les LLMs qui arrivent en force, les FMs multimodaux, les RAG, les Agents, les annonces de sorties de nouveaux LLMs frontier.

Mais cette époque est-elle si riche que cela, ou le surinvestissement (par centaines de milliards dans le scaling de transformers) est-il comme un arbre qui cacherait la forêt ?

---

## Réponse (Gemini 3.6 Flash)

C'est une excellente lecture de l'histoire récente de l'IA. Pour répondre sans détours : **oui, le surinvestissement sur le *scaling* des Transformers masque une réelle stagnation conceptuelle**, mais cela ne veut pas dire qu'il ne se passe rien.

On vit une époque paradoxale : c'est la période de **déploiement et d'impact industriel** la plus violente de l'histoire de l'informatique, mais sur le plan de la **fondation théorique**, c'est peut-être la moins inventive des trois.

---

### La comparaison des trois époques

| Époque | Moteur principal | Nature des avancées |
| :--- | :--- | :--- |
| **1990-2005** *(ML classique)* | Mathématiques & Statistiques | **Ruptures conceptuelles majeures** : SVM, Kernel Tricks, Random Forests, Théorie de la généralisation (VC-dimension). On posait les bases rigoureuses. |
| **2006-2017** *(Deep Learning)* | Architectures & Algorithmes | **Ruptures d'ingénierie et d'idées** : AlexNet, CNNs, LSTM/GRU, GANs, Variational Autoencoders, Attention mechanism et enfin l'article *Attention Is All You Need* (2017). |
| **2017-2026** *(Scaling & FMs)* | **Brute Force & Capital** | **Optimisation et ingénierie système** : Scaling laws, RLHF/DPO, Quantization (GGUF/AWQ), RAG, Orchestration d'Agents. |

---

### L'arbre du Scaling qui cache la forêt

Le cœur du problème réside dans ce qu'on appelle la **"Scaling Law"**. Depuis 2020, le secteur s'est rendu compte qu'augmenter le volume de données et les FLOPS sur une même architecture (le Transformer) donnait des gains prévisibles. 

Résultat : la majorité de la recherche d'élite (et des centaines de milliards de dollars de Venture Capital et de Capex) s'est concentrée sur la **résolution de problèmes d'ingénierie brute** plutôt que sur la recherche de nouveaux paradigmes.

1. **Une architecture verrouillée :** L'écrasante majorité des modèles *frontier* en 2026 repose encore sur le Transformer (ou des variantes MoE / Mixture-of-Experts).
2. **Le rendement décroissant du calcul :** Entraîner un modèle 10x plus gros ne le rend plus 10x plus intelligent. On touche aux limites de la qualité des données de texte (data wall) et aux limites énergétiques.
3. **Le pansement du RAG et des Agents :** Les Agents, le RAG ou les chaînes de pensée (*Chain of Thought*) ne sont pas de nouvelles théories de l'intelligence. Ce sont des **artifices logiciels et des architectures système** conçus pour contourner les faiblesses structurelles des LLMs (hallucinations, fenêtre de contexte limitée, manque de mémoire long terme).

---

### Ce qui pousse sous l'arbre (les vraies pistes d'avenir)

Même si le rouleau compresseur des Transformers monopolise l'attention et les budgets, cette période prépare le passage à l'étape suivante, notamment poussée par la nécessité d'échapper à la complexité quadratique de l'attention $O(N^2)$ :

* **Les architectures alternatives :** Les modèles de type *State Space Models* (Mamba, RWKV, Hyena) cherchent à retrouver une complexité linéaire $O(N)$ pour traiter des contextes infinis sans exploser les serveurs.
* **L'IA Fondée sur la Physique et le Monde (World Models) :** Les travaux autour de l'apprentissage auto-supervisé pour la vidéo/monde physique (comme les travaux d'Yann LeCun sur JEPA / V-JEPA) cherchent à dépasser la simple prédiction du "prochain token" pour capturer une réelle compréhension du monde matériel.
* **Le Raisonnement/Test-Time Compute :** Le basculement d'une IA qui "sait" tout au moment de l'entraînement à une IA qui prend du temps de calcul au moment de la génération pour vérifier, planifier et raisonner de manière logique (ex. recherche dans des arbres de preuves).

---

### Le verdict

L'époque actuelle n'est pas pauvre, mais elle a changé de nature. Nous sommes passés de la **phase des chercheurs et des mathématiciens** (1990-2017) à la **phase des ingénieurs système et des capitaines d'industrie** (2017-2026).

L'arbre du Transformer est géant et produit des fruits immédiatement exploitables par la société, mais ses racines ne s'enfoncent plus. La prochaine vraie rupture ne viendra pas d'un cluster d'un million de GPUs de plus, mais d'une idée mathématique ou architecturale à laquelle le surinvestissement actuel refuse encore de prêter attention.

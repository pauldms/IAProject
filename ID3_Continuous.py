# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:17:01 2020

@author: pauld
"""


class NoeudDeDecision:
    """ Un noeud dans un arbre de dÃ©cision. 
    
        This is an updated version from the one in the book (Intelligence Artificielle par la pratique).
        Specifically, if we can not classify a data point, we return the predominant class (see lines 53 - 56). 
    """

    def __init__(self, attribut, donnees, p_class, enfants=None):
        """
            :param attribut: l'attribut de partitionnement du noeud (``None`` si\
            le noeud est un noeud terminal).
            :param list donnees: la liste des donnÃ©es qui tombent dans la\
            sous-classification du noeud.
            :param enfants: un dictionnaire associant un fils (sous-noeud) Ã \
            chaque valeur de l'attribut du noeud (``None`` si le\
            noeud est terminal).
        """

        self.attribut = attribut
        self.donnees = donnees
        self.enfants = enfants
        self.p_class = p_class

    def terminal(self):
        """ VÃ©rifie si le noeud courant est terminal. """

        return self.enfants is None

    def classe(self):
        """ Si le noeud est terminal, retourne la classe des donnÃ©es qui\
            tombent dans la sous-classification (dans ce cas, toutes les\
            donnÃ©es font partie de la mÃªme classe. 
        """

        if self.terminal():
            return self.donnees[0][0]

    def classifie(self, donnee):
        """ Classifie une donnÃ©e Ã  l'aide de l'arbre de dÃ©cision duquel le noeud\
            courant est la racine.

            :param donnee: la donnÃ©e Ã  classifier.
            :return: la classe de la donnÃ©e selon le noeud de dÃ©cision courant.
        """

        rep = ''
        if self.terminal():
            rep += 'Alors {}'.format(self.classe().upper())
        else:
            valeur = donnee[self.attribut]
            enfant = self.enfants[valeur]
            rep += 'Si {} = {}, '.format(self.attribut, valeur.upper())
            try:
                rep += enfant.classifie(donnee)
            except:
                rep += self.p_class
        return rep

    def repr_arbre(self, level=0):
        """ ReprÃ©sentation sous forme de string de l'arbre de dÃ©cision duquel\
            le noeud courant est la racine. 
        """

        rep = ''
        if self.terminal():
            rep += '---'*level
            rep += 'Alors {}\n'.format(self.classe().upper())
            rep += '---'*level
            rep += 'DÃ©cision basÃ©e sur les donnÃ©es:\n'
            for donnee in self.donnees:
                rep += '---'*level
                rep += str(donnee) + '\n' 

        else:
            for valeur, enfant in self.enfants.items():
                rep += '---'*level
                rep += 'Si {} = {}: \n'.format(self.attribut, valeur.upper())
                rep += enfant.repr_arbre(level+1)

        return rep

    def __repr__(self):
        """ ReprÃ©sentation sous forme de string de l'arbre de dÃ©cision duquel\
            le noeud courant est la racine. 
        """

        return str(self.repr_arbre(level=0))
    
#CREATION DE LA CLASSE ID3
        
from math import log

class ID3:
    """ Algorithme ID3. 

        This is an updated version from the one in the book (Intelligence Artificielle par la pratique).
        Specifically, in construit_arbre_recur(), if donnees == [] (line 70), it returns a terminal node with the predominant class of the dataset -- as computed in construit_arbre() -- instead of returning None.
        Moreover, the predominant class is also passed as a parameter to NoeudDeDecision().
    """
    
    def construit_arbre(self, donnees):
        """ Construit un arbre de décision à partir des données d'apprentissage.

            :param list donnees: les données d'apprentissage\
            ``[classe, {attribut -> valeur}, ...]``.
            :return: une instance de NoeudDeDecision correspondant à la racine de\
            l'arbre de décision.
        """
        
        # Nous devons extraire les domaines de valeur des 
        # attributs, puisqu'ils sont nécessaires pour 
        # construire l'arbre.
        attributs = {}
        for donnee in donnees:
            for attribut, valeur in donnee[1].items():
                valeurs = attributs.get(attribut)
                if valeurs is None:
                    valeurs = set()
                    attributs[attribut] = valeurs
                valeurs.add(valeur)

        # Find the predominant class
        classes = set([row[0] for row in donnees])
        # print(classes)
        predominant_class_counter = -1
        for c in classes:
            # print([row[0] for row in donnees].count(c))
            if [row[0] for row in donnees].count(c) >= predominant_class_counter:
                predominant_class_counter = [row[0] for row in donnees].count(c)
                predominant_class = c
        # print(predominant_class)
            
        arbre = self.construit_arbre_recur(donnees, attributs, predominant_class)

        return arbre

    def construit_arbre_recur(self, donnees, attributs, predominant_class):
        """ Construit rédurcivement un arbre de décision à partir 
            des données d'apprentissage et d'un dictionnaire liant
            les attributs à la liste de leurs valeurs possibles.

            :param list donnees: les données d'apprentissage\
            ``[classe, {attribut -> valeur}, ...]``.
            :param attributs: un dictionnaire qui associe chaque\
            attribut A à son domaine de valeurs a_j.
            :return: une instance de NoeudDeDecision correspondant à la racine de\
            l'arbre de décision.
        """
        
        def classe_unique(donnees):
            """ Vérifie que toutes les données appartiennent à la même classe. """
            
            if len(donnees) == 0:
                return True 
            premiere_classe = donnees[0][0]
            for donnee in donnees:
                if donnee[0] != premiere_classe:
                    return False 
            return True

        if donnees == []:
            return NoeudDeDecision(None, [str(predominant_class), dict()], str(predominant_class))

        # Si toutes les données restantes font partie de la même classe,
        # on peut retourner un noeud terminal.         
        elif classe_unique(donnees):
            return NoeudDeDecision(None, donnees, str(predominant_class))
            
        else:
            h_C_As_attribs_values = []
            for attribut in attributs :
                for value in attributs[attribut] :
                    h_C_As_attribs_values.append(
                        (self.h_C_aj(donnees, attribut, value), 
                               (attribut,value)))
            # Sélectionne l'attribut qui réduit au maximum l'entropie.
            ##h_C_As_attribs = [(self.h_C_A(donnees, attribut, attributs[attribut]), 
            ##                   attribut) for attribut in attributs]
            pair = min(h_C_As_attribs_values, key=lambda h_a: h_a[0])[1]

            # Crée les sous-arbres de manière récursive.
            ##attributs_restants = attributs.copy()
            ##del attributs_restants[attribut] 

            partitions = self.divise(donnees, pair[0], pair[1])
            
            enfants = {}
            enfants[0] = self.construit_arbre_recur(partitions[0],
                                                             attributs,
                                                             predominant_class)
            enfants[1] = self.construit_arbre_recur(partitions[1],
                                                             attributs,
                                                             predominant_class)

            return NoeudDeDecision(attribut, donnees, str(predominant_class), enfants)
        
        
    def divise(self,donnees,attribut,valeur) :
        partitions = ([],[])
        for donnee in donnees :
            if (donnee[1][attribut] < valeur) :
                partitions[0].append(donnee)
            else : 
                partitions[1].append(donnee)
        
        return partitions
    
    def partitionne(self, donnees, attribut, valeurs):
        """ Partitionne les données sur les valeurs a_j de l'attribut A.

            :param list donnees: les données à partitioner.
            :param attribut: l'attribut A de partitionnement.
            :param list valeurs: les valeurs a_j de l'attribut A.
            :return: un dictionnaire qui associe à chaque valeur a_j de\
            l'attribut A une liste l_j contenant les données pour lesquelles A\
            vaut a_j.
        """
        partitions = {valeur: [] for valeur in valeurs}
        
        for donnee in donnees:
            partition = partitions[donnee[1][attribut]]
            partition.append(donnee)
            
        return partitions

    def p_aj(self, donnees, attribut, valeur):
        """ p(a_j) - la probabilité que la valeur de l'attribut A soit a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.            
            :return: p(a_j)
        """
        # Nombre de données.
        nombre_donnees = len(donnees)
        
        # Permet d'éviter les divisions par 0.
        if nombre_donnees == 0:
            return 0.0
        
        # Nombre d'occurrences de la valeur a_j parmi les données.
        nombre_aj = 0
        for donnee in donnees:
            if donnee[1][attribut] == valeur:
                nombre_aj += 1

        # p(a_j) = nombre d'occurrences de la valeur a_j parmi les données / 
        #          nombre de données.
        return nombre_aj / nombre_donnees

    def p_ci_aj(self, donnees, attribut, valeur, classe):
        """ p(c_i|a_j) - la probabilité conditionnelle que la classe C soit c_i\
            étant donné que l'attribut A vaut a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :param classe: la valeur c_i de la classe C.
            :return: p(c_i | a_j)
        """
        # Nombre d'occurrences de la valeur a_j parmi les données.
        donnees_aj = [donnee for donnee in donnees if donnee[1][attribut] == valeur]
        nombre_aj = len(donnees_aj)
        
        # Permet d'éviter les divisions par 0.
        if nombre_aj == 0:
            return 0
        
        # Nombre d'occurrences de la classe c_i parmi les données pour lesquelles 
        # A vaut a_j.
        donnees_ci = [donnee for donnee in donnees_aj if donnee[0] == classe]
        nombre_ci = len(donnees_ci)

        # p(c_i|a_j) = nombre d'occurrences de la classe c_i parmi les données 
        #              pour lesquelles A vaut a_j /
        #              nombre d'occurrences de la valeur a_j parmi les données.
        return nombre_ci / nombre_aj

    def h_C_aj(self, donnees, attribut, valeur):
        """ H(C|a_j) - l'entropie de la classe parmi les données pour lesquelles\
            l'attribut A vaut a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :return: H(C|a_j)
        """
        # Les classes attestées dans les exemples.
        classes = list(set([donnee[0] for donnee in donnees]))
        
        # Calcule p(c_i|a_j) pour chaque classe c_i.
        p_ci_ajs = [self.p_ci_aj(donnees, attribut, valeur, classe) 
                    for classe in classes]

        # Si p vaut 0 -> plog(p) vaut 0.
        return -sum([p_ci_aj * log(p_ci_aj, 2.0) 
                    for p_ci_aj in p_ci_ajs 
                    if p_ci_aj != 0])

    def h_C_A(self, donnees, attribut, valeurs):
        """ H(C|A) - l'entropie de la classe après avoir choisi de partitionner\
            les données suivant les valeurs de l'attribut A.
            
            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param list valeurs: les valeurs a_j de l'attribut A.
            :return: H(C|A)
        """
        # Calcule P(a_j) pour chaque valeur a_j de l'attribut A.
        p_ajs = [self.p_aj(donnees, attribut, valeur) for valeur in valeurs]

        # Calcule H_C_aj pour chaque valeur a_j de l'attribut A.
        h_c_ajs = [self.h_C_aj(donnees, attribut, valeur) 
                   for valeur in valeurs]

        return sum([p_aj * h_c_aj for p_aj, h_c_aj in zip(p_ajs, h_c_ajs)])
    
    
donnees = []
import csv

with open('train_continuous.csv') as f:
    f_csv = csv.reader(f)
    en_tetes = next(f_csv)
    for ligne in f_csv:
        target = ligne.pop()
        donnees.append([target,{'age' : ligne[0], 'sex' : ligne[1], 'cp' : ligne[2], 'trestbps' : ligne[3],'chol' :ligne[4], 'fbs':ligne[5] , 'restecg':ligne[6], 'thalach':ligne[7], 'exang':ligne[8], 'oldpeak':ligne[9], 'slope' :ligne[10], 'ca':ligne[11], 'thal':ligne[12] }])

#print(donnees)


    
#ENTRAINER LE MODELE
        
id3 = ID3()
arbre = id3.construit_arbre(donnees)

('Arbre de décision :')
(arbre)
()

#QUESTION 2

#IMPORTER LES DONNEES DE TEST

donneestest = []
targets = []
import csv

with open('test_public_continuous.csv') as f:
    f_csv = csv.reader(f)
    en_tetes = next(f_csv)
    for ligne in f_csv:
        target = ligne.pop()
        targets.append(target)
        donneestest.append({'age' : ligne[0], 'sex' : ligne[1], 'cp' : ligne[2], 'trestbps' : ligne[3],'chol' :ligne[4], 'fbs':ligne[5] , 'restecg':ligne[6], 'thalach':ligne[7], 'exang':ligne[8], 'oldpeak':ligne[9], 'slope' :ligne[10], 'ca':ligne[11], 'thal':ligne[12] })

#print(donneestest)

# TESTER LES DONNEES ET CALCULER LE POURCENTAGE DE BONNES REPONSES
        
S = 0
for k in range(len(donneestest)):
    #print(donneestest[k])
    result = arbre.classifie(donneestest[k])
    print(result)
    resultat = result[-1] 
    #print(resultat)

    if resultat == targets[k]:
        S += 1
pourcentage = S*100/len(donneestest)

print(pourcentage)
    
    
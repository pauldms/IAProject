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
            val_tri = list(self.enfants.keys())[1]
            
            if(float(valeur) < float(val_tri)):
                enfant = self.enfants['0']
                rep += 'Si {} <= {}, '.format(self.attribut, val_tri)
            else : 
                enfant = self.enfants[val_tri]
                rep += 'Si {} > {}, '.format(self.attribut,val_tri)
            
           
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
        predominant_class_counter = -1
        for c in classes:
            if [row[0] for row in donnees].count(c) >= predominant_class_counter:
                predominant_class_counter = [row[0] for row in donnees].count(c)
                predominant_class = c
            
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
                        (self.h_C_A(donnees, attribut, value), 
                               (attribut,value)))
           
                    
            
            # Sélectionne l'attribut qui réduit au maximum l'entropie.
            
                    
            #filtered_H_C_As = [p for p in h_C_As_attribs_values if p[0] > 0]
            #attribut,valeur = min(filtered_H_C_As, key=lambda h_a: h_a[0])[1]
            attribut,valeur = min(h_C_As_attribs_values, key=lambda h_a: h_a[0])[1]
        
            
            partitions = self.divise(donnees, attribut,valeur)
            
            enfants = {}
            
            print(str(len(partitions['inf'])) + " length " + str(len(partitions['sup'])))
            if(len(partitions['inf'])==0 or len(partitions['sup'])==0):
                rest = donnees.copy();
                first = [rest.pop()]
                print("first " + str(len(first)) + str(first))
                print("rest " + str(len(rest)) + str(rest))
                enfants['0'] = self.construit_arbre_recur(first,
                                                             attributs,
                                                             predominant_class)
                enfants[str(valeur)] = self.construit_arbre_recur(rest,
                                                             attributs,
                                                             predominant_class)
            
            else :
            
                enfants['0'] = self.construit_arbre_recur(partitions['inf'],
                                                                 attributs,
                                                                 predominant_class)
                enfants[str(valeur)] = self.construit_arbre_recur(partitions['sup'],
                                                                 attributs,
                                                                 predominant_class)

            return NoeudDeDecision(attribut, donnees, str(predominant_class), enfants)
        
        
    
    def divise(self,donnees,attribut,valeur) :
        partitions = {'inf': [], 'sup': []} 
        for donnee in donnees :
            if (float(donnee[1][attribut]) < float(valeur)) :
                partitions['inf'].append(donnee)
            else : 
                partitions['sup'].append(donnee)
        return partitions
    
    
    def p_inf_aj(self, donnees, attribut, valeur):
        """ p(<=a_j) - la probabilité que la valeur de l'attribut A 
            soit inférieure ou égale à a_j.

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
        
        # Nombre d'occurrences de valeurs <= a_j parmi les données.
        nombre_aj = 0
        for donnee in donnees:
            if donnee[1][attribut] < valeur:
                nombre_aj += 1

        # p(a_j) = nombre d'occurrences de valeurs <= a_j parmi les données / 
        #          nombre de données.
        return nombre_aj / nombre_donnees
    
    def p_sup_aj(self,donnees,attribut,valeur):
        """ p(>a_j) - la probabilité que la valeur de l'attribut A 
            soit suppérieure à a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.            
            :return: p(a_j)
        """
        return 1 - self.p_inf_aj(donnees,attribut,valeur)
    
    def p_ci_inf_aj(self, donnees, attribut, valeur, classe):
        """ p(c_i|<=a_j) - la probabilité conditionnelle que la classe C soit c_i\
            étant donné que l'attribut A est inférieur ou égal à a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :param classe: la valeur c_i de la classe C.
            :return: p(c_i | a_j)
        """
        # Nombre d'occurrences de valeurs <= a_j parmi les données.
        donnees_aj = [donnee for donnee in donnees if donnee[1][attribut] < valeur]
        nombre_aj = len(donnees_aj)
        
        # Permet d'éviter les divisions par 0.
        if nombre_aj == 0:
            return 0
        
        # Nombre d'occurrences de la classe c_i parmi les données pour lesquelles 
        # A est inférieur ou égal à a_j.
        donnees_ci = [donnee for donnee in donnees_aj if donnee[0] == classe]
        nombre_ci = len(donnees_ci)

        # p(c_i|a_j) = nombre d'occurrences de la classe c_i parmi les données 
        #              pour lesquelles A est inférieur ou égal à a_j /
        #              nombre d'occurrences de valeurs <= a_j parmi les données.
        return nombre_ci / nombre_aj
    
    def p_ci_sup_aj(self, donnees, attribut, valeur, classe):
        """ p(c_i|>a_j) - la probabilité conditionnelle que la classe C soit c_i\
            étant donné que l'attribut A est supérieur à a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :param classe: la valeur c_i de la classe C.
            :return: p(c_i | a_j)
        """
        # Nombre d'occurrences de valeurs > a_j parmi les données.
        donnees_aj = [donnee for donnee in donnees if donnee[1][attribut] >= valeur]
        nombre_aj = len(donnees_aj)
        
        # Permet d'éviter les divisions par 0.
        if nombre_aj == 0:
            return 0
        
        # Nombre d'occurrences de la classe c_i parmi les données pour lesquelles 
        # A est > à a_j.
        donnees_ci = [donnee for donnee in donnees_aj if donnee[0] == classe]
        nombre_ci = len(donnees_ci)

        # p(c_i|a_j) = nombre d'occurrences de la classe c_i parmi les données 
        #              pour lesquelles A est > à a_j /
        #              nombre d'occurrences de valeurs > a_j parmi les données.
        return nombre_ci / nombre_aj
    
    def h_C_inf_aj(self, donnees, attribut, valeur):
        """ H(C|<=a_j) - l'entropie de la classe parmi les données pour lesquelles\
            l'attribut A est inférieur ou égal à a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :return: H(C|a_j)
        """
        # Les classes attestées dans les exemples.
        classes = list(set([donnee[0] for donnee in donnees]))
        
        # Calcule p(c_i|<=a_j) pour chaque classe c_i.
        p_ci_ajs = [self.p_ci_inf_aj(donnees, attribut, valeur, classe) 
                    for classe in classes]

        # Si p vaut 0 -> plog(p) vaut 0.
        return -sum([p_ci_aj * log(p_ci_aj, 2.0) 
                    for p_ci_aj in p_ci_ajs 
                    if p_ci_aj != 0])
    
    def h_C_sup_aj(self, donnees, attribut, valeur):
        """ H(C|<=a_j) - l'entropie de la classe parmi les données pour lesquelles\
            l'attribut A est inférieur ou égal à a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :return: H(C|a_j)
        """
        # Les classes attestées dans les exemples.
        classes = list(set([donnee[0] for donnee in donnees]))
        
        # Calcule p(c_i|<=a_j) pour chaque classe c_i.
        p_ci_ajs = [self.p_ci_sup_aj(donnees, attribut, valeur, classe) 
                    for classe in classes]

        # Si p vaut 0 -> plog(p) vaut 0.
        return -sum([p_ci_aj * log(p_ci_aj, 2.0) 
                    for p_ci_aj in p_ci_ajs 
                    if p_ci_aj != 0])
    
    def h_C_A(self, donnees, attribut, valeur):
        """ H(C|A) - l'entropie de la classe après avoir choisi de partitionner\
            les données suivant quelles soint inferieures
            ou superieures à la valeur de l'attribut A.
            
            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: les valeurs suivant laquelle on partitionne.
            :return: H(C|A)
        """
        # Calcule P(a_j) pour chaque valeur a_j de l'attribut A.
        p_inf_aj = self.p_inf_aj(donnees, attribut, valeur)
        p_sup_aj = self.p_sup_aj(donnees, attribut, valeur)

        # Calcule H_C_aj pour chaque valeur a_j de l'attribut A.
        h_c_inf_aj = self.h_C_inf_aj(donnees, attribut, valeur) 
        h_c_sup_aj = self.h_C_sup_aj(donnees, attribut, valeur) 
                   
        return p_inf_aj*h_c_inf_aj + p_sup_aj*h_c_sup_aj
    
    
donnees = []
import csv

with open('train_continuous.csv') as f:
    f_csv = csv.reader(f)
    en_tetes = next(f_csv)
    for ligne in f_csv:
        target = ligne.pop()
        donnees.append([target,{'age' : ligne[0], 'sex' : ligne[1], 'cp' : ligne[2], 'trestbps' : ligne[3],'chol' :ligne[4], 'fbs':ligne[5] , 'restecg':ligne[6], 'thalach':ligne[7], 'exang':ligne[8], 'oldpeak':ligne[9], 'slope' :ligne[10], 'ca':ligne[11], 'thal':ligne[12] }])




    
#ENTRAINER LE MODELE
        
id3 = ID3()
arbre = id3.construit_arbre(donnees)

#print('Arbre de décision :')
#rint(arbre)
#print()

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
    resultat = result[-1] 
    #print(resultat + " " + targets[k])
    if resultat == targets[k]:
        S += 1
        
pourcentage = S*100/len(donneestest)

print(pourcentage)
#print(len(donneestest))


    
    
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:14:19 2020

@author: mathi
"""


### Class NoeudDeDecision from 27/04 Eercises
    
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
            :param enfants: un dictionnaire associant un fils (sous-noeud) Ã \
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
        """ Classifie une donnÃ©e Ã  l'aide de l'arbre de dÃ©cision duquel le noeud\
            courant est la racine.

            :param donnee: la donnÃ©e Ã  classifier.
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
    
### Class ID3 from 27/04 Eercises
        
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
            # Sélectionne l'attribut qui réduit au maximum l'entropie.
            h_C_As_attribs = [(self.h_C_A(donnees, attribut, attributs[attribut]), 
                               attribut) for attribut in attributs]

            attribut = min(h_C_As_attribs, key=lambda h_a: h_a[0])[1]

            # Crée les sous-arbres de manière récursive.
            attributs_restants = attributs.copy()
            del attributs_restants[attribut]

            partitions = self.partitionne(donnees, attribut, attributs[attribut])
            
            enfants = {}
            for valeur, partition in partitions.items():
                enfants[valeur] = self.construit_arbre_recur(partition,
                                                             attributs_restants,
                                                             predominant_class)

            return NoeudDeDecision(attribut, donnees, str(predominant_class), enfants)

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
    

### Class RegleSansVariables from 24/02 Eercises

class RegleSansVariables:
    """ Représentation d'une règle d'inférence pour le chaînage sans\
        variables. 
    """

    def __init__(self, conditions, conclusion):
        """ Construit une règle étant donné une liste de conditions et une\
            conclusion.
            
            :param list conditions: une collection de propositions (sans\
            variables) nécessaires pour déclencher la règle.
            :param conclusion: la proposition (sans variables) résultant du\
            déclenchement de la règle.
        """

        self.conditions = set(conditions)
        self.conclusion = conclusion

    def depend_de(self, fait):
        """ Vérifie si un fait est pertinent pour déclencher la règle.
            
            :param fait: un fait qui doit faire partie des conditions de\
            déclenchement.
            :return: ``True`` si le fait passé en paramètre fait partie des\
            conditions de déclenchement.
        """
        return fait in self.conditions

    def satisfaite_par(self, faits):
        """ Vérifie si un ensemble de faits est suffisant pour prouver la\
            conclusion.
            
            :param list faits: une liste de faits.
            :return: ``True`` si les faits passés en paramètres suffisent à\
            déclencher la règle.
        """
        return self.conditions.issubset(faits)

    def __repr__(self):
        """ Représentation d'une règle sous forme de string. """
        return '{} => {}'.format(str(list(self.conditions)), 
                                 str(self.conclusion))




class ResultValues():

    def __init__(self):
    
#QUESTION 1
        
#IMPORTATION OF THE TRAINING DATA
        
        donnees = []
        import csv
        
        with open('train_bin.csv') as f:
            f_csv = csv.reader(f)
            next(f_csv)
            for ligne in f_csv:
                target = ligne.pop()
                donnees.append([target,{'age' : ligne[0], 'sex' : ligne[1], 'cp' : ligne[2], 'trestbps' : ligne[3],'chol' :ligne[4], 'fbs':ligne[5] , 'restecg':ligne[6], 'thalach':ligne[7], 'exang':ligne[8], 'oldpeak':ligne[9], 'slope' :ligne[10], 'ca':ligne[11], 'thal':ligne[12] }])
        
        
#TRAINING OF THE TREE WITH THE TRAINING DATASET
                
        id3 = ID3()
        arbre = id3.construit_arbre(donnees)
        #TODO add tree stats 
        
        
#QUESTION 2
        
#IMPORTATION OF THE TEST DATA
        
        donneestest = []
        targets = []
        import csv
        
        with open('test_public_bin.csv') as f:
            f_csv = csv.reader(f)
            next(f_csv)
            for ligne in f_csv:
                target = ligne.pop()
                targets.append(target)
                donneestest.append({'age' : ligne[0], 'sex' : ligne[1], 'cp' : ligne[2], 'trestbps' : ligne[3],'chol' :ligne[4], 'fbs':ligne[5] , 'restecg':ligne[6], 'thalach':ligne[7], 'exang':ligne[8], 'oldpeak':ligne[9], 'slope' :ligne[10], 'ca':ligne[11], 'thal':ligne[12] })
        
        
#CHECK THE RESULT OF THE PREDICTION AND COMPARE IT WITH THE TRUE RESULT IN ORDER
#TO GET THE PERCENTAGE OF WELL-CLASSIFIED DATA
                
        S = 0
        for k in range(len(donneestest)):
            result = arbre.classifie(donneestest[k])
            resultat = result[-1]
            if resultat == targets[k]:
                S += 1
        pourcentage = S*100/len(donneestest)
        
        print("The accuracy is" + " " + str(pourcentage))
       
        
#QUESTION 3
        
        #FIRST WE CREATE INITIAL FACTS FROM THE TRAINING DATASET
        
        faits_initiaux = []
        for donnee in donnees :
            faits_initiaux.append([str(attr) + " = " + 
                                          str(donnee[1][attr]) for attr in donnee[1]])
        
        
        #WE USE DFS ON THE TREE AND STORE THE PATHS TO THE TERMINAL NODES
        
        paths = []
        noeuds = [(arbre,[("start",'0')])]
        
        while noeuds :
            (n,p) = noeuds.pop(0)
            if (n.terminal()) :
                paths.append((p,n.classe()))
            else : 
                for key in n.enfants:
                    noeuds = [(n.enfants[key], p.copy()+ [(n.attribut,key)])] + noeuds
        
        
        #WE USE THOSE PATHS TO GENERATE RULES
        
        regles = []
        for k in range(len(paths)):
            L , end = paths[k]
            conditions = []
            for i in range(1,len(L)):
                att, val = L[i]
                conditions.append(str(att) + " = " + val)
            regles.append(RegleSansVariables(conditions, end))
        
        
        #WE IMPLEMENT A FUNCTION WHICH PROVIDES AN EXPLANATION GIVEN THE RULES AND A DATA POINT
        
        def find_explanation(entree, regles) :
            for regle in regles:
                if (regle.satisfaite_par([str(attr) + " = " + 
                                          (entree[attr]) for attr in entree])):
                    return regle
        
        
#QUESTION 4 
                    
        #WE EXTRACT THE DOMAIN OF VALUES FOR EACH ATTRIBUTE
        #WE USE THE SAME CODE AS IN ID3
                
        domain = {}
        for donnee in donnees:
            for attribut, valeur in donnee[1].items():
                valeurs = domain.get(attribut)
                if valeurs is None:
                    valeurs = set()
                    domain[attribut] = valeurs
                valeurs.add(valeur)
        
        
        #WE IMPLEMENT TO FUNCTIONS TO FIND THE ATTRIBUTES AND VALUES 
        #TO CHANGE IN ORDER TO CURE A DATA POINT
    
        def find_single_cure(row, new_row, domain, arbre) :
            for attr in new_row:
                tmp = row[attr]
                for i in domain[attr] :
                    row[attr] = str(i)
                    is_sick = arbre.classifie(row)[-1]
                    if (is_sick == "0") :
                        row[attr] = tmp
                        return (attr,i)
                row[attr] = tmp
                    
        def find_double_cure(row, new_row, domain, arbre) :
            for attr1 in new_row:
                tmp1 = row[attr1]
                for i in domain[attr1] :
                    row[attr1] = str(i)
                    for attr2 in new_row :
                        tmp2 = row[attr2]
                        for j in domain[attr2]:
                            row[attr2] = str(j)
                            is_sick = arbre.classifie(row)[-1]
                            if (is_sick == "0") :
                                row[attr1] = tmp1 
                                row[attr2] = tmp2
                                return (attr1, attr2, i, j)
                    row[attr2] = tmp2
                row[attr1] = tmp1          
        
        
        counter = 0
        total = 0
       
        
        #WE TRY TO CURE THE DATA POINTS IN OUR DATASET
        
        for donnee in donneestest :
            
            new_donnee = donnee.copy()
            del new_donnee["age"]
            del new_donnee["sex"]
            
            #WE GET THE LIST OF CONDITIONS RESPONSIBLE FOR THE CLASSIFICATION
            regle = find_explanation(donnee, regles)
            if(regle != None) :
                print("\nThis person was classified " + str(regle.conclusion) + 
                      " because he/she satisfies the conditions " + str(regle.conditions))
            
            #IF THE PATIENT IS SICK, WE TRY TO FIND A CURE BY CHANGING AT MOST 2 ATTRIBUTES
            result = arbre.classifie(donnee)
            if (result[-1] == "1"):
                total += 1
                cure = find_single_cure(donnee, new_donnee, domain, arbre)
                if (cure == None) :
                    cure = find_double_cure(donnee, new_donnee, domain, arbre)
                    if (cure == None) :
                        print("We did not manage to cure the patient by changing 2 attributes")
                        counter += 1
                    else :
                        print ("We manage to cure the patient by setting " + cure[0] + " to " 
                                   + str(cure[2]) + " and " + cure[1] + " to " + str(cure[3]))
                else : 
                    print ("We manage to cure the patient by setting " + cure[0] 
                               + " to " + str(cure[1]))
                
        print("\nBy changing one or two attributes, we managed to cure " +
              str(100 - counter*100/total) + "% of the originaly sick patients\n")



#WE STORE THE RESULT VALUES
        
        # Task 1
        self.arbre = arbre
        # Task 3
        self.faits_initiaux = faits_initiaux
        self.regles = regles
        # Task 5
        self.arbre_advance = None

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.regles, self.arbre_advance]
    

ResultValues().get_results()
#print(ResultValues().get_results())
    
    

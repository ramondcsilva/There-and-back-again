# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:39:42 2019

@author: Adlla Katarine, Daniel Alves, Ramon Silva with python 3.6
"""

import pandas as pd
import numpy as np 
# Leitura de arquivo para criação da base_herois de dados
base_herois = pd.read_csv('Datasets\herois.csv')
base_herois_superpower = pd.read_csv('Datasets\superpoderes.csv')

# Tratamento de valores negativos e agrupamento de classes do Atributo WEIGTH

'''
# Leve = 0, Medio = 1, Pesado = 2
A classe agora só contem dois resultados, 0 ou 1. A mudança de 3 para 2 classes não influenciou em nada no
resultado da acurácia da árvore de decisão e naive bayes.

#base_herois.loc[base_herois.Weight > 75, 'Weight'] = 2
#base_herois.loc[base_herois.Weight == 75, 'Weight'] = 1
'''
base_herois.loc[base_herois.Weight < 0, 'Weight'] = 0 
base_herois.loc[base_herois.Weight == 0, 'Weight'] = int(base_herois['Weight'].mean())
base_herois.loc[base_herois.Weight == 75, 'Weight'] = 1
base_herois.loc[base_herois.Weight > 75, 'Weight'] = 2


# Agrupamento de classes do Atributo Publisher
# Dividido entre Marvel Comics e Outhers
base_herois.loc[base_herois.Publisher != 'Marvel Comics', 'Publisher'] = 'Outhers'

# Tratamento de valores negativos e agrupamento de classes do Atributo HEIGTH
# Baixo = 0, Alto = 1
base_herois.loc[base_herois.Height < 0, 'Height'] = 0 
base_herois.loc[base_herois.Height == 0, 'Height'] = int(base_herois['Height'].mean())
base_herois.loc[base_herois.Height <= 170, 'Height'] = 0
base_herois.loc[base_herois.Height > 170, 'Height'] = 1

# Mescla base de dados 
result = base_herois.merge(base_herois_superpower, left_on ='name', right_on='hero_names', how='outer')

# Exclusao do atributo do nome e de herois que estavam duplicados
result.drop("hero_names",1,inplace=True)
result = result.drop(50)
result = result.drop(62)
result = result.drop(69)
result = result.drop(115)
result = result.drop(156)
result = result.drop(259)
result = result.drop(289)
result = result.drop(290)
result = result.drop(481)
result = result.drop(617)
result = result.drop(696)

#Personagens com nomes iguais(não necessariamente duplicados) tiveram seus nomes alterados
result.loc[23, 'name'] = "Angel II"
result.loc[48, 'name'] = "Atlas II"
result.loc[97, 'name'] = "Black Canary II"
result.loc[623, 'name'] = "Spider-Man II"
result.loc[624, 'name'] = "Spider-Man III"
result.loc[674, 'name'] = "Toxin II"

# Criação de atributo para previsao de dados, excluindo herois sem caracteristicas
previsores = result.iloc[0:734,2:178].values

# Tratando valores 'nan' da base de dados
from sklearn.impute import SimpleImputer
# Imputer recebe a classe que tratar dados nulos

# Preenche os valores nulos com traços
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value= '-')
imputer = imputer.fit(previsores[:,:])
# Atribui as modificação de valores nulos, a mesma variavel
previsores[:,:] = imputer.fit_transform(previsores[:,:])

# Preenche os traços com valores nulos para uso do algoritmo KNN
# Para predição de atributos fatlantes
imputer = SimpleImputer(missing_values='-', strategy='constant', fill_value= 'NaN')
imputer = imputer.fit(previsores[:,:])
# Atribui as modificação de valores nulos, a mesma variavel
previsores[:,:] = imputer.fit_transform(previsores[:,:])

# Transforma os dados categoricos/nominais em numericos 
from sklearn.preprocessing import LabelEncoder
previsores[:, 0] = LabelEncoder().fit_transform(previsores[:, 0].astype('str'))
previsores[:, 1] = LabelEncoder().fit_transform(previsores[:, 1].astype('str'))
previsores[:, 2] = LabelEncoder().fit_transform(previsores[:, 2].astype('str'))
previsores[:, 3] = LabelEncoder().fit_transform(previsores[:, 3].astype('str'))
previsores[:, 5] = LabelEncoder().fit_transform(previsores[:, 5].astype('str'))
previsores[:, 6] = LabelEncoder().fit_transform(previsores[:, 6].astype('str'))
previsores[:, 7] = LabelEncoder().fit_transform(previsores[:, 7].astype('str'))

# Pacote para uso de algoritmos para tratatar valores faltantes em um dataset
from fancyimpute import KNN    
# Usa 5NN que tenham um recurso para preencher os valores ausentes de cada linha
previsores = KNN(k = 5).fit_transform(previsores)


# Transforma Objeto em DATAFRAME para verificar pre-processamento
result = pd.DataFrame(previsores)

# Cria atributo a ser previsto
classe = result.iloc[:,26].values
# Exclui o mesmo da base de dados previsora
result = result.drop(columns = 26)
# Retorna a modificação
previsores = result.iloc[:,:].values

# Determina o tipo int para todas bases usadas
previsores = previsores.astype('int')
classe = LabelEncoder().fit_transform(classe)

'''
#################################################################################################
######################################## CLASSIFICADORES ########################################
#################################################################################################
'''


from sklearn.model_selection import train_test_split    #Função do pacote sklearn que divide automaticamente dados teste e dados de treinamento
from sklearn.model_selection import cross_val_score     #importação do algoritmo de validação cruzada
from sklearn.model_selection import cross_validate      #Retorna a taxa de previsao, tempo de execução e recall
from sklearn.metrics import confusion_matrix, f1_score  #Avalização por meio de Matriz de Confução
from sklearn import metrics
import matplotlib.pyplot as plt

# Criando variaveis para treinamento e teste, usando o metodo de divisao dos dados
# Usou-se 30%(test_size = 0.30) como quantidade de atributos para teste e o restante para treinamento
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.30, random_state=0)

'''
######################################## ÁRVORE DE DECISÃO ########################################
'''

# Hiperparamenters para achar a melhores paramentros para a arvore de decisao
paramenter = {"max_depth": [3,20],
              "min_samples_leaf": [1,5],
              'criterion': ('gini','entropy')}  

# Treinamento a partir de uma Arvore de Decisao 
from sklearn.tree import DecisionTreeClassifier

# Criação de uma Arvore de Decisao 
tree = DecisionTreeClassifier()

# Uso de Validação Cruzada em Grade, buscando uma melhor paramentrização para a arvore,
# Inibindo Overfitting
# Ela testa todos situações, requerendo um maior custo computacional
from sklearn.model_selection import GridSearchCV
classificadorTREE = GridSearchCV(tree, paramenter, cv=3)
# Execuçaão do treinamento 
classificadorTREE.fit(previsores_treinamento, classe_treinamento)

# Retorna o melhor paramentro e seu melhor score
print("Tuned: {}".format(classificadorTREE.best_params_))
print("Best score is {}".format(classificadorTREE.best_score_))

# Testamos os dados para achar sua taxa de acerto
previsoesTREE = classificadorTREE.predict(previsores_teste)

#resultado da avaliação cruzada feita com 3 testes. k=3
resultado_cvTREE = cross_val_score(classificadorTREE, previsores, classe, cv = 3)
#média dos resultados da avaliação cruzada
print("TREE Cross Validation Mean: {}".format(resultado_cvTREE.mean()))
#desvio padrão dos resultados da avaliação cruzada
print("TREE Cross-Validation Standard Deviation: {}".format(resultado_cvTREE.std()))

# Usando o Cross_validate para avaliar o classificadorTREE
scoring = ['precision_macro', 'recall_macro']
scores_cvTREE = cross_validate(classificadorTREE, 
                           previsores, 
                           classe,
                           scoring=scoring, 
                           cv=3)

# Metrica que usar valores de precisão e recall
f1TREE = f1_score(classe_teste, previsoesTREE, average='micro')

# Cria uma matriz para comparação de dados dos dois atributos
matrizTREE = confusion_matrix(classe_teste, previsoesTREE)


# Avaliação da precisão do modelo de predição por meio de curva ROC
# Ajusta dados para criar medidas de curva
cls_testeTREE = pd.DataFrame(classe_teste).astype('float')
predsTREE = classificadorTREE.predict_proba(previsores_teste)[::,1]
# Cria atributos Falso positivo e Verdadeiro positivo
fprTREE, tprTREE, = metrics.roc_curve(cls_testeTREE, predsTREE)
# Calcula area embaixo da curva roc
aucTREE = metrics.roc_auc_score(cls_testeTREE, predsTREE)

# Uso de biblioteca para Plotagem de Gráfico
plt.plot(fprTREE, tprTREE, '', label="SuperStrength, auc= %0.2f"% aucTREE)
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend(loc=4)
plt.show()


'''
######################################## NAIVE BAYES ########################################
'''


from sklearn.naive_bayes import BernoulliNB # importação do algoritmo e sua classe BernoulliNB
classificadorNB = BernoulliNB()
classificadorNB.fit(previsores_treinamento, classe_treinamento) #treina o algoritmo(cria a tabela de probabilidade)
previsoesNB = classificadorNB.predict(previsores_teste)	# Testamos os dados para achar sua taxa de acerto
#Retorna a precisão média nos dados e rótulos de teste fornecidos.
print("Best test score is {}".format(classificadorNB.score(previsores_teste,classe_teste)))
#Retorna a precisão média nos dados e rótulos de treinamento fornecidos.
print("Best training score is {}".format(classificadorNB.score(previsores_treinamento,classe_treinamento)))
 
# Metrica que usar valores de precisão e recall
f1NB = f1_score(classe_teste, previsoesNB, average='micro')

# Cria uma matriz para comparação de dados dos dois atributos 
matrizNB = confusion_matrix(classe_teste, previsoesNB)

#resultado da avaliação cruzada feita com 3 testes. k=3
resultado_cvNB = cross_val_score(classificadorNB, previsores, classe, cv = 3)
#média dos resultados da avaliação cruzada
print("Naive Bayes Cross Validation Mean: {}".format(resultado_cvNB.mean()))
#desvio padrão dos resultados da avaliação cruzada
print("Naive Bayes Cross-Validation Standard Deviation: {}".format(resultado_cvNB.std()))


# Usando o Cross_validate para avaliar o classificadorNB
scoring = ['precision_macro', 'recall_macro']
scores_cvNB = cross_validate(classificadorNB, 
                           previsores, 
                           classe,
                           scoring=scoring, 
                           cv=3)


# Avaliação da precisão do modelo de predição por meio de curva ROC
# Ajusta dados para criar medidas de curva
cls_testeNB = pd.DataFrame(classe_teste).astype('float')
predsNB = classificadorNB.predict_proba(previsores_teste)[::,1]
# Cria atributos Falso positivo e Verdadeiro positivo
fprNB, tprNB,_ = metrics.roc_curve(cls_testeNB, predsNB)
# Calcula area embaixo da curva roc
aucNB = metrics.roc_auc_score(cls_testeNB, predsNB)

# Uso de biblioteca para Plotagem de Gráfico
plt.plot(fprNB, tprNB, '', label="SuperStrength, auc= %0.2f"% aucNB)
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend(loc=4)
plt.show()


'''
######################################## RANDOM FOREST ########################################
'''

# RandomForestClassifier é a classe que gera a floresta
from sklearn.ensemble import RandomForestClassifier
# instancia a classe RandomForestClassifier
classificadorRF = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
classificadorRF.fit(previsores_treinamento, classe_treinamento) # constrói a floresta

# Testamos os dados para achar sua taxa de acerto
previsoesRF = classificadorRF.predict(previsores_teste)

#Retorna a precisão média nos dados e rótulos de teste fornecidos.
print("Best test score is {}".format(classificadorRF.score(previsores_teste,classe_teste)))
#Retorna a precisão média nos dados e rótulos de treinamento fornecidos.
print("Best training score is {}".format(classificadorRF.score(previsores_treinamento,classe_treinamento)))
 
# Metrica que usar valores de precisão e recall
f1RF = f1_score(classe_teste, previsoesRF, average='micro')

# Cria uma matriz para comparação de dados dos dois atributos 
matrizRF = confusion_matrix(classe_teste, previsoesRF)

#resultado da avaliação cruzada feita com 3 testes. k=3
resultado_cvRF = cross_val_score(classificadorRF, previsores, classe, cv = 3)
#média dos resultados da avaliação cruzada
print("Random Forest Cross Validation Mean: {}".format(resultado_cvRF.mean()))
#desvio padrão dos resultados da avaliação cruzada
print("Random Forest Cross-Validation Standard Deviation: {}".format(resultado_cvRF.std()))

# Usando o Cross_validate para avaliar o classificadorRF
scoring = ['precision_macro', 'recall_macro']
scores_cvRF = cross_validate(classificadorRF, 
                           previsores, 
                           classe,
                           scoring=scoring, 
                           cv=3)


# Avaliação da precisão do modelo de predição por meio de curva ROC
# Ajusta dados para criar medidas de curva
cls_testeRF = pd.DataFrame(classe_teste).astype('float')
predsRF = classificadorRF.predict_proba(previsores_teste)[::,1]
# Cria atributos Falso positivo e Verdadeiro positivo
fprRF, tprRF,_ = metrics.roc_curve(cls_testeRF, predsRF)
# Calcula area embaixo da curva roc
aucRF = metrics.roc_auc_score(cls_testeRF, predsRF)

# Uso de biblioteca para Plotagem de Gráfico
plt.plot(fprRF, tprRF, '', label="SuperStrength, auc= %0.2f"% aucRF)
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend(loc=4)
plt.show()

'''
####################################### VOTING_CLASSIFIER  ########################################
'''
from sklearn.ensemble import VotingClassifier

votingClf = VotingClassifier(estimators=[('tr', classificadorTREE), 
                                         ('rf', classificadorRF), 
                                         ('nb', classificadorNB)], 
                                           voting='soft', weights = [1,2,1])


for clf, label in zip([classificadorTREE, classificadorRF, classificadorNB, votingClf], ['Decision Tree', 'Random Forest', 'Naive Bayes', 'Ensemble']):

    scores = cross_val_score(clf, previsores, classe, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))

'''
#################################################################################################
############################################ ENSEMBLE ###########################################
#################################################################################################
'''

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

'''
################################### BOOSTTRAP AGGREGATING(BAGGING) ########################################
'''

classificadorBagging = BaggingClassifier(votingClf, max_samples=0.5, max_features=1.0, n_estimators=5)
classificadorBagging.fit(previsores_treinamento, classe_treinamento)
print("Bagging " + str(classificadorBagging.score(previsores_teste, classe_teste)))

'''
################################### ADAPTIVE BOOSTING(ADA-BOOST) ########################################
'''
#criando uma ensemble de AdaBoost com 20 árvores de decisão
classificadorAdaBoost = AdaBoostClassifier(votingClf, n_estimators = 5, learning_rate = 1)
classificadorAdaBoost.fit(previsores_treinamento, classe_treinamento)
print("Ada-Boost " + str(classificadorAdaBoost.score(previsores_teste, classe_teste)))


'''
xt = previsores[:10]

plt.figure()
plt.plot(classificadorTREE.predict(xt), 'gd', label='DecisionTree')
plt.plot(classificadorRF.predict(xt), 'b^', label='RandomForestRegressor')
plt.plot(classificadorNB.predict(xt), 'ys', label='NaiveBayers')
plt.plot(classificadorAdaBoost.predict(xt), 'r*', label='Voting')
plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Comparison of individual predictions with averaged')
plt.show()
'''
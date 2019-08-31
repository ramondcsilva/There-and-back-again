# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:39:42 2019

@author: Adlla Katarine, Daniel Alves, Ramon Silva
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
base_herois.loc[base_herois.Weight < 75, 'Weight'] = 0
base_herois.loc[base_herois.Weight >= 75, 'Weight'] = 1


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
# Imputer recebe a classe que trata dados nulos

# Preenchendo valores nulos com os mais frequentes
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(previsores[:,:]) 
# Atribui as modificação de valores nulos, a mesma variavel
previsores[:,:] = imputer.fit_transform(previsores[:,:])

# Atribui os valores vagos de GENDER como a classe NO-GENDER
imputer = SimpleImputer(missing_values = '-', strategy='constant', fill_value='no-gender')
imputer = imputer.fit(previsores[:,0].reshape(1,-1)) 
previsores[:,0] = imputer.fit_transform(previsores[:,0].reshape(1,-1))

# Atribui os valores vagos de RACE como a classe NO-RACE
imputer = SimpleImputer(missing_values = '-', strategy='constant', fill_value='no-race')
imputer = imputer.fit(previsores[:,2].reshape(1,-1)) 
previsores[:,2] = imputer.fit_transform(previsores[:,2].reshape(1,-1))

# Atribui os valores vagos de EYE COLOR como a classe NO-COLOR
# Atribui os valores vagos de HAIR COLOR como a classe NO-COLOR
# Atribui os valores vagos de SKIN COLOR como a classe NO-COLOR
imputer = SimpleImputer(missing_values = '-', strategy='constant', fill_value='no-color')
# EYE COLOR
imputer = imputer.fit(previsores[:,1].reshape(1,-1)) 
previsores[:,1] = imputer.fit_transform(previsores[:,1].reshape(1,-1))
# HAIR COLOR
imputer = imputer.fit(previsores[:,3].reshape(1,-1)) 
previsores[:,3] = imputer.fit_transform(previsores[:,3].reshape(1,-1))
# SKIN COLOR
imputer = imputer.fit(previsores[:,6].reshape(1,-1)) 
previsores[:,6] = imputer.fit_transform(previsores[:,6].reshape(1,-1))

# Atribui os valores BAD de ALIGNMENT como a classe NO-GOOD
imputer = SimpleImputer(missing_values = 'bad', strategy='constant', fill_value='no-good')
imputer = imputer.fit(previsores[:,7].reshape(1,-1)) 
previsores[:,7] = imputer.fit_transform(previsores[:,7].reshape(1,-1))

# Atribui os valores NEUTRAL de ALIGNMENT como a classe NO-GOOD
imputer = SimpleImputer(missing_values = 'neutral', strategy='constant', fill_value='no-good')
imputer = imputer.fit(previsores[:,7].reshape(1,-1)) 
previsores[:,7] = imputer.fit_transform(previsores[:,7].reshape(1,-1))

# Atribui os valores vagos de ALIGNMENT como a classe NO-GOOD
imputer = SimpleImputer(missing_values = '-', strategy='constant', fill_value='no-good')
imputer = imputer.fit(previsores[:,7].reshape(1,-1)) 
previsores[:,7] = imputer.fit_transform(previsores[:,7].reshape(1,-1))

# Atribui os valores vagos com o mais frequentes aos dados restantes
imputer = SimpleImputer(missing_values='-', strategy='most_frequent')
imputer = imputer.fit(previsores[:,:]) 
previsores[:,:] = imputer.fit_transform(previsores[:,:])

# Transforma Objeto em DATAFRAME para verificar pre-processamento
result = pd.DataFrame(previsores)
guarda = result

# Cria atributo a ser previsto
classe = result.iloc[:,10].values
# Exclui o mesmo da base de dados previsora
result = result.drop(columns = 10)
# Retorna a modificação
previsores = result.iloc[:,:].values

# Transforma os dados categoricos/nominais em numericos 
from sklearn.preprocessing import LabelEncoder
previsores[:, 0] = LabelEncoder().fit_transform(previsores[:, 0])
previsores[:, 1] = LabelEncoder().fit_transform(previsores[:, 1])
previsores[:, 2] = LabelEncoder().fit_transform(previsores[:, 2])
previsores[:, 3] = LabelEncoder().fit_transform(previsores[:, 3])
previsores[:, 5] = LabelEncoder().fit_transform(previsores[:, 5])
previsores[:, 6] = LabelEncoder().fit_transform(previsores[:, 6])
previsores[:, 7] = LabelEncoder().fit_transform(previsores[:, 7])

# Determina o tipo int para todas bases usadas
previsores = previsores.astype('int')
classe = classe.astype('int')

'''
#################################################################################################
######################################## CLASSIFICADORES ########################################
#################################################################################################
'''

'''
######################################## ÁRVORE DE DECISÃO ########################################
'''
# Função do pacote sklearn que divide automaticamente dados teste e dados de treinamento
from sklearn.model_selection import train_test_split
# Criando variaveis para treinamento e teste, usando o metodo de divisao dos dados
# Usou-se 30%(test_size = 0.30) como quantidade de atributos para teste e o restante para treinamento
previsores_treinamentoTREE, previsores_testeTREE, classe_treinamentoTREE, classe_testeTREE = train_test_split(previsores, classe, test_size=0.30, random_state=0)

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
classificadorTREE.fit(previsores_treinamentoTREE, classe_treinamentoTREE)

# Retorna o melhor paramentro e seu melhor score
print("Tuned: {}".format(classificadorTREE.best_params_))
print("Best score is {}".format(classificadorTREE.best_score_))

# Testamos os dados para achar sua taxa de acerto
previsoesTREE = classificadorTREE.predict(previsores_testeTREE)

# Usando o Cross_validate para avaliar o classificadorTREE
# Retornando sua taxa de previsao, tempo de execução e recall
from sklearn.model_selection import cross_validate
scoring = ['precision_macro', 'recall_macro']
scores_cvTREE = cross_validate(classificadorTREE, 
                           previsores, 
                           classe,
                           scoring=scoring, 
                           cv=3)

# Avalização por meio de Matriz de Confução e Pontução de Acerto
from sklearn.metrics import accuracy_score, confusion_matrix
# Compara dados de dois atributos retornando o percentual de acerto
accuracyTREE = accuracy_score(classe_testeTREE, previsoesTREE) 
# Cria uma matriz para comparação de dados dos dois atributos
matrizTREE = confusion_matrix(classe_testeTREE, previsoesTREE)

'''
# Avaliação da precisão do modelo de predição por meio de curva ROC
from sklearn import metrics
import matplotlib.pyplot as plt
# Ajusta dados para criar medidas de curva
cls_testeTREE = pd.DataFrame(classe_testeTREE).astype('float')
predsTREE = classificadorTREE.predict_proba(previsores_testeTREE)[::,1]
# Cria atributos Falso positivo e Verdadeiro positivo
fprTREE, tprTREE,_ = metrics.roc_curve(cls_testeTREE, predsTREE)
# Calcula area embaixo da curva roc
aucTREE = metrics.roc_auc_score(cls_testeTREE, predsTREE)

# Uso de biblioteca para Plotagem de Gráfico
plt.plot(fprTREE, tprTREE, '', label="Accelerated Healing, auc= %0.2f"% aucTREE)
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend(loc=4)
plt.show()

######################################## NAIVE BAYES ########################################
'''
# Função do pacote sklearn que divide automaticamente dados teste e dados de treinamento
from sklearn.model_selection import train_test_split
# Criando variaveis para treinamento e teste, usando o metodo de divisao dos dados
# Usou-se 20%(test_size = 0.20) como quantidade de atributos para teste e o restante para treinamento
previsores_treinamentoNB, previsores_testeNB, classe_treinamentoNB, classe_testeNB = train_test_split(previsores, classe, test_size=0.20, random_state=0)

from sklearn.naive_bayes import BernoulliNB # importação do algoritmo e sua classe BernoulliNB
classificadorNB = BernoulliNB()
classificadorNB.fit(previsores_treinamentoNB, classe_treinamentoNB) #treina o algoritmo(cria a tabela de probabilidade)
previsoesNB = classificadorNB.predict(previsores_testeNB)	# Testamos os dados para achar sua taxa de acerto

from sklearn.metrics import accuracy_score, confusion_matrix

accuracyNB = accuracy_score(classe_testeNB, previsoesNB) 
matrizNB = confusion_matrix(classe_testeNB, previsoesNB)

'''
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

classificadorGaussian = GaussianNB() # instancia da classe GaussianNB
classificadorGaussian.fit(previsores_treinamento, classe_treinamento) #treina o algoritmo(cria a tabela de probabilidade)
previsoesGaussian = classificadorGaussian.predict(previsores_teste)
accuracyGaussian = accuracy_score(classe_teste, previsoesGaussian) 
matrizGaussian = confusion_matrix(classe_teste, previsoesGaussian)

classificadorMultinomial = MultinomialNB()
classificadorMultinomial.fit(previsores_treinamento, classe_treinamento) #treina o algoritmo(cria a tabela de probabilidade)
previsoesMultinomial = classificadorMultinomial.predict(previsores_teste)	
accuracyMultinomial = accuracy_score(classe_teste, previsoesMultinomial) 
matrizMultinomial = confusion_matrix(classe_teste, previsoesMultinomial)
'''


'''
######################################## RANDOM FOREST ########################################
'''
# Função do pacote sklearn que divide automaticamente dados teste e dados de treinamento
from sklearn.model_selection import train_test_split
# Criando variaveis para treinamento e teste, usando o metodo de divisao dos dados
# Usou-se 20%(test_size = 0.20) como quantidade de atributos para teste e o restante para treinamento
previsores_treinamentoRF, previsores_testeRF, classe_treinamentoRF, classe_testeRF = train_test_split(previsores, classe, test_size=0.20, random_state=0)

# RandomForestClassifier é a classe que gera a floresta
from sklearn.ensemble import RandomForestClassifier
# instancia a classe RandomForestClassifier
classificadorRF = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificadorRF.fit(previsores_treinamentoRF, classe_treinamentoRF) # constrói a floresta

# Testamos os dados para achar sua taxa de acerto
previsoesRF = classificadorRF.predict(previsores_testeRF)

'''
#################################################################################################
############################################ ENSEMBLE ###########################################
#################################################################################################
'''

# Função do pacote sklearn que divide automaticamente dados teste e dados de treinamento
from sklearn.model_selection import train_test_split
# Criando variaveis para treinamento e teste, usando o metodo de divisao dos dados
# Usou-se 20%(test_size = 0.20) como quantidade de atributos para teste e o restante para treinamento
previsores_treinamentoBagging, previsores_testeBagging, classe_treinamentoBagging, classe_testeBagging = train_test_split(previsores, classe, test_size=0.20, random_state=0)

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
'''
######################################## BOOSTTRAP AGGREGATING(BAGGING) ########################################
'''

bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0,n_estimators=20)
bg.fit(previsores_treinamentoBagging, classe_treinamentoBagging)
print("bagging " + str(bg.score(previsores_teste, classe_teste)))

'''
######################################## BOOSTING ########################################
'''

bt = GradientBoostingClassifier(n_estimators=85).fit(previsores_treinamento, classe_treinamento)
print("boosting " + str(bt.score(previsores_teste, classe_teste)))
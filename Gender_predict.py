# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:40:31 2019

@author: Adlla Katarine, Daniel Alves, Ramon Silva
"""

import pandas as pd
import numpy as np 
# Leitura de arquivo para criação da base_herois de dados
base_herois = pd.read_csv('herois.csv')
base_herois_superpower = pd.read_csv('superpoderes.csv')

# Tratamento de valores negativos e agrupamento de classes do Atributo WEIGTH
# Leve = 0, Medio = 1, Pesado = 2
base_herois.loc[base_herois.Weight < 0, 'Weight'] = 0 
base_herois.loc[base_herois.Weight == 0, 'Weight'] = int(base_herois['Weight'].mean())
base_herois.loc[base_herois.Weight < 75, 'Weight'] = 0
base_herois.loc[base_herois.Weight > 75, 'Weight'] = 2
base_herois.loc[base_herois.Weight == 75, 'Weight'] = 1

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
result = result.drop(columns=10)
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
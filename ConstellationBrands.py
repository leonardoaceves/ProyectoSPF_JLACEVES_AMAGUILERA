'''
Codigo del proyecto de SPF
Autores: Leonardo Aceves y María Aguilera.

Variables a considerar: 
 1. Sold-To Party Country
 2. Brand Name
 3. Delivery Date - Order Date (Delivery)
 4. Varietal/Blend Description
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as st
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
#%%
consbrand = pd.read_excel("C:\Sales report.xlsx")
dtypes = consbrand.dtypes
resume = consbrand.describe(exclude=[object]) 

consbrand['Sold-To Party State'].replace(['AR', 'WY', 'UT', 'CT', 'ID', 'GA', 'MT', 'IL', 'CA', 'NY', 'MS',
       'MA', 'FL', 'NJ', 'NC', 'VA', 'HI', 'KS', 'NV', 'TN', 'LA', 'WA',
       'IN', 'ME', 'TX', 'MD', 'AZ', 'RI', 'OH', 'PA', 'SC', 'MN', 'MI',
       'CO', 'NE', 'IA', 'OR', 'NH', 'KY', 'SD', 'MO', 'DE', 'NM', 'AL',
       'VT', 'OK', 'WI', 'ND', 'WV'], np.arange(1,50), inplace=True)


'''Para esta primera entrega nos vamos a concentrar en hacer estimaciones de unicamente
dos variables que son "Sold-To Party Country y "Product Hierarchy Description", que corresponden a 
las ventas totales por pais y por marca.'''

#%%
#FUNCIONES AUTOGENERADAS

#Kernel
def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    kde = KDEMultivariate(x, 
                          bw='cv_ml',
                          var_type='u')
    return(kde.pdf(x_grid))

#Funcion de generacion de la distribucion discreta.
def distri_dis(p_acum, indices, N):
    #p_acum, es la lista de probabildad acumulada.
    #indices, son los valores reales.
    #N, numero de simulaciones
    U =np.random.rand(N)
    val2id = {i:val_i for i, val_i in enumerate(indices)}
    V = pd.Series([sum([1 for i in p_acum if i < ui]) for ui in U]).map(val2id)
    return(V)


def redu_var(p_acum, indices, U):
    #U, cantidad de numeros aleatorias a generar.
    rand2reales = {i: idx for i, idx in enumerate(indices)}
    y = pd.Series([sum([1 for p in p_acum if p < ui]) for ui in U]).map(rand2reales)
    return(y)

def estra_i_espa(B):
    U_estra = (np.random.rand(B) + np.arange(0, B))/B
    return(U_estra)    


#%%

#METODO DE LA TRANFORMADA INVERSA
print('Número de estados: ', consbrand['Sold-To Party State'].nunique())
print(consbrand['Sold-To Party State'].head())
state = consbrand.groupby("Sold-To Party State").count()
state_acum = pd.DataFrame(index = state.index)
freq = state['Sales Order Number'].values
state_acum['Probability'] = freq/len(consbrand)
state_acum['Cumulative'] = np.cumsum(freq/len(consbrand))

plt.figure(figsize=(18,10))
plt.bar(state.index, freq, width = .92)
plt.title('Distribucion de probabilidad de ordenes por estado, USA')
plt.ylabel('frequencia')
plt.xlabel('Estados')
plt.show()

#Cumulative
plt.figure(figsize=(18,10))
plt.bar(state.index, state_acum['Cumulative'], width = .92)
plt.title('Distribucion de probabilidad acumulada, USA')
plt.ylabel('frequencia')
plt.xlabel('Estados')
plt.show()

distri_diss = distri_dis(state_acum['Cumulative'].values, state.index, 200000) #ES IGUAL A M1
plt.figure(figsize=(18,10))
plt.bar(state.index, distri_diss.value_counts(sort = False))
plt.title('Simulacion de funcion de distribución, USA')
plt.ylabel('frequencia')
plt.xlabel('Estados')
plt.show()

#%%
#KERNEL
'''
kde1 = kde_statsmodels_m_func(consbrand['Sold-To Party State'])
x1 = np.arange(-4.5,2,0.01)
plt.plot(x1,kde1(x1))'''

#TECNICA DE REDUCCION DE VARIANZA
print("TECNICA DE REDUCCION DE VARIANZA")
print("Media distribución discreta: ", 
      round((consbrand['Sold-To Party State']*state_acum['Probability']).sum(),2))

#MEDIA POR METODO MONTECARLO
montecacr = distri_diss.mean()
print('Media Montecarlo crudo', round(montecacr,2))
#MUESTREO ESTRATIFICADO
e = [np.random.uniform(0, 0.3, int(0.3 * 1000)),
     np.random.uniform(0.3, 0.7, int(0.4 * 1000)),
     np.random.uniform(0.7, 1, int(0.3 * 1000))]
w = [(len(consbrand)*.30/len(consbrand))/.3,
     (len(consbrand)*.30/len(consbrand))/.4 ,
     (len(consbrand)*.30/len(consbrand))/.3] 
m2= list(map(lambda ui,wi: redu_var(state_acum['Cumulative'], 
                                    consbrand['Sold-To Party State'], 
                                    ui)/wi, e, w))
print('Media muestreo estratificado(1):', round(np.concatenate(m2).mean(),2))

m3 = redu_var(state_acum['Cumulative'], 
              consbrand['Sold-To Party State'], 
              estra_i_espa(10000))

print('Media muestreo estratificado(2): ', round(np.mean(m3), 2))

#NUMEROS COMPLEMENTARIOS
m4 = redu_var(state_acum['Cumulative'], 
              consbrand['Sold-To Party State'],  
              np.concatenate([np.random.rand(10000), 1 - np.random.rand(10000)]))
print('Media numeros complementario: ', round(np.mean(m4), 2))

#%%
#PRUEBA DE BONDAD Y AJUSTE
#(Datos reales)
print("PRUEBA DE BONDAD Y AJUSTE")
print("Media (MonteCarlo): ", round(consbrand['Sold-To Party State'].mean(), 2))
print("Desviacón estandar: ", round(consbrand['Sold-To Party State'].std(), 2))
print("Varianza: ", round(st.sem(consbrand['Sold-To Party State']),2), "vs", 
      round(np.std(consbrand['Sold-To Party State'])/np.sqrt(len(consbrand['Sold-To Party State'])),2))
#Podemos calcular la varianza ya que estas son variables independientes.
#Intervalo de confianza 
np.random.seed(5555)
X = np.random.normal(consbrand['Sold-To Party State'].mean(), 
                     consbrand['Sold-To Party State'].std(),
                     100000)
confianza = 0.05

#Intervalo de confianza usando t-student
intervalo1 = st.t.interval(.90, 
                       len(X)-1, 
                       loc=np.mean(X), 
                       scale=st.sem(X))
intervalo2 = st.t.interval(.95, 
                       len(X)-1, 
                       loc=np.mean(X), 
                       scale=st.sem(X))
intervalo3 = st.t.interval(.99, 
                       len(X)-1, 
                       loc=np.mean(X), 
                       scale=st.sem(X))

print("Intervalo al 90% de confianza: ", intervalo1)
print("Intervalo al 95% de confianza: ", intervalo2)
print("Intervalo al 99% de confianza: ", intervalo3)

#HIPOTESIS NULA
H0 = st.ttest_1samp(X, montecacr)
print('Prueba de hipotesis: ', H0)

#ESTADISTICO NORMALIZADO
t = (np.mean(X) - montecacr)/(np.std(X)/np.sqrt(100000))
print('Estadístico de prueba teórico: ', round(t, 5))
cuantil = st.t(100000 - 1).cdf(1 - confianza/2)
print(f'Región de rechazo: (t < {-round(cuantil,4)}) U (t > {round(cuantil,4)})')

'''
El valor de PValue es un valor significativamente mayor al indice de confianza
previamente definido, por lo tanto NO rechazamos la media calculada en la simulación,
sin embargo, el estadistico esta fuera de la región de aceptación.
'''

#KERNEL DENSITY ESTIMATION
kernel = kde_statsmodels_m(consbrand['Sold-To Party State'], state.index, bandwidth=0.2)
plt.bar(state.index ,state_acum['Probability'])
plt.plot(state.index, kernel)
plt.show()

#CHI CUADRADA


#%%
botellas = consbrand.groupby("Order Quantity").count()

#METODO DE LA TRANFORMADA INVERSA
print(consbrand['Order Quantity'].head())
botellas_acum = pd.DataFrame(index = botellas.index)
freqb = botellas['Sales Order Number'].values
botellas_acum['Probability'] = freqb/len(consbrand)
botellas_acum['Cumulative'] = np.cumsum(freqb/len(consbrand))

plt.figure(figsize=(18,10))
plt.bar(botellas.index, np.log(freqb), width = .92)
plt.title('Distribucion de probabilidad de ordenes por estado, USA')
plt.ylabel('frequencia')
plt.xlabel('Botellas')
plt.show()

#Cumulative
plt.figure(figsize=(18,10))
plt.bar(botellas.index, botellas_acum['Cumulative'], width = .92)
plt.title('Distribucion de probabilidad acumulada, USA')
plt.ylabel('frequencia')
plt.xlabel('Botellas')
plt.show()

distri_dissb = distri_dis(botellas_acum['Cumulative'].values, botellas.index, 200000) #ES IGUAL A M1
plt.figure(figsize=(18,10))
plt.bar(botellas.index[0:len(distri_dissb.value_counts(sort = False))], np.log(distri_dissb.value_counts(sort = False)))
plt.title('Simulacion de funcion de distribución, USA')
plt.ylabel('frequencia')
plt.xlabel('Botellas')
plt.show()

#%%
#KERNEL

#TECNICA DE REDUCCION DE VARIANZA
print("TECNICA DE REDUCCION DE VARIANZA")
print("Media distribución discreta: ", 
      round((consbrand['Order Quantity']*botellas_acum['Probability']).sum(),2))

#MEDIA POR METODO MONTECARLO
montecacrb = distri_dissb.mean()
print('Media Montecarlo crudo', round(montecacrb,2))
#MUESTREO ESTRATIFICADO
e = [np.random.uniform(0, 0.3, int(0.3 * 1000)),
     np.random.uniform(0.3, 0.7, int(0.4 * 1000)),
     np.random.uniform(0.7, 1, int(0.3 * 1000))]
w = [(len(consbrand)*.30/len(consbrand))/.3,
     (len(consbrand)*.30/len(consbrand))/.4 ,
     (len(consbrand)*.30/len(consbrand))/.3] 
m2= list(map(lambda ui,wi: redu_var(botellas_acum['Cumulative'], 
                                    consbrand['Order Quantity'], 
                                    ui)/wi, e, w))
print('Media muestreo estratificado(1):', round(np.concatenate(m2).mean(),2))

m3 = redu_var(botellas_acum['Cumulative'], 
              consbrand['Order Quantity'], 
              estra_i_espa(10000))

print('Media muestreo estratificado(2): ', round(np.mean(m3), 2))

#NUMEROS COMPLEMENTARIOS
m4 = redu_var(botellas_acum['Cumulative'], 
              consbrand['Order Quantity'],  
              np.concatenate([np.random.rand(10000), 1 - np.random.rand(10000)]))
print('Media numeros complementario: ', round(np.mean(m4), 2))

#%%
#PRUEBA DE BONDAD Y AJUSTE
#(Datos reales)
print("PRUEBA DE BONDAD Y AJUSTE")
print("Media (MonteCarlo): ", round(consbrand['Order Quantity'].mean(), 2))
print("Desviacón estandar: ", round(consbrand['Order Quantity'].std(), 2))
print("Varianza: ", round(st.sem(consbrand['Order Quantity']),6), "vs", 
      round(np.std(consbrand['Order Quantity'])/np.sqrt(len(consbrand['Order Quantity'])),6))
#Podemos calcular la varianza ya que estas son variables independientes.
#Intervalo de confianza 
np.random.seed(5555)
X = np.random.normal(consbrand['Order Quantity'].mean(), 
                     consbrand['Order Quantity'].std(),
                     100000)
confianza = 0.05

#Intervalo de confianza usando t-student
intervalo1 = st.t.interval(.90, 
                       len(X)-1, 
                       loc=np.mean(X), 
                       scale=st.sem(X))
intervalo2 = st.t.interval(.95, 
                       len(X)-1, 
                       loc=np.mean(X), 
                       scale=st.sem(X))
intervalo3 = st.t.interval(.99, 
                       len(X)-1, 
                       loc=np.mean(X), 
                       scale=st.sem(X))

print("Intervalo al 90% de confianza: ", intervalo1)
print("Intervalo al 95% de confianza: ", intervalo2)
print("Intervalo al 99% de confianza: ", intervalo3)

#HIPOTESIS NULA
H0 = st.ttest_1samp(X, montecacrb)
print('Prueba de hipotesis: ', H0)

#ESTADISTICO NORMALIZADO
t = (np.mean(X) - montecacrb)/(np.std(X)/np.sqrt(100000))
print('Estadístico de prueba teórico: ', round(t, 5))
cuantil = st.t(100000 - 1).cdf(1 - confianza/2)
print(f'Región de rechazo: (t < {-round(cuantil,4)}) U (t > {round(cuantil,4)})')

#%%






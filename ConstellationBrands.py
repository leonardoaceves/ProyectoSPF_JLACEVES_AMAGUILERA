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
#%%
consbrand = pd.read_excel("C:\Sales report.xlsx")
dtypes = consbrand.dtypes
resume = consbrand.describe(exclude=[object]) 

'''Para esta primera entrega nos vamos a concentrar en hacer estimaciones de unicamente
dos variables que son "Sold-To Party Country y "Product Hierarchy Description", que corresponden a 
las ventas totales por pais y por marca.'''


#Referencias:
    #https://relopezbriega.github.io/blog/2016/06/29/distribuciones-de-probabilidad-con-python/

#%%
#FUNCIONES AUTOGENERADAS

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

distri_diss = distri_dis(state_acum['Cumulative'].values, state.index, 73000) #ES IGUAL A M1
plt.figure(figsize=(18,10))
plt.bar(state.index, distri_diss.value_counts(sort = False))
plt.title('Simulacion de funcion de distribución, USA')
plt.ylabel('frequencia')
plt.xlabel('Estados')
plt.show()

#%%
#TECNICA DE REDUCCION DE VARIANZA
print("TECNICA DE REDUCCION DE VARIANZA")
print("Media distribución discreta: ", 
      round((freq*state_acum['Probability']).sum(),2))

#MEDIA POR METODO MONTECARLO
montecacr = distri_diss.value_counts(sort = False).mean()
print('Media Montecarlo crudo', round(montecacr,2))
#MUESTREO ESTRATIFICADO
e = [np.random.uniform(0, 0.3, int(0.3 * 1000)),
     np.random.uniform(0.3, 0.7, int(0.4 * 1000)),
     np.random.uniform(0.7, 1, int(0.3 * 1000))]
w = [(len(consbrand)*.30/len(consbrand))/.3,
     (len(consbrand)*.30/len(consbrand))/.4 ,
     (len(consbrand)*.30/len(consbrand))/.3] 
m2= list(map(lambda ui,wi: redu_var(state_acum['Cumulative'], 
                                    freq, 
                                    ui)/wi, e, w))
print('Media muestreo estratificado(1):', round(np.concatenate(m2).mean(),2))

m3 = redu_var(state_acum['Cumulative'], 
              freq, 
              estra_i_espa(10000))

print('Media muestreo estratificado(2): ', round(np.mean(m3), 2))

#NUMEROS COMPLEMENTARIOS
m4 = redu_var(state_acum['Cumulative'], 
              freq,  
              np.concatenate([np.random.rand(10000), 1 - np.random.rand(10000)]))
print('Media numeros complementario: ', round(np.mean(m4), 2))

#%%
#PRUEBA DE BONDAD Y AJUSTE
#(Datos reales)
print("PRUEBA DE BONDAD Y AJUSTE")
print("Media (MonteCarlo): ", round(freq.mean(), 2))
print("Desviacón estandar: ", round(freq.std(), 2))
print("Varianza: ", round(st.sem(freq),2), "vs", round(np.std(freq)/np.sqrt(len(freq)),2))
#Podemos calcular la varianza ya que estas son variables independientes.

#Intervalo de confianza usando t-student
intervalo1 = st.t.interval(.90, 
                       len(freq)-1, 
                       loc=np.mean(freq), 
                       scale=st.sem(freq))
intervalo2 = st.t.interval(.95, 
                       len(freq)-1, 
                       loc=np.mean(freq), 
                       scale=st.sem(freq))
intervalo3 = st.t.interval(.99, 
                       len(freq)-1, 
                       loc=np.mean(freq), 
                       scale=st.sem(freq))

#REDODEAR TUPLAS
print("Intervalo al 90% de confianza: ", intervalo1)
print("Intervalo al 95% de confianza: ", intervalo2)
print("Intervalo al 99% de confianza: ", intervalo3)

#HIPOTESIS NULA
f = 15
ho = st.ttest_1samp(X, media2)
print('La prueba de hipótesis arroja como resultado\n',ho)

# Calculamos el estadístico normalizado
t = (np.mean(X) - media2)/(np.std(X) / np.sqrt(N))
print('Cálculo del estadístico de prueba teórico=', t)

# Cálculo de la región de rechazo
confianza_ph = 0.05
cuantil = st.t(N-1).cdf(1-confianza_ph/2)
print(f'Región de rechazo = (t<{-cuantil}) U (t>{cuantil})')

# Gráfica t-student
# %matplotlib inline
dat = np.arange(-4,4,.1)
# for i in range(1):
y = st.t.pdf(dat,df=N-1)
y1=st.norm.pdf(dat)
plt.plot(dat,y,label='t-student %d df' %(N-1))
plt.plot(dat,y1,label='Normal')
plt.legend()
plt.show()


#%%
#Mara ('Varietal/Blend Description')
print("Número de marcas: ", consbrand['Varietal/Blend Description'].nunique())
brand = consbrand.groupby('Varietal/Blend Description').count()
plt.figure(figsize=(18,10))
plt.bar(brand.index, brand['Sales Order Number'].values/len(consbrand), width = .92)
plt.title('Distribucion de ventas por tipo de bebida, USA')
plt.ylabel('frequencia')
plt.xlabel('Estados')
plt.show()










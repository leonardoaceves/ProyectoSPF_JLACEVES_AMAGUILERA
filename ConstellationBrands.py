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

#%%
#Carga de base de datos Y ajustamos el tipo de dato de las columnas para el punto tres.
consbrand = pd.read_excel("C:\Order Report.xlsx")
consbrand['Requested Delivery Date'] = consbrand['Requested Delivery Date'].apply(pd.to_datetime)
consbrand['Delivery Date'] = consbrand['Delivery Date'].apply(pd.to_datetime)

#El archivo contiene ordenes con varios estatus. Filtramos solo ordenes que ya esan entregadas.
#consbrand = consbrand[consbrand['Incoterms Location 1'] == 'Delivered Duty Paid'] 

resume = consbrand.describe(exclude=[object])

library(arules)
library(arulesViz)
library(tidyverse)
library(plyr)
library(ggplot2)
library(knitr)
library(lubridate)
library(RColorBrewer)


# lectura de los datos

 datos<- read.csv('ejOnlineRetail.csv',sep=';',dec=',',stringsAsFactors = FALSE)
 
 
# se eliminan filas con datos faltantes
 
 datosCompletos<- na.omit(datos)
 
 # se convierten a factor las variables Description y Country
 
 datosCompletos %>% mutate(Description=as.factor(Description))
 
 datosCompletos %>% mutate(Country=as.factor(Country))
 
 # se convierte la fecha a tipo Date
 datosCompletos$Date<- as.Date(datosCompletos$InvoiceDate,format = '%d/%m/%y')
 
 # se convierte a numerico el # de factura
 datosCompletos$InvoiceNo<- as.numeric(as.character(datosCompletos$InvoiceNo))
 
 head(datosCompletos)
 

 # Transformar a formato transacción
 # se agrupan las transacciones con igual InvoiceNo y Date
 # luego se concatena con una ,
 
 transacciones<- ddply(datosCompletos,c('InvoiceNo','Date'),
                       function(df1) paste(df1$Description,collapse = ','))
 
 # se descartan la fecha y el # de facturas para las reglas de asociación
 
 transacciones$InvoiceNo<-NULL
 transacciones$InvoiceDate<- NULL
 transacciones$Date<-NULL
 
 # se renombra la columna restante como items
 
 colnames(transacciones) <- c('items')
 
 # se guardan los datos como archivo csv en formato transaccion
 write.csv(transacciones,'transaccionesOR2.csv',quote = FALSE)
 
 
 # se leerá el archivo como transacciones formato basket
 
 transacc<- read.transactions('transaccionesOR2.csv',sep=',',format = 'basket')
 
 # exploracion de las transacciones
 
  summary(transacc)
  
 
 # Generar las reglas
  reglas<- apriori(transacc,parameter = list(supp=0.001,conf=0.8,maxlen=10))
  
  inspect(reglas[1:10])
  
 # Generar menos reglas
  reglas<- apriori(transacc,parameter = list(supp=0.001,conf=0.8,maxlen=3))
  
  inspect(reglas[1:10])
 
  # eliminar reglas que son subconjuntos de otras 
  subconjuntos<- which(colSums(is.subset(reglas,reglas))>1)
  
  reglasFinal<- reglas[-subconjuntos]
  inspect(reglasFinal[1:10])

  # se observan las 10 mejores reglas según soporte
  inspect(sort(reglasFinal,by='support',decreasing = TRUE)[1:10])
  
  # se filtran las reglas con confianza superior a 0.25
  mejoresReglas<- reglasFinal[quality(reglasFinal)$confidence>0.25]
  
  # graficar las 5 mejores reglas
  
  cincoMejores<- head(mejoresReglas,n=5,by='confidence')
  
  plot(cincoMejores,method = 'graph',engine = 'htmlwidget')
  
  # graficar 20 reglas individuales para observar lhs y rhs
  
  Top20<-head(mejoresReglas,n=20,by='confidence')
  
  plot(Top20,method ='paracoord')
  
  # reglas relacionadas con ciertos items:
  # Ejemplo: ¿qué compraron los clientes antes de comprar greeting card ?
  
  reglasAntGreeting<- apriori(transacc,parameter =list(supp=0.001,conf=0.8), 
                           appearance = list(default='lhs',rhs='GREETING CARD')  )
  
  inspect(head(reglasAntGreeting))
  
  # Ejemplo: si un cliente compró un producto específico, cuáles otros llevó?
  # qué compraron los clientes que compraron algún producto específico ej:Decoupage ?
  reglasGreeting<- apriori(transacc,parameter =list(supp=0.001,conf=0.8), 
                           appearance = list(lhs='DECOUPAGE',default='rhs')  )
  
  inspect(head(reglasGreeting))
  
  # exportar las reglas
  write(mejoresReglas,file='mejoresReglas.csv',sep=',',quote=F,row.names=F)
    
     
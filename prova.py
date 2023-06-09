import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi
from scipy.stats import pearsonr 

#FUNZIONE CHE TI APRE UN FILE, TI CONVERTE I DATI DA ESADECIMALI A DECIMALI E, INFINE, TI RESTITUISCE UN VETTORE arr
def apri(file):
    with open(file) as f:
        arr=[int(item,16) for item in f.readlines()]
    return np.array(arr)

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

#FIT EQUATION
def gaus(X,C,mean,sigma):
    return C*np.exp(-(X-mean)**2/(2*sigma**2)) 


def gaus3(X,C,mean,sigma,C2,mean2,sigma2):
    return gaus(X,C,mean,sigma)+gaus(X,C2,mean2,sigma2)

def clean_data(file,inf,sup,sorgente):
    e=apri(file)
    minimo=min(e)
    massimo=max(e) 
    

    if (sorgente == "americio"):
        binstot = 2000
    else:
        binstot = 500

    e=e[e>inf]
    e=e[e<sup]

    #(Differenza tra massimo e minimo del vettore complpeto) / (il numero dei bin totali) = (Differenza tra massimo e minimo del vettore selezionato) / (il numero dei bin da determinare, ossia binsfit)
    R = (massimo - minimo) / binstot 
    binsfit = round((sup - inf) / R)

    a=np.histogram(e, bins=binsfit)
    tops=a[0] #y data
    d_tops=np.sqrt(a[0]) #errore sulle y, ossia errore di una possoniana
    bin_edges=a[1]
    bin_centers=list() #x data
    for i in range(len(tops)):
        bin_center=(bin_edges[i]+bin_edges[i+1])/2
        bin_centers.append(bin_center)
    
    
    d_bin=[]
    for i in range(len(tops)):
        d=bin_edges[i+1]-bin_edges[i]
        d_bin.append(d)




    mean=np.mean(e)
    varianza=np.var(e)
    sigma=np.sqrt(varianza)
    ampiezza = np.max(e)

    
    y=[] #ARRAY WITHOUTH ZEROES ELEMENTS
    E=[] #ARRAY WITHOUTH ZEROES ELEMENTS
    dE=[]
    dy=[]
    for i in range(0,len(tops)):
        if(tops[i]!=0):
            y.append(tops[i])
            E.append(bin_centers[i])
            dE.append((1/np.sqrt(12))*d_bin[i])
            dy.append(d_tops[i])
        
    y=np.array(y)
    E=np.array(E)

    dy=np.array(dy)
    dE=np.array(dE)
    initParams=np.array([ampiezza,mean,sigma])

    return y,E,dy,dE,initParams,binsfit









#FUNZIONE CHE FA LA CALIBRAZIONE PER UNA SORGENTE; RESTITUISCE PLOT, CANALE E LARGHEZZA DEL PICCO
def best_fit(file, inf, sup, sorgente):
    """return optimal parameters for the fit of a gaussian

    Args:
        file (path): path of the file
        inf (float): lowest extremum of the fit funciton
        sup (float): the highest extremum of the fit funtion
        sorgente (str): name of the source

    Returns:
       popt, pcov (np.array): the optimal parameters of the fit
    """
    e=apri(file)
    minimo=min(e)
    massimo=max(e)
   
    y,E,dy,dE,initParams,_ = clean_data(file, inf, sup, sorgente)


    #FIT
    ampiezza,mean,sigma=initParams
    popt,pcov = scipy.optimize.curve_fit(gaus,E,y,initParams,dy, absolute_sigma=False)


    # ITERATIVELY UPDATE THE ERRORS AND REFIT.

    for i in range(10):  
        dTT=np.sqrt(dy**2+( ((E-mean)/(sigma**2))*ampiezza*np.exp(-((E-mean)**2)/2*(sigma**2))*dE   )**2) 
        popt, pcov=scipy.optimize.curve_fit(gaus, E, y,initParams , dTT, absolute_sigma=False)

    return np.array(popt), np.array(pcov), minimo, massimo



def risoluzione_quad(pippo, baudo):

    
    energie = np.array([0.060, 0.511, 0.662, 1.173, 1.275, 1.333])
    energie1 = np.array([0.060, 0.511, 0.662, 1.173, 1.333])

    #'''
    pippo1 = pippo[np.where(energie != 1.275)]
    baudo1 = baudo[np.where(energie != 1.275)]
    #'''

    weights=1/baudo1


    z, cov = np.polyfit(energie1, pippo1, deg = 2, w = weights, full = False, cov = 'unscaled')

    global ar
    ar = z[0]
    global br
    br = z[1]
    global cr 
    cr= z[2]

    global err_ar
    err_ar = np.sqrt(cov[0][0])
    global err_br
    err_br = np.sqrt(cov[1][1])
    global err_cr
    err_cr = np.sqrt(cov[2][2])



    print('_______________________________ \n')
    print("Il parametro a vale: ", ar, " +- ", err_ar)
    print("Il parametro b vale: ", br, " +- ", err_br)
    print("Il parametro c vale: ", cr, " +- ", err_cr)
    print('_______________________________')

    chisq = ((((pippo1 - (ar*energie1*energie1 + br*energie1 + cr))**2 / (ar*energie1*energie1 + br*energie1 + cr)))).sum()
    print(f'Chisquare = {chisq:.1f}')
    chisq_norm=chisq/(len(pippo1)-4)
    print(f'Chisquare norm = {chisq_norm:.1f}')



    xp = np.linspace(0, 1.5, 1000)
    yp = xp*xp*ar + br*xp + cr


    plt.title('Risoluzione del rivelatore NaI in funzione dell\'energia')
    plt.xlabel('Energia (MeV)')
    plt.ylabel('Sigma dei picchi (canali)')
    plt.minorticks_on()
    plt.xticks(ticks=np.arange(0, 1.5001, step = 0.25))
    plt.xlim(left=0)
    plt.xlim(right=1.5)
    plt.ylim(bottom=0)
    plt.ylim(top=200)
    plt.tick_params(axis='both', which='minor')
    plt.grid(visible=True, which='major')
    plt.rc('axes', labelsize=10)


    plt.plot(xp, yp)
    plt.errorbar(energie, pippo, yerr=baudo, marker = '.', linestyle = '', c='r')

    plt.savefig('ris-quad.png', dpi=300, bbox_inches='tight')


    plt.show()

def dedA(franco):
    return (-B**2 - 2*A*(franco-C) + B*np.sqrt(B**2 + 4*A*(franco-C)))/(2*A*A*np.sqrt(B**2 + 4*A*(franco-C)))

def dedB(franco):
    return (-1 + B/(np.sqrt(B**2 + 4*A*(franco-C))))/(2*A)

def dedC(franco):
    return (-1)/(np.sqrt(B**2 + 4*A*(franco-C)))

def dedx(franco):
    return 1/(np.sqrt(B**2 + 4*A*(franco-C)))
    
def m(picco, energia, err_energia, angolo):
    me = picco*(1 - np.cos(angolo))/(picco/energia - 1)
    err_me = np.sqrt((err_energia*(picco**2)*(np.cos(angolo) - 1)/((picco-energia)**2))**2 + (np.radians(0.5)*picco*(np.sin(angolo))/(picco/energia - 1))**2)
    return me, err_me

def dati_1cal(file, inf, med, sup, angolo):
    e=apri(file)

    ex=np.array(e)

    binstot = 300

    fitinf = inf
    fitmed = med
    fitsup = sup

    ex=ex[ex>fitinf]
    ex=ex[ex<fitmed]

    ex2=np.array(e)

    ex2=ex2[ex2>fitmed]
    ex2=ex2[ex2<fitsup]

    ex3=np.array(e)

    ex3=ex3[ex3>fitinf]
    ex3=ex3[ex3<fitsup]



    plt.hist(e,bins=binstot, histtype='step', label=f'Dati: {len(e)} eventi')
    plt.legend()



        


    #PRIMO PICCO
    a=np.histogram(ex, bins=round(binstot*(max(ex)-min(ex))/(max(e)-min(e))))
    tops=a[0] #y data
    d_tops=np.sqrt(a[0])
    bin_edges=a[1]
    bin_centers=list() #x data
    for i in range(len(tops)):
        bin_center=(bin_edges[i]+bin_edges[i+1])/2
        bin_centers.append(bin_center)
        
        
    d_bin=[]
    for i in range(len(tops)):
        d=bin_edges[i+1]-bin_edges[i]
        d_bin.append(d)

    #FIT EQUATION
    def gaus(X,C,mean,sigma):
        return C*np.exp(-(X-mean)**2/(2*sigma**2))

    mean=np.mean(ex)
    varianza=np.var(ex)
    sigma=np.sqrt(varianza)
    ampiezza=np.max(ex)

        
    y=[] #ARRAY WITHOUTH ZEROES ELEMENTS
    E=[] #ARRAY WITHOUTH ZEROES ELEMENTS
    dE=[]
    dy=[]
    for i in range(0,len(tops)):
        if(tops[i]!=0):
            y.append(tops[i])
            E.append(bin_centers[i])
            dE.append((1/np.sqrt(12))*d_bin[i])
            dy.append(d_tops[i])
            
    y=np.array(y)
    E=np.array(E)
    dy=np.array(dy)
    dE=np.array(dE)
    

    #FIT
    initParams=np.array([ampiezza,mean,sigma])
    fitting_params,cov_matrix = scipy.optimize.curve_fit(gaus,E,y,initParams,dy, absolute_sigma=False)


    # ITERATIVELY UPDATE THE ERRORS AND REFIT.

    for i in range(10):  
        dTT=np.sqrt(dy**2+( ((E-mean)/(sigma**2))*ampiezza*np.exp(-((E-mean)**2)/2*(sigma**2))*dE   )**2)
        fitting_params, cov_matrix=scipy.optimize.curve_fit(gaus, E, y,initParams , dTT, absolute_sigma=False)



    p_sigma = np.sqrt(np.diag(cov_matrix))
    err=np.array(p_sigma)

    print('_______________________________')
    z=np.linspace(min(ex), max(ex), 1000)


    y_output = gaus(z, C=fitting_params[0],mean=fitting_params[1],sigma=fitting_params[2])

    plt.plot(z,y_output,color='r', ls= '--', label='Picco 1')

    plt.axvline(x=fitting_params[1], color='r', linestyle='--', label=f'Canale picco 1: {round(fitting_params[1])} $\pm$ ' f'{round(err[1])}')

    plt.legend()
   




    #SECONDO PICCO
    a2=np.histogram(ex2, bins=round(binstot*(max(ex2)-min(ex2))/(max(e)-min(e))))
    tops2=a2[0] #y data
    d_tops2=np.sqrt(a2[0])
    bin_edges2=a2[1]
    bin_centers2=list() #x data
    for i in range(len(tops2)):
        bin_center2=(bin_edges2[i]+bin_edges2[i+1])/2
        bin_centers2.append(bin_center2)
        
        
    d_bin2=[]
    for i in range(len(tops2)):
        d2=bin_edges2[i+1]-bin_edges2[i]
        d_bin2.append(d2)

    #FIT EQUATION
    def gaus2(X,C2,mean2,sigma2):
        return C2*np.exp(-(X-mean2)**2/(2*sigma2**2))

    mean2=np.mean(ex2)
    varianza2=np.var(ex2)
    sigma2=np.sqrt(varianza2)
    ampiezza2=np.max(ex2)


        
    y2=[] #ARRAY WITHOUTH ZEROES ELEMENTS
    E2=[] #ARRAY WITHOUTH ZEROES ELEMENTS
    dE2=[]
    dy2=[]
    for i in range(0,len(tops2)):
        if(tops2[i]!=0):
            y2.append(tops2[i])
            E2.append(bin_centers2[i])
            dE2.append((1/np.sqrt(12))*d_bin2[i])
            dy2.append(d_tops2[i])
            
    y2=np.array(y2)
    E2=np.array(E2)
    dy2=np.array(dy2)
    dE2=np.array(dE2)


    #FIT
    initParams2=np.array([ampiezza2,mean2,sigma2])
    fitting_params2,cov_matrix2 = scipy.optimize.curve_fit(gaus2,E2,y2,initParams2,dy2, absolute_sigma=False)


    # ITERATIVELY UPDATE THE ERRORS AND REFIT.

    for i in range(10):  
        dTT2=np.sqrt(dy2**2+( ((E2-mean2)/(sigma2**2))*ampiezza2*np.exp(-((E2-mean2)**2)/2*(sigma2**2))*dE2   )**2) 
        fitting_params2, cov_matrix2=scipy.optimize.curve_fit(gaus2, E2, y2,initParams2 , dTT2, absolute_sigma=False)



    p_sigma2 = np.sqrt(np.diag(cov_matrix2))
    err2=np.array(p_sigma2)

    print('_______________________________')
    z2=np.linspace(min(ex2), max(ex2), 1000)


    y_output2 = gaus2(z2, C2=fitting_params2[0],mean2=fitting_params2[1],sigma2=fitting_params2[2])

    plt.plot(z2,y_output2,color='g', ls= '--', label='Picco 2')

    plt.axvline(x=fitting_params2[1], color='g', linestyle='--', label=f'Canale picco 2: {round(fitting_params2[1])} $\pm$ ' f'{round(err2[1])}')


    plt.legend()









    #DOPPIA GAUSSIANA
    a3=np.histogram(ex3, bins=round(binstot*(max(ex3)-min(ex3))/(max(e)-min(e))))
    tops3=a3[0] #y data
    d_tops3=np.sqrt(a3[0])
    bin_edges3=a3[1]
    bin_centers3=list() #x data
    for i in range(len(tops3)):
        bin_center3=(bin_edges3[i]+bin_edges3[i+1])/2
        bin_centers3.append(bin_center3)
        
        
    d_bin3=[]
    for i in range(len(tops3)):
        d3=bin_edges3[i+1]-bin_edges3[i]
        d_bin3.append(d3)

    #FIT EQUATION
    def gaus3(X,C,mean,sigma,C2,mean2,sigma2):
        return gaus(X,C,mean,sigma)+gaus2(X,C2,mean2,sigma2)



        
    y3=[] #ARRAY WITHOUTH ZEROES ELEMENTS
    E3=[] #ARRAY WITHOUTH ZEROES ELEMENTS
    dE3=[]
    dy3=[]
    for i in range(0,len(tops3)):
        if(tops3[i]!=0):
            y3.append(tops3[i])
            E3.append(bin_centers3[i])
            dE3.append((1/np.sqrt(12))*d_bin3[i])
            dy3.append(d_tops3[i])
            
    y3=np.array(y3)
    E3=np.array(E3)
    dy3=np.array(dy3)
    dE3=np.array(dE3)


    #FIT
    initParams3=np.array([ampiezza,mean,sigma,ampiezza2,mean2,sigma2])
    fitting_params3,cov_matrix3 = scipy.optimize.curve_fit(gaus3,E3,y3,initParams3,dy3, absolute_sigma=False)

    #'''
    # ITERATIVELY UPDATE THE ERRORS AND REFIT.

    for i in range(50):  
        dTT3=np.sqrt(dy3**2+( ((E3-mean2)/(sigma2**2))*ampiezza2*np.exp(-((E3-mean2)**2)/(2*(sigma2**2)))*dE3   )**2 +( ((E3-mean)/(sigma**2))*ampiezza*np.exp(-((E3-mean)**2)/(2*(sigma**2)))*dE3   )**2) 
        fitting_params3, cov_matrix3=scipy.optimize.curve_fit(gaus3, E3, y3, initParams3, dTT3, absolute_sigma=False)
    #'''


    p_sigma3 = np.sqrt(np.diag(cov_matrix3))
    err3=np.array(p_sigma3)

    print('_______________________________')
    z3=np.linspace(min(ex3), max(ex3), 1000)


    y_output3 = gaus3(z3, C=fitting_params3[0],mean=fitting_params3[1],sigma=fitting_params3[2],C2=fitting_params3[3],mean2=fitting_params3[4],sigma2=fitting_params3[5])



        
    chisq = ((((y3 - gaus3(E3, *fitting_params3)))**2 / gaus3(E3, *fitting_params3))).sum()
    print(f'Chisquare = {chisq:.1f}')
    chisq_norm=chisq/(len(y3)-7)
    print(f'Chisquare norm = {chisq_norm:.1f}')





    plt.plot(z3, y_output3, color='k', label='Doppia gaussiana')
    plt.legend()

    plt.title('Dati')
    plt.xlabel('Canali')
    plt.ylabel('Eventi')
    plt.minorticks_on()
    plt.xticks(ticks=np.arange(0, 8300, step = 1000))
    plt.xlim(left=0)
    plt.xlim(right=8300)
    plt.tick_params(axis='both', which='minor')
    plt.grid(visible=True, which='major')
    plt.rc('axes', labelsize=10)
    
    plt.show()


    E1_sing = fitting_params[1]
    err_E1_sing = err[1]

    E2_sing = fitting_params2[1]
    err_E2_sing = err2[1]

    E1_multi = fitting_params3[1]
    err_E1_multi = err3[1]

    E2_multi = fitting_params3[4]
    err_E2_multi = err3[4]


    E_prime1 = (1.173)/(1 + (1.173/0.511)*(1 - np.cos(angolo)))
    E_prime2 = (1.333)/(1 + (1.333/0.511)*(1 - np.cos(angolo)))


    conversione = conversione_lin(media, err_media)[0]
    err_conversione = conversione_lin(media, err_media)[1]
    offset = conversione_lin(media, err_media)[2]
    err_offset = conversione_lin(media, err_media)[3]
    correlazione_conv_off = conversione_lin(media, err_media)[4]



    E1_sing_lin = (E1_sing - offset)/conversione
    err_E1_sing_lin = np.sqrt((err_offset/conversione)**2 + (err_conversione*(offset-E1_sing)/(conversione**2))**2 + 2*(E1_sing - offset)/(conversione*conversione*conversione)*correlazione_conv_off + (err_E1_sing/conversione)**2)

    E2_sing_lin = (E2_sing - offset)/conversione
    err_E2_sing_lin = np.sqrt((err_offset/conversione)**2 + (err_conversione*(offset-E2_sing)/(conversione**2))**2 + 2*(E2_sing - offset)/(conversione*conversione*conversione)*correlazione_conv_off + (err_E2_sing/conversione)**2)



    E1_sing_quad = (-B + np.sqrt(B**2 - 4*A*(C - E1_sing)))/(2*A)
    err_E1_sing_quad = np.sqrt(  (err_E1_sing*dedx(E1_sing))**2  + (err_A*dedA(E1_sing))**2 + (err_B*dedB(E1_sing))**2 + (err_C*dedC(E1_sing))**2 + 2*dedA(E1_sing)*dedB(E1_sing)*correlazione_A_B + 2*dedA(E1_sing)*dedC(E1_sing)*correlazione_A_C + 2*dedB(E1_sing)*dedC(E1_sing)*correlazione_B_C)

    E2_sing_quad = (-B + np.sqrt(B**2 - 4*A*(C - E2_sing)))/(2*A)
    err_E2_sing_quad = np.sqrt(  (err_E2_sing*dedx(E2_sing))**2 + (err_A*dedA(E2_sing))**2 + (err_B*dedB(E2_sing))**2 + (err_C*dedC(E2_sing))**2 + 2*dedA(E2_sing)*dedB(E2_sing)*correlazione_A_B + 2*dedA(E2_sing)*dedC(E2_sing)*correlazione_A_C + 2*dedB(E2_sing)*dedC(E2_sing)*correlazione_B_C)



    E1_multi_lin = (E1_multi - offset)/conversione
    err_E1_multi_lin = np.sqrt((err_offset/conversione)**2 + (err_conversione*(offset-E1_multi)/(conversione**2))**2 + 2*(E1_multi - offset)/(conversione*conversione*conversione)*correlazione_conv_off + (err_E1_multi/conversione)**2)

    E2_multi_lin = (E2_multi - offset)/conversione
    err_E2_multi_lin = np.sqrt((err_offset/conversione)**2 + (err_conversione*(offset-E2_multi)/(conversione**2))**2 + 2*(E2_multi - offset)/(conversione*conversione*conversione)*correlazione_conv_off + (err_E2_multi/conversione)**2)




    E1_multi_quad = (-B + np.sqrt(B**2 - 4*A*(C - E1_multi)))/(2*A)
    err_E1_multi_quad = np.sqrt( (err_E1_multi*dedx(E1_multi))**2 +  (err_A*dedA(E1_multi))**2 + (err_B*dedB(E1_multi))**2 + (err_C*dedC(E1_multi))**2 + 2*dedA(E1_multi)*dedB(E1_multi)*correlazione_A_B + 2*dedA(E1_multi)*dedC(E1_multi)*correlazione_A_C + 2*dedB(E1_multi)*dedC(E1_multi)*correlazione_B_C)

    E2_multi_quad = (-B + np.sqrt(B**2 - 4*A*(C - E2_multi)))/(2*A)
    err_E2_multi_quad = np.sqrt(  (err_E2_multi*dedx(E2_multi))**2 + (err_A*dedA(E2_multi))**2 + (err_B*dedB(E2_multi))**2 + (err_C*dedC(E2_multi))**2 + 2*dedA(E2_multi)*dedB(E2_multi)*correlazione_A_B + 2*dedA(E2_multi)*dedC(E2_multi)*correlazione_A_C + 2*dedB(E2_multi)*dedC(E2_multi)*correlazione_B_C)







    print("Il valore teorico in energia del primo picco è: ", E_prime1, "MeV")
    print()
    print("Il valore in energia del primo picco ricavato attraverso la calibrazione lineare e il fit con singola gaussiana è: ", E1_sing_lin, " +- ", err_E1_sing_lin,"MeV")
    print("Il valore in energia del primo picco ricavato attraverso la calibrazione lineare e il fit con doppia gaussiana è: ", E1_multi_lin, " +- ", err_E1_multi_lin,"MeV")
    print()
    print("Il valore in energia del primo picco ricavato attraverso la calibrazione quadratica e il fit con singola gaussiana è: ", E1_sing_quad, " +- ", err_E1_sing_quad,"MeV")
    print("Il valore in energia del primo picco ricavato attraverso la calibrazione quadratica e il fit con doppia gaussiana è: ", E1_multi_quad, " +- ", err_E1_multi_quad,"MeV")
    print()
    print("Il valore teorico in energia del secondo picco è: ", E_prime2, "MeV")
    print()
    print("Il valore in energia del secondo picco ricavato attraverso la calibrazione lineare e il fit con singola gaussiana è: ", E2_sing_lin, " +- ",  err_E2_sing_lin,"MeV")
    print("Il valore in energia del secondo picco ricavato attraverso la calibrazione lineare e il fit con doppia gaussiana è: ", E2_multi_lin, " +- ", err_E2_multi_lin,"MeV")
    print()
    print("Il valore in energia del secondo picco ricavato attraverso la calibrazione quadratica e il fit con singola gaussiana è: ", E2_sing_quad, " +- ", err_E2_sing_quad,"MeV")
    print("Il valore in energia del secondo picco ricavato attraverso la calibrazione quadratica e il fit con doppia gaussiana è: ", E2_multi_quad, " +- ", err_E2_multi_quad,"MeV")
    print()
    print()
    print("Il valore teorico in energia della massa dell'elettrone è: 0.511 MeV")
    print()
    print("Il valore di m in energia dal primo picco ricavato attraverso la calibrazione lineare e il fit con singola gaussiana è: ", m(1.173, E1_sing_lin, err_E1_sing_lin, theta)[0], " +- ", m(1.173, E1_sing_lin, err_E1_sing_lin, theta)[1],"MeV")
    print("Il valore di m in energia dal primo picco ricavato attraverso la calibrazione lineare e il fit con doppia gaussiana è: ", m(1.173, E1_multi_lin, err_E1_multi_lin, theta)[0], " +- ", m(1.173, E1_multi_lin, err_E1_multi_lin, theta)[1],"MeV")
    print()
    print("Il valore di m in energia dal primo picco ricavato attraverso la calibrazione quadratica e il fit con singola gaussiana è: ", m(1.173, E1_sing_quad, err_E1_sing_quad, theta)[0], " +- ", m(1.173, E1_sing_quad, err_E1_sing_quad, theta)[1],"MeV")
    print("Il valore di m in energia dal primo picco ricavato attraverso la calibrazione quadratica e il fit con doppia gaussiana è: ", m(1.173, E1_multi_quad, err_E1_multi_quad, theta)[0], " +- ", m(1.173, E1_multi_quad, err_E1_multi_quad, theta)[1],"MeV")
    print()
    print("Il valore di m in energia dal secondo picco ricavato attraverso la calibrazione lineare e il fit con singola gaussiana è: ", m(1.333, E2_sing_lin, err_E2_sing_lin, theta)[0], " +- ", m(1.333, E2_sing_lin, err_E2_sing_lin, theta)[1],"MeV")
    print("Il valore di m in energia dal secondo picco ricavato attraverso la calibrazione lineare e il fit con doppia gaussiana è: ", m(1.333, E2_multi_lin, err_E2_multi_lin, theta)[0], " +- ", m(1.333, E2_multi_lin, err_E2_multi_lin, theta)[1],"MeV")
    print()
    print("Il valore di m in energia dal secondo picco ricavato attraverso la calibrazione quadratica e il fit con singola gaussiana è: ", m(1.333, E2_sing_quad, err_E2_sing_quad, theta)[0], " +- ", m(1.333, E2_sing_quad, err_E2_sing_quad, theta)[1],"MeV")
    print("Il valore di m in energia dal secondo picco ricavato attraverso la calibrazione quadratica e il fit con doppia gaussiana è: ", m(1.333, E2_multi_quad, err_E2_multi_quad, theta)[0], " +- ", m(1.333, E2_multi_quad, err_E2_multi_quad, theta)[1],"MeV")

    return









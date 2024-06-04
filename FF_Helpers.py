from Basic_Model import *
from Newton import *

def Compute_gamma_nu(theta,omega):
    fe = -33*np.sin(theta[0]+theta[1])
    fs = -33*np.sin(theta[0]+theta[1]) - 30*np.sin(theta[0])
    ge = 33*np.cos(theta[0]+theta[1])
    gs = 33*np.cos(theta[0]+theta[1]) + 30*np.cos(theta[0])
    fse = -33*np.cos(theta[0]+theta[1])
    gse = -33*np.sin(theta[0]+theta[1])
    fee = -33*np.cos(theta[0]+theta[1])
    fss = -33*np.cos(theta[0]+theta[1]) - 30*np.cos(theta[0])
    gee = fe
    gss = fs
    gamma = gs*omega[0] + ge*omega[1] - fss*omega[0]*omega[0] - 2*fse*omega[0]*omega[1] - fee * omega[1] * omega[1]
    nu = - gss*omega[0]*omega[0] - 2*gse*omega[0]*omega[1] - gee * omega[1] * omega[1]
    return gamma,nu,fs,fe,gs,ge,fss,fse,fee,gss,gse,gee

def Compute_F(theta,omega):
    gamma,nu,fs,fe,gs,ge,fss,fse,fee,gss,gse,gee = Compute_gamma_nu(theta,omega)
    F1 = (fe*nu-ge*gamma)/(fe*gs+fs)
    F2 = (gs*gamma-fs*nu)/(gs*fe+ge)
    return np.array([F1,F2])


def pre_Compute(theta,omega):
    fe = -33*np.sin(theta[0]+theta[1])
    fs = -33*np.sin(theta[0]+theta[1]) - 30*np.sin(theta[0])
    ge = 33*np.cos(theta[0]+theta[1])
    gs = 33*np.cos(theta[0]+theta[1]) + 30*np.cos(theta[0])
    fse = -33*np.cos(theta[0]+theta[1])
    gse = -33*np.sin(theta[0]+theta[1])
    fee = -33*np.cos(theta[0]+theta[1])
    fss = -33*np.cos(theta[0]+theta[1]) - 30*np.cos(theta[0])
    gee = fe
    gss = fs
    return fs,fe,gs,ge,fss,fse,fee,gss,gse,gee

def Compute_f_new_version(theta,omega,acc,factor):
    fs,fe,gs,ge,fss,fse,fee,gss,gse,gee = pre_Compute(theta,omega)
    xddot = 13*(gs*omega[0] + ge*omega[1])*factor + fss*omega[0]*omega[0] + 2*fse*omega[0]*omega[1] + fee * omega[1] * omega[1] + fs*acc[0] + fe*acc[1]
    yddot = gss*omega[0]*omega[0] + 2*gse*omega[0]*omega[1] + gee * omega[1] * omega[1] + gs*acc[0] + ge*acc[1]
    gamma = xddot - fss*omega[0]*omega[0] - 2*fse*omega[0]*omega[1] - fee * omega[1] * omega[1]
    nu = yddot - gss*omega[0]*omega[0] - 2*gse*omega[0]*omega[1] - gee * omega[1] * omega[1]
    F1 = (fe*nu-ge*gamma)/(fe*gs-ge*fs) - acc[0]
    F2 = (gs*gamma-fs*nu)/(gs*fe-ge*fs) - acc[1]
    return np.array([F1,F2])


def Feedback_Linearization_with_FF(w1,w2,w3,w4,r1,r2,targets = [0,55],starting_point = [0,30],plot = True,Noise_Variance = 1e-6,DisplayNonlinear = True,proportionnality = 0,pert = 0,alpha = 1):

    Num_iter = 600
    dt = 0.001
    LOAD_ACTIVATION = True

    st1,st2 = newton(f,df,1e-8,1000,starting_point[0],starting_point[1])
    obj1,obj2 = newton(f,df,1e-8,1000,0,targets[1])
    obj3,obj4 = newton(f,df,1e-8,1000,targets[0],targets[1])


    xstart = np.array([st1,0,0,st2,0,0,obj1,0,obj2,0])
    x0 = np.array([st1,0,0,st2,0,0,obj1,obj2])

    Bruit = True
    NbreVar = 8
    
    #Define Weight Matrices
    R = np.array([[r1,0],[0,r2]])
    Q = np.array([[w1,0,0,0,0,0,-w1,0],[0,w3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
               [0,0,0,w2,0,0,0,-w2],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
               [0-w1,0,0,0,0,0,w1,0],[0,0,0,-w2,0,0,0,w2]])
    #Q = np.array([[2*w1/5+w1,0,0,-2*w1/5,0,0,w1/5-w1,0],[0,w3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
    #            [-2*w1/5,0,0,2*w2/5+w2,0,0,-w2/5,-w2],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
     #           [w1/5-w1,0,0,-w2/5,0,0,w1+w1/5,0],[0,0,0,-w2,0,0,0,w2]])
    Qnonlin = Q
    
    
    #Define Dynamic Matrices  
    A = np.array([[1,dt,0,0,0,0,0,0],[0,1,dt,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,dt,0,0,0],[0,0,0,0,1,dt,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
    B = np.array([[0,0],[0,0],[dt,0],[0,0],[0,0],[0,dt],[0,0],[0,0]])
    
    
    #Reverse Iterations
    #print(Qnonlin.astype(int))
    S = Qnonlin

    
    array_L = np.zeros((Num_iter-1,2,NbreVar))   
    array_S = np.zeros((Num_iter,NbreVar,NbreVar)) 
    array_S[-1] = Qnonlin
    for k in range(Num_iter-1):
        L = np.linalg.inv(R+B.T@S@B)@B.T@S@A
        array_L[Num_iter-2-k] = L
        S = A.T@S@(A-B@L)
        array_S[Num_iter-2-k] = S
        
    #print(array_L[0])
    #Feedback
    H,L = np.identity(8),array_L
        
    array_x = np.zeros((Num_iter,NbreVar))
    array_xhat = np.zeros((Num_iter,NbreVar))
    array_reelx = np.zeros((Num_iter,NbreVar-2)) 
    y = np.zeros((Num_iter,NbreVar))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()
    xhat = np.copy(x0)
    x = np.copy(x0)
    x_changevar = np.copy(x0)
    x_internalmodel = np.copy(x0)

    reelx = np.zeros(NbreVar-2)
    reelx[0],reelx[1] = x[0],x[3]

    new_reelx = np.copy(reelx)


    Command_Array = np.zeros((Num_iter-1,2,2))
    sigma = np.identity(NbreVar)*10**-6 #Espérance de (erreur erreur^) avec erreur = x - xhat
    v = np.zeros(2)
    OldF = 0
    for k in range(Num_iter-1):

        acc = np.array([(reelx[2]-array_reelx[k-1][2])/dt,(reelx[3]-array_reelx[k-1][3])/dt])
        if np.sin(reelx[0]+reelx[1])*33+np.sin(reelx[0])*30 > pert:

            if LOAD_ACTIVATION :
                if plot :
                    X = np.cos(reelx[0]+reelx[1])*33+np.cos(reelx[0])*30
                    Y = np.sin(reelx[0]+reelx[1])*33+np.sin(reelx[0])*30
                    plt.scatter(X,Y,color = "red")
                x[6:8] = [obj3,obj4]
                xhat[6:8] = [obj3,obj4]
                x_changevar[6:8] = [obj3,obj4]
                x_internalmodel[6:8] = [obj3,obj4]
                LOAD_ACTIVATION = False
                
            F = Compute_f_new_version(reelx[0:2],reelx[2:4],acc,proportionnality)
            Fdot = (F-OldF)/dt
            OldF = F
        else : 
            F = 0
            Fdot = 0
        #x[2] += dt*v[0]*0.1
        #x[5]+= dt*v[1]*0.1
        #x_internalmodel[2] += dt*v[0]*0.1
        #x_internalmodel[5]+= dt*v[1]*0.1
        v = -L[k].reshape(np.flip(B.shape))@xhat
        #x = A@x-B@L[k].reshape(np.flip(B.shape))@xhat+motor_noise
        
            
        C = np.array([-reelx[3]*(2*reelx[2]+reelx[3])*a2*np.sin(reelx[1]),reelx[2]*reelx[2]*a2*np.sin(reelx[1])])
        Denominator = a3*(a1-a3)-a2*a2*np.cos(reelx[1])*np.cos(reelx[1])
        Minv = np.array([[a3/Denominator,(-a2*np.cos(reelx[1])-a3)/Denominator],[(-a2*np.cos(reelx[1])-a3)/Denominator,(2*a2*np.cos(reelx[1])+a1)/Denominator]])
        
        Denominator = a3*(a1-a3)-a2*a2*np.cos(xhat[3])*np.cos(xhat[3])
        M = np.array([[a1+2*a2*cos(xhat[3]),a3+a2*cos(xhat[3])],[a3+a2*cos(xhat[3]),a3]])
        Minvdot = np.array([[-a3*a2*a2*sin(2*xhat[3])*xhat[4]/(Denominator*Denominator),
                             (a2*sin(xhat[3])*xhat[4]*Denominator+(a2*cos(xhat[3])+a3)*a2*a2*sin(2*xhat[3])*xhat[4])/(Denominator*Denominator)],
                            [(a2*sin(xhat[3])*xhat[4]*Denominator+(a2*cos(xhat[3])+a3)*a2*a2*sin(2*xhat[3])*xhat[4])/(Denominator*Denominator),
                            (-2*a2*sin(x[3])*xhat[4]*Denominator+(2*a2*cos(xhat[3])+a1)*a2*a2*sin(2*xhat[3])*xhat[4])/(Denominator*Denominator)]])
            
        Cdot = np.array([-a2*xhat[5]*(2*xhat[1]+xhat[4])*sin(xhat[3])-a2*xhat[4]*(2*xhat[2]+xhat[5])*sin(xhat[3])
                         -a2*xhat[4]*xhat[4]*(2*xhat[1]+xhat[4])*cos(xhat[3]),2*xhat[1]*xhat[2]*a2*sin(xhat[3])+xhat[1]*xhat[1]*a2*cos(xhat[3])*xhat[4]])
        K = 1/0.06

        if alpha == 1 :
            COLORS = "blue"
            LABEL = r"Adapted Feedback Linearization: $\alpha$ = 1"
        elif alpha == 0 :
            COLORS = "red"
            LABEL = "Classic Feedback Lineariation"
        else :
            COLORS = "orange"
            LABEL = r"Adapting Feedback Linearization: $\alpha$ =  "+str(alpha)
        u = 1/K*M@(v-alpha*Fdot)-1/K*M@Minvdot@M@(np.array([xhat[2],xhat[5]])-alpha*F)+M@(np.array([xhat[2],xhat[5]])-alpha*F)+C+Bdyn@np.array([xhat[1],xhat[4]])+1/K*Cdot+1/K*Bdyn@np.array([xhat[2],xhat[5]])
        Command_Array[k,0,:] = u
        Command_Array[k,1,:] = v

        new_reelx[0:2] += dt*reelx[2:4]
        #print("Start\n",new_reelx[2:4]-np.array([x[1],x[4]]),"\n")
        new_reelx[2:4] += dt*(Minv@(reelx[4:6]-Bdyn@(reelx[2:4])-C)+F)  
        #print(new_reelx[2:4]-np.array([(A@x+B@v)[1],(A@x+B@v)[4]]),"\n")
        new_reelx[4:6] += dt*K*(u-reelx[4:6])
        #print("HERE: ",new_reelx,new_reelxhat)
            
        array_xhat[k+1] = xhat.flatten()
        array_x[k+1]= x.flatten()
        array_reelx[k+1] = new_reelx.flatten()

        

        x_changevar[0],x_changevar[1],x_changevar[3],x_changevar[4] = new_reelx[0],new_reelx[2],new_reelx[1],new_reelx[3]
        #print("Diff :",x-x_internalmodel,"\n")
        x_changevar = x_changevar + B@v
        Omega_sens,motor_noise,Omega_measure,measure_noise = Bruitage(Bruit,NbreVar,Noise_Variance)
        x = A@x +B@v + motor_noise
        #print(x-x_changevar)
        #print("TEST: ",x-newx,"\n")
        xhat = A@xhat+B@v

        #print(x-newx)
        #x+=motor_noise  
        if DisplayNonlinear : y[k] = (H@x_changevar+measure_noise).flatten()
        else : y[k] = (H@x+measure_noise).flatten() #For numerical errors, both lines computes same estimates up to numerical errors.
        K = A@sigma@H.T@np.linalg.inv(H@sigma@H.T+Omega_measure)
        sigma = Omega_sens + (A - K@H)@sigma@A.T
        #print(y[k]-H@x_internalmodel)
        xhat = xhat + K@(y[k]-H@xhat)
        reelx = np.copy(new_reelx)
        #print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])   

#Plot
    
    x0 = xstart
    if DisplayNonlinear : 
        x[0],x[1],x[3],x[4] = new_reelx[0],new_reelx[2],new_reelx[1],new_reelx[3]
        reelx = array_reelx.T[:,1:][:,::1]
        X = np.cos(reelx[0]+reelx[1])*33+np.cos(reelx[0])*30
        Y = np.sin(reelx[0]+reelx[1])*33+np.sin(reelx[0])*30
    else : 
        reelx = array_x.T[:,1:][:,::1]
        X = np.cos(reelx[0]+reelx[3])*33+np.cos(reelx[0])*30
        Y = np.sin(reelx[0]+reelx[3])*33+np.sin(reelx[0])*30

    if plot : 
        plt.grid(linestyle='--')
        plt.axis("equal")
        plt.plot(X,Y,color = COLORS,label = LABEL,linewidth = .8)
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        plt.scatter([starting_point[0],targets[0]],[starting_point[1],targets[1]],color = "black")

    #print("Optimum values " + str(J1)[:8]+" and "+str(J2)[:8])
    return X,Y

def LQG_with_FF(w1,w2,w3,w4,r1,r2,targets = [0,55],starting_point = [0,30],proportionnality = 0,pert = 0,plot = True,NoiseVar = 1e-6):

    Num_iter = 600
    dt = 0.001

    obj1,obj2 = newton(f,df,1e-8,1000,targets[0],targets[1]) #Defini les targets
    st1,st2 = newton(f,df,1e-8,1000,starting_point[0],starting_point[1])

    xstart = np.array([st1,0,0,st2,0,0,obj1,0,obj2,0])
    x0 = np.array([st1,0,0,st2,0,0,obj1,0,obj2,0])
    xnonlin0 = np.concatenate((x0[:7],np.array([x0[8]])))
    Bruit = True
    NbreVar = 8
    
    #Define Weight Matrices
    #Q = np.array([[2*w1/5+w1,0,0,-2*w1/5,0,0,w1/5-w1,0],[0,w3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
    #            [-2*w1/5,0,0,2*w2/5+w2,0,0,-w2/5,-w2],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
     #           [w1/5-w1,0,0,-w2/5,0,0,w1+w1/5,0],[0,0,0,-w2,0,0,0,w2]])
    R = np.array([[r1,0],[0,r2]])
    Q = np.array([[w1,0,0,0,0,0,-w1,0],[0,w2,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
             [0,0,0,w3,0,0,0,-w3],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
             [-w1,0,0,0,0,0,w1,0],[0,0,0,-w3,0,0,0,w3]])
    
    
    #Define Dynamic Matrices  

    Az = np.array([[1,dt,0,0,0,0,0,0],[0,1+dt*(-0.5*a1+0.025*a3)/((a1-a3)*a3),dt*a1/((a1-a3)*a3),0,dt*(-0.025*a1+0.5*a3)/((a1-a3)*a3),dt/(a3-a1),0,0],
     [0,0,1-dt/tau,0,0,0,0,0],[0,0,0,1,dt,0,0,0],[0,dt*0.475/(a1-a3),-dt/(a1-a3),0,1-dt*0.475/(a1-a3),dt/(a1-a3),0,0],
     [0,0,0,0,0,1-dt/tau,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])

    Bz = np.transpose([[0,0,dt/tau,0,0,0,0,0],[0,0,0,0,0,dt/tau,0,0]])
    
    
    #Reverse Iterations
    #print(Qnonlin.astype(int))
    S = Q

    
    array_L = np.zeros((Num_iter-1,2,NbreVar))   
    array_S = np.zeros((Num_iter,NbreVar,NbreVar)) 
    array_S[-1] = Q
    for k in range(Num_iter-1):
        L = np.linalg.inv(R+Bz.T@S@Bz)@Bz.T@S@Az
        array_L[Num_iter-2-k] = L
        S = Az.T@S@(Az-Bz@L)
        array_S[Num_iter-2-k] = S
        
    #print(array_L[0])
    #Feedback
    H,L,x0,A,B = np.identity(8),array_L,xnonlin0,Az,Bz
        
    array_x = np.zeros((Num_iter,NbreVar))
    array_xhat = np.zeros((Num_iter,NbreVar))
    array_reelx = np.zeros((Num_iter,NbreVar-2)) 
    y = np.zeros((Num_iter,NbreVar))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()
    xhat = x0
    x = x0
    reelx = np.zeros(NbreVar-2)
    reelx[0] = x[0]
    reelx[1] = x[3]
    new_reelx = reelx
    sigma = np.identity(NbreVar)*10**-6 #Espérance de (erreur erreur^) avec erreur = x - xhat
    F = np.zeros(2)
    for k in range(Num_iter-1):
        Omega_sens,motor_noise,Omega_measure,measure_noise = Bruitage(Bruit,NbreVar,NoiseVar)
        acc = np.array([(reelx[2]-array_reelx[k-1][2])/dt,(reelx[3]-array_reelx[k-1][3])/dt])
        if np.sin(reelx[0]+reelx[1])*33+np.sin(reelx[0])*30 > pert:
            F = Compute_f_new_version(reelx[0:2],reelx[2:4],acc,proportionnality)
        else : 
            F = 0
        

        x[0],x[1],x[3],x[4],x[2],x[5] = reelx[0],reelx[2],reelx[1],reelx[3],reelx[4],reelx[5]
        x = x+motor_noise        
        
        y[k] = (H@x+measure_noise).flatten()
        K = A@sigma@H.T@np.linalg.inv(H@sigma@H.T+Omega_measure)
        sigma = Omega_sens + (A - K@H)@sigma@A.T
        xhat = A@xhat - B@L[k].reshape(np.flip(B.shape))@xhat + K@(y[k]-H@xhat)
        u = -L[k].reshape(np.flip(B.shape))@xhat
            
        C = np.array([-reelx[3]*(2*reelx[2]+reelx[3])*a2*np.sin(reelx[1]),reelx[2]*reelx[2]*a2*np.sin(reelx[1])])
        Denominator = a3*(a1-a3)-a2*a2*np.cos(reelx[1])*np.cos(reelx[1])
        Minv = np.array([[a3/Denominator,(-a2*np.cos(reelx[1])-a3)/Denominator],[(-a2*np.cos(reelx[1])-a3)/Denominator,(2*a2*np.cos(reelx[1])+a1)/Denominator]])
        new_reelx[0:2] += dt*reelx[2:4]
        new_reelx[2:4] += dt*(Minv@(reelx[4:6]-Bdyn@(reelx[2:4])-C)+F)  
        K = 1/0.06
        new_reelx[4:6] += dt*K*(u-reelx[4:6])
            
        array_xhat[k+1] = xhat.flatten()
        array_x[k+1]= x.flatten()
        array_reelx[k+1] = new_reelx.flatten()
        reelx = new_reelx 
        #print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])   

#Plot
    x0 = xstart
    reelx = array_reelx.T[:,1:][:,::1]
    if plot : plt.plot(np.cos(reelx[0]+reelx[1])*33+np.cos(reelx[0])*30,np.sin(reelx[0]+reelx[1])*33+np.sin(reelx[0])*30,color = "green",label = "Control of the Linear System")


class InvalidTrial(Exception):
    "Raised when the input value is less than 18"
    pass

from scipy.interpolate import interp1d

def preprocess(data,trial,N = 600):
    go = np.where(data[trial]["EVENTS"]["LABELS"][0,0] == 'gocue')[1]
    stop = np.where(data[trial]["EVENTS"]["LABELS"][0,0] == 'target_reached')
    if len(stop[1]) == 0 : 
        raise InvalidTrial
    else : 
        stop = stop[1]
    start = data[trial]["Right_FS_TimeStamp"][0] + data[trial]["EVENTS"]["TIMES"][0,0][0,go]
    end = data[trial]["Right_FS_TimeStamp"][0] + data[trial]["EVENTS"]["TIMES"][0,0][0,stop]
    start_index = np.where(data[trial]["Right_FS_TimeStamp"] > start)[0][0]
    end_index = np.where(data[trial]["Right_FS_TimeStamp"] > end)[0][0]
    X = data[trial]["Right_HandX"].flatten()[start_index:end_index]
    Y = data[trial]["Right_HandY"].flatten()[start_index:end_index]
    goload = np.where(data[trial]["EVENTS"]["LABELS"][0,0] == 'pert_load')[1]
    start_load = data[trial]["Right_FS_TimeStamp"][0] + data[trial]["EVENTS"]["TIMES"][0,0][0,goload]
    start_index_load = Y[np.where(data[trial]["Right_FS_TimeStamp"] > start_load)[0][0] - start_index] * 100
    thetas = np.pi - data[trial]["Right_L1Ang"].flatten()[start_index:end_index]
    thetae = -1*(data[trial]["Right_L2Ang"].flatten()[start_index:end_index] - data[trial]["Right_L1Ang"].flatten()[start_index:end_index])
    Time = data[trial]["Right_FS_TimeStamp"][start_index:end_index] - data[trial]["Right_FS_TimeStamp"][0]
    x_interp = interp1d( Y,X)
    Y = np.linspace(0.095,0.26,N)
    X = x_interp(Y) - x_interp(0.095)




    return X*100,Y*100+DEVIATION,Time,thetas,thetae,start_index_load

def Valid_Trial(data,trial):


    for u in data[trial]["EVENTS"][0,0][0][0]:
        if u[0] == "success": 
            if (np.max(data[trial]["Right_HandYVel"].flatten())*13-np.max((data[trial]["Right_Hand_ForceCMD_X"]).flatten()))>1: return False
            
            return True
    return False

def Average_Curve(data,trial):
    total_X = np.zeros(1000)
    total_valid_trials = 0
    for trial in range(250):
        if Valid_Trial(data,trial):
            total_valid_trials += 1
            X,Y,t,thetas,thetae,load = preprocess(data,trial)
            total_X+=X
        #print(data[trial]["ANALOG"]["DESCRIPTIONS"])
        #print(np.array([thetas[700],thetae[700]]))
        #print(np.cos(t1+t2)*33+np.cos(t1)*30,np.sin(t1+t2)*33+np.sin(t1)*30)
        #print(newton(f,df,1e-8,1000,9,8))
    Y = np.linspace(9.5,26,1000)+DEVIATION
    total_X = total_X/total_valid_trials
    plt.plot(total_X,Y)

    print(total_X[0],total_X[-1],Y[0],Y[-1])
    plt.axis("equal")
    plt.show()

def Compute_deviation(X1,X2):
    total_error = 0
    for i in range(len(X1)):
        total_error += (X1[i]-X2[i])*(X1[i]-X2[i])
    return total_error

OPTIMAL_FACTORS = np.array([0.3775,0.3475,0.5,0.4725,0.43,0.445,0.37,0.2,0.4225,0.4075,0.2825,0.5])

def Compute_Mean_Error():
    length = 0
    TOTAL_SUBJ = 12
    Total_Better_Array = np.zeros((TOTAL_SUBJ,250-length))
    for subj in range(1,TOTAL_SUBJ+1):
        data = loadmat("Data/XP1/F"+str(subj)+"_data.mat")["F"+str(subj)+"_data"][0]
        X,Y = Feedback_Linearization_with_FF(1e7,1e7,1e4,1e4,1e-6,1e-6,alpha = 1,starting_point= [0,9.4+DEVIATION],targets = [1,26.1+DEVIATION],proportionnality=OPTIMAL_FACTORS[subj-1],plot = False,pert = ONSET_PERTURBATION,DisplayNonlinear=True)
        x_interp = interp1d( Y,X)
        Y = np.linspace(9.5+DEVIATION,26+DEVIATION,100) 
        Xref = x_interp(Y)
        Error_Array = []
        for trial in range(250):
            if Valid_Trial(data,trial):
                Error_Array.append(Compute_deviation(preprocess(data,trial,N = 100)[0],Xref))
            else:
                Error_Array.append(None)
        Total_Better_Array[subj-1] = Error_Array

    plt.plot(np.arange(0,250-length),np.nanmean(Total_Better_Array,axis = 0))

def Compute_Alpha():
    TOTAL_SUBJ = 12
    Alphas = np.zeros((TOTAL_SUBJ,250))
    for subj in range(1,TOTAL_SUBJ+1):
        data = loadmat("Data/XP1/F"+str(subj)+"_data.mat")["F"+str(subj)+"_data"][0]
        Big_X = np.zeros((101,100))
        for alpha in np.linspace(0,1,100):
            
            X,Y = Feedback_Linearization_with_FF(1e7,1e7,1e4,1e4,1e-6,1e-6,alpha = 1,starting_point= [0,9.4+DEVIATION],targets = [1,26.1+DEVIATION],proportionnality=OPTIMAL_FACTORS[subj-1],plot = False,pert = ONSET_PERTURBATION,DisplayNonlinear=True)
            x_interp = interp1d( Y,X)
            Y = np.linspace(9.5+DEVIATION,26+DEVIATION,100)
            Xref = x_interp(Y)
            Big_X[int(alpha*100)] = Xref
        for trial in range(250):
            min_e = np.infty
            min_a = -1
            if Valid_Trial(data,trial):
                for best_alpha in range(101):
                    e = Compute_deviation(preprocess(data,trial,N = 100)[0],Big_X[best_alpha])
                    if e < min_e :
                        min_e = e
                        min_a = best_alpha/100
                Alphas[subj-1,trial] = min_a
            else:
                Alphas[subj-1,trial] = nan
        #Better_Error_Array = np.zeros(250-length)
        #Total_Better_Array[subj-1] = Better_Error_Array
    plt.plot(np.arange(0,250),np.nanmean(Alphas,axis = 0))
    plt.xlabel("Trials")
    plt.ylabel("Alpha")
def Optimal_Factors(ACTIVATION):
    if ACTIVATION :
        OPTIMAL_FACTORS = []
        for subj in range(1,13):
            min_error = np.infty
            best_factor = -1
            data = loadmat("Data/XP1/F"+str(subj)+"_data.mat")["F"+str(subj)+"_data"][0]
            for factor in np.linspace(.1,.5,161):
                X,Y = Feedback_Linearization_with_FF(1e7,1e7,1e4,1e4,1e-6,1e-6,alpha = 0,starting_point= [0,9.4+DEVIATION],targets = [1,26.1+DEVIATION],proportionnality=factor,plot = False,pert = ONSET_PERTURBATION,DisplayNonlinear=True)
                x_interp = interp1d( Y,X)
                Y = np.linspace(9.5+DEVIATION,26+DEVIATION,1000)
                Xref = x_interp(Y)
                total_error = 0
                #Error_Array = []
                for trial in range(10):
                    if Valid_Trial(data,trial):
                        #Error_Array.append(Compute_deviation(preprocess(data,trial)[0]*100,Xref))
                        total_error += Compute_deviation(preprocess(data,trial)[0],Xref)
                if total_error < min_error :
                    min_error = total_error
                    best_factor = factor
            print(best_factor)
            OPTIMAL_FACTORS.append(best_factor)
        return OPTIMAL_FACTORS
    return 0

def Resampling_Illustration():
    trial = 30
    data = loadmat("Data/XP1/F12_data.mat")["F12_data"][0]
    plt.figure(figsize=(7,11))
    stop = np.where(data[trial]["EVENTS"]["LABELS"][0,0] == 'target_reached')[1]
    start = data[trial]["Right_FS_TimeStamp"][0] + data[trial]["EVENTS"]["TIMES"][0,0][0,np.where(data[trial]["EVENTS"]["LABELS"][0,0] == 'gocue')[1]]
    start_index = np.where(data[trial]["Right_FS_TimeStamp"] > start)[0][0]
    end = data[trial]["Right_FS_TimeStamp"][0] + data[trial]["EVENTS"]["TIMES"][0,0][0,stop]
    end_index = np.where(data[trial]["Right_FS_TimeStamp"] > end)[0][0]
    X = data[trial]["Right_HandX"].flatten()[start_index:end_index]
    Y = data[trial]["Right_HandY"].flatten()[start_index:end_index]
    x_interp = interp1d( Y,X)
    Y = np.linspace(0.095,0.26,100)
    X = x_interp(Y) - x_interp(0.095)
    plt.plot(data[trial]["Right_HandX"].flatten()*100- x_interp(0.095)*100,data[trial]["Right_HandY"].flatten()*100+20,label = "Initial trajectory from the dataset")


    X,Y,t,thetas,thetae,load = preprocess(data,trial)
    print(Y)


    plt.plot(X,Y,label = "Preprocessed trajectory")
    plt.axis("equal")
            
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.grid()
    plt.legend()
    plt.plot(np.linspace(-1.25,1.25,100),np.ones(100)*27,color = "black")
    plt.plot(np.linspace(-1.25,1.25,100),np.ones(100)*29.5,color = "black")
    plt.plot(np.ones(100)*1.25,np.linspace(27,29.5,100),color = "black")
    plt.plot(np.ones(100)*-1.25,np.linspace(27,29.5,100),color = "black",label = "Initial square")

    plt.plot(np.linspace(-1.25,1.25,100),np.ones(100)*47,color = "black")
    plt.plot(np.linspace(-1.25,1.25,100),np.ones(100)*49.5,color = "black")
    plt.plot(np.ones(100)*1.25,np.linspace(47,49.5,100),color = "black")
    plt.plot(np.ones(100)*-1.25,np.linspace(47,49.5,100),color = "black",label = "Initial square")
    #plt.xlim((-10,10))


DEVIATION = 20
ONSET_PERTURBATION = 13.75 + DEVIATION


def phi_arm(xhat):
    return np.array([np.cos(xhat[0]+xhat[3]),np.sin(xhat[0]+xhat[3]),np.cos(xhat[0]),np.sin(xhat[0])]).reshape(4,1)

def phi_extended(xhat):
    return np.concatenate((xhat,[np.cos(xhat[0]+xhat[3]),np.sin(xhat[0]+xhat[3]),np.cos(xhat[0]),np.sin(xhat[0])])).reshape(12,1)

def Simulation_FF_Adaptative(w1,w2,w3,w4,r1,r2,targets = [0,55],starting_point = [0,30],proportionnality = 0,plot = True,pert = 0,theta = -1,color = "blue",adaptive = False,ref = np.zeros((1,1)),errortype = "Kalman",gamma = 2.5,toBasis = None,NLTorque = False):
    if type(theta) == int: theta = np.zeros((len(toBasis(np.zeros(8))),2))
    Num_iter = 60
    dt = 0.01
    LOAD_ACTIVATION = 0
    st1,st2 = newton(f,df,1e-8,1000,starting_point[0],starting_point[1])
    obj1,obj2 = newton(f,df,1e-8,1000,targets[0],targets[1])
    obj3,obj4 = newton(f,df,1e-8,1000,targets[0],targets[1])

    xstart = np.array([st1,0,0,st2,0,0,obj1,0,obj2,0])
    x0 = np.array([st1,0,0,st2,0,0,obj1,0,obj2,0])
    xnonlin0 = np.concatenate((x0[:7],np.array([x0[8]])))
    Bruit = True
    NbreVar = 8
    Td = 0.066
    
    #Define Weight Matrices
    Rnonlin = np.array([[r1,0],[0,r2]])
    Q = np.array([[w1,0,0,0,0,0,-w1,0],[0,w3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
               [0,0,0,w2,0,0,0,-w2],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
               [-w1,0,0,0,0,0,w1,0],[0,0,0,-w2,0,0,0,w2]])
    #Q = np.array([[2*w1/5+w1,0,0,-2*w1/5,0,0,w1/5-w1,0],[0,w3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
    #            [-2*w1/5,0,0,2*w2/5+w2,0,0,-w2/5,-w2],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
     #           [w1/5-w1,0,0,-w2/5,0,0,w1+w1/5,0],[0,0,0,-w2,0,0,0,w2]])
    Qnonlin = Q
    
    
    #Define Dynamic Matrices  
    Az = np.array([[1,dt,0,0,0,0,0,0],[0,1,dt,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,dt,0,0,0],[0,0,0,0,1,dt,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
    Bz = np.array([[0,0],[0,0],[dt,0],[0,0],[0,0],[0,dt],[0,0],[0,0]])
    
    
    #Reverse Iterations
    #print(Qnonlin.astype(int))
    S = Qnonlin

    
    array_L = np.zeros((Num_iter-1,2,NbreVar))   
    array_S = np.zeros((Num_iter,NbreVar,NbreVar)) 
    array_S[-1] = Qnonlin
    for k in range(Num_iter-1):
        L = np.linalg.inv(Rnonlin+Bz.T@S@Bz)@Bz.T@S@Az
        array_L[Num_iter-2-k] = L
        S = Az.T@S@(Az-Bz@L)
        array_S[Num_iter-2-k] = S
        
    #print(array_L[0])
    #Feedback
    H,L,x0,A,B = np.identity(8),array_L,xnonlin0,Az,Bz
        
    array_x = np.zeros((Num_iter,NbreVar))
    array_xhat = np.zeros((Num_iter,NbreVar))
    array_reelx = np.zeros((Num_iter,NbreVar-2)) 
    y = np.zeros((Num_iter,NbreVar))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()
    xhat = x0
    x = x0
    reelx = np.zeros(NbreVar-2)
    reelx[0] = x[0]
    reelx[1] = x[3]
    new_reelx = reelx
    sigma = np.identity(NbreVar)*10**-6 #Espérance de (erreur erreur^) avec erreur = x - xhat
    F = np.zeros(2)
    tau = np.zeros(2)
    a = np.zeros(2)
    new_ref = np.zeros((Num_iter,8))
    if ref[0,0] == 0: 
        ref = new_ref
        adaptive = False
    new_ref[0] = x0
    P = scipy.linalg.solve_continuous_lyapunov(A, -np.identity(A.shape[0]))
    
    for k in range(Num_iter-1):
        acc = np.array([(reelx[2]-array_reelx[k-1][2])/dt,(reelx[3]-array_reelx[k-1][3])/dt])
        if np.sin(reelx[0]+reelx[1])*33+np.sin(reelx[0])*30 > pert:

            if LOAD_ACTIVATION == 0:
                x[6:8] = np.array([obj3,obj4])
                xhat[6:8] = np.array([obj3,obj4])
                X = np.cos(reelx[0]+reelx[1])*33+np.cos(reelx[0])*30
                Y = np.sin(reelx[0]+reelx[1])*33+np.sin(reelx[0])*30
                LOAD_ACTIVATION = 1
                #if plot : 
                    #plt.scatter(X,Y,color = "red")
            F = Compute_f_new_version(reelx[0:2],reelx[2:4],acc,proportionnality)
        else : 
            F = 0
        
        
        x[0],x[1],x[3],x[4] = reelx[0],reelx[2],reelx[1],reelx[3]
        x[2] = acc[0]
        x[5] = acc[1]   
        v = -L[k].reshape(np.flip(B.shape))@xhat
        if errortype == "Kalman":
            e = x-xhat
        elif errortype == "Reference":
            e = x-ref[k]
        
        phi = toBasis(xhat)
        theta = theta - gamma *(1/(1+np.exp((k-Num_iter/2)/Num_iter*10)))* phi @ e.reshape((8,1)).T@P@B
        
        if adaptive : 
            v-= (theta.T@phi).reshape(2) 
        Omega_sens,motor_noise,Omega_measure,measure_noise = Bruitage(Bruit,NbreVar,1e-12)
        y[k] = (H@x+measure_noise).flatten()
        K = A@sigma@H.T@np.linalg.inv(H@sigma@H.T+Omega_measure)
        sigma = Omega_sens + (A - K@H)@sigma@A.T
        xhat = A@xhat + B@v + K@(y[k]-H@xhat)
        x = A@x+B@v+motor_noise
        new_ref[k+1] = x

        """
        error computation
        """
        #e = x - xhat
        
            
        C = np.array([-reelx[3]*(2*reelx[2]+reelx[3])*a2*np.sin(reelx[1]),reelx[2]*reelx[2]*a2*np.sin(reelx[1])])
        Denominator = a3*(a1-a3)-a2*a2*np.cos(reelx[1])*np.cos(reelx[1])
        Minv = np.array([[a3/Denominator,(-a2*np.cos(reelx[1])-a3)/Denominator],[(-a2*np.cos(reelx[1])-a3)/Denominator,(2*a2*np.cos(reelx[1])+a1)/Denominator]])
        new_reelx[0:2] += dt*reelx[2:4]
        new_reelx[2:4] += dt*(Minv@(reelx[4:6]-Bdyn@(reelx[2:4])-C)+F)  
        M = np.array([[a1+2*a2*cos(reelx[1]),a3+a2*cos(reelx[1])],[a3+a2*cos(reelx[1]),a3]])
        Minvdot = np.array([[-a3*a2*a2*sin(2*reelx[1])*reelx[3]/(Denominator*Denominator),
                             (a2*sin(reelx[1])*reelx[3]*Denominator+(a2*cos(reelx[1])+a3)*a2*a2*sin(2*reelx[1])*reelx[3])/(Denominator*Denominator)],
                            [(a2*sin(reelx[1])*reelx[3]*Denominator+(a2*cos(reelx[1])+a3)*a2*a2*sin(2*reelx[1])*reelx[3])/(Denominator*Denominator),
                            (-2*a2*sin(reelx[1])*reelx[3]*Denominator+(2*a2*cos(reelx[1])+a1)*a2*a2*sin(2*reelx[1])*reelx[3])/(Denominator*Denominator)]])
            
        Cdot = np.array([-a2*x[5]*(2*x[1]+x[4])*sin(x[3])-a2*x[4]*(2*x[2]+x[5])*sin(x[3])
                         -a2*x[4]*x[4]*(2*x[1]+x[4])*cos(x[3]),2*x[1]*x[2]*a2*sin(x[3])+x[1]*x[1]*a2*cos(x[3])*x[4]])
        K = 1/0.06
        
        COLORS = color
        dottau = M@(v-Minvdot@M@np.array([x[2],x[5]]))+Cdot+Bdyn@np.array([x[2],x[5]])
        if NLTorque:
            u = Compute_Command_NL(a,dottau)
            a = a + dt*(u-a)/Td
        else:
            #u = Compute_Command(dottau,M,x,C,Bdyn)
            u = dottau/K + tau 
            tau = tau + dt * (u-tau)*K
        new_reelx[4:6] += dt*K*(u-reelx[4:6])
            
        array_xhat[k+1] = xhat.flatten()
        array_x[k+1]= x.flatten()
        array_reelx[k+1] = new_reelx.flatten()
        reelx = new_reelx 
        #print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])   

#Plot
    x0 = xstart
    reelx = array_reelx.T[:,1:][:,::1]
    X = np.cos(reelx[0]+reelx[1])*33+np.cos(reelx[0])*30
    Y = np.sin(reelx[0]+reelx[1])*33+np.sin(reelx[0])*30
    if plot : 
        lw = .8 if (COLORS == "black" or COLORS == "orange") else .1
        plt.plot(X,Y,color = COLORS,linewidth = lw,label = "Adaptive Controller")
    return theta,new_ref,x@Q@x

def run(Num_Sim = 100,Basis = phi_arm,error_func = "Kalman",gamma_choice = 3.5,plotting = 1,factor = 0):
    theta,reelx,cost = Simulation_FF_Adaptative(1e7,1e7,1000,1000,1e-6,1e-6,starting_point= [0,29.4],targets = [0,46.1],proportionnality=factor,plot = False,pert = 69,adaptive = True,errortype=error_func,toBasis=Basis)
    oldcost = cost
    for sim in range(Num_Sim-1):
        color = "orange" if sim < 5 else "green"
        if sim > Num_Sim - 4: color = "black"
        willIplot = 1 if plotting == 1 else 0
        theta,useless,cost = Simulation_FF_Adaptative(1e7,1e7,1000,1000,1e-6,1e-6,starting_point= [0,29.4],targets = [0,46.1],proportionnality=factor,plot = willIplot,pert = 29,theta = theta,color = color,adaptive = True,ref = reelx,errortype = error_func,gamma = gamma_choice,toBasis=Basis)
        if cost > oldcost and sim>0 and error_func == "Kalman":
            print(sim)
            break
        oldcost = cost
    if plotting > 0:
        theta,useless,cost = Simulation_FF_Adaptative(1e7,1e7,1000,1000,1e-6,1e-6,starting_point= [0,29.4],targets = [0,46.1],proportionnality=factor,plot = True,pert = 29,theta = theta,color = "black",adaptive = True,ref = reelx,errortype = error_func,gamma = gamma_choice,toBasis=Basis)
        plt.axis("equal")   
        plt.xlabel,plt.ylabel ="x [cm]","y [cm]"
        plt.grid()
        plt.scatter([0],[46.1],color = "red",marker= "s")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
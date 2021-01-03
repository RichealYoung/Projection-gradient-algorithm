import torch
from torch.autograd.functional import jacobian
import matplotlib.pylab as plt
import numpy as np
from matplotlib.patches import Circle
plt.rcParams['savefig.dpi'] = 300
def plot_solution(sol_list,sol_hat_list):
    # Track of error
    xstar=sol_list[-1].squeeze()
    pstar=objective(xstar)
    iter_list=np.arange(len(sol_list))
    logerror=[torch.log(abs(objective(sol)-pstar)) for sol in sol_list]
    # logerror
    plt.grid()
    plt.xlabel('Step')
    plt.ylabel('Log Error')
    plt.plot(iter_list,logerror,'-r.',)
    plt.title('Track of Log Error for PGD method')
    plt.savefig('Track of Log Error for PGD demo.png')
    plt.close()
    # Track of solutions
    fig,ax=plt.subplots()
    sol_list=torch.stack(sol_list)
    sol_hat_list=torch.stack(sol_hat_list)
    step = 0.01
    xmin=-0.1
    xmax=0.8
    xnum=int((xmax-xmin)/step)
    ymin=-0.4
    ymax=0.4
    ynum=int((ymax-ymin)/step)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    # objective
    x1 = torch.arange(xmin,xmax,step)
    x2 = torch.arange(ymin,ymax,step)
    X1,X2 = torch.meshgrid(x1,x2)
    X=torch.stack([X1.unsqueeze(0),
    X2.unsqueeze(0)])
    Y=objective(X).squeeze()
    cs=plt.contour(X1,X2,Y,15,alpha=1,cmap = plt.get_cmap('viridis'))
    ax.clabel(cs, inline=True, fontsize=8)
    # constraints
    circle = Circle(center, radius,facecolor='cadetblue',alpha=1,edgecolor='k')
    ax.add_patch(circle)
    # solutions
    plt.plot(sol_list[:,0],sol_list[:,1],'-',marker='.',color='red')
    for i in range(5):
        plt.arrow(sol_hat_list[i+1,0],sol_hat_list[i+1,1],sol_list[i+1,0]-sol_hat_list[i+1,0],sol_list[i+1,1]-sol_hat_list[i+1,1],
            head_width=0.01,color='darkgreen',length_includes_head=True)
        plt.arrow(sol_list[i+0,0],sol_list[i+0,1],sol_hat_list[i+1,0]-sol_list[i+0,0],sol_hat_list[i+1,1]-sol_list[i+0,1],
            head_width=0.01,color='royalblue',length_includes_head=True)

    # set 
    plt.title('Track of solutions for PGD method')
    plt.savefig('Track of solutions for PGD demo.png')
    plt.close()
def objective(x):
    # return torch.log(torch.sum(torch.exp(x)))
    return (1-x[0])**2+2*(x[1]-x[0]**2)**2

def constraints(x):
    return torch.sum((x-center)**2)<=radius**2
def projector(x):
    eps=1e-16
    if constraints(x):
        return x
    else:
        return center+(x-center)/(torch.norm(x-center)+eps)*radius
def PGD_step(xk):
    # GD_step
    xk=xk.reshape(-1,1)
    dx_orig=-jacobian(objective,xk).reshape(-1,1)
    alphak=backtracking(xk,dx_orig)
    # project
    xk1_hat=xk+alphak*dx_orig
    xpk=projector(xk1_hat)
    dx=xpk-xk
    return dx,xk1_hat
def backtracking(xk,dx):
    t=1
    alpha=0.49
    beta=0.8
    for i in range(100000):
        if objective(xk+t*dx)<=objective(xk)+alpha*t*jacobian(objective,xk).reshape(-1,1).T@dx:
            return t
        t*=beta
    return t
def main():
    eps=1e-6
    sol_list=[]
    sol_hat_list=[]
    xk=x0
    sol_list.append(xk)
    sol_hat_list.append(xk)
    for i in range(100):
        dx,xk1_hat=PGD_step(xk)
        t=backtracking(xk,dx)
        xk=xk+t*dx
        sol_list.append(xk)
        sol_hat_list.append(xk1_hat)
        if torch.norm(sol_list[-1]-sol_list[-2])<=eps:
            break
    # optim solution
    xstar=sol_list[-1].squeeze()
    print('============xstar=============\n',xstar)
    pstar=objective(xstar)
    print('============pstar=============\n',pstar)
    plot_solution(sol_list,sol_hat_list)
if __name__=='__main__':
    torch.manual_seed(1)
    radius=0.4
    center=torch.tensor([0,0.3]).reshape(-1,1)
    x0=torch.tensor([0,-0.25]).reshape(-1,1)
    main()
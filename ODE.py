import numpy as np
import matplotlib.pyplot as plt

print("Seán Mooney 20334066")


# define functional form of f(x,y)
def f(x,y):
	f = y * (1 + x) + 1 - 3 * x + x ** 2
	return f


# define simple euler update
def seuler(x,y,step):
	y_new = y + step*f(x,y)
	return y_new


# define improved euler update
def ieuler(x,y,step):
	y_new = y + 0.5*step*( f(x,y) + f(x+step, y + step*f(x, y)) )
	return y_new

# define Runge Kutta update
def rk(x,y,step):
	k1 = f(x,y)
	k2 = f(x + 0.5*step, y + 0.5*step*k1)
	k3 = f(x + 0.5*step, y + 0.5*step*k2)
	k4 = f(x + step, y + step*k3)
	y_new = y + step/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
	return y_new

# Set the initial values of x and y, define the step size
step = 0.04; start = 0.0; end = 4
y_zero = 0.0655

# n is the number of steps
n = int((end-start)/step)

# pre-allocate arrays
x = np.arange(start,end,step)
seul = np.zeros(n)
ieul = np.zeros(n)
ruku = np.zeros(n)

seul[0] = y_zero
ieul[0] = y_zero
ruku[0] = y_zero

# increment solutions
for i in range(1,n):
	seul[i] = seuler(x[i-1], seul[i-1], step)
	ieul[i] = ieuler(x[i-1], ieul[i-1], step)
	ruku[i] = rk(x[i-1], ruku[i-1], step)

# direction field
nx, ny = .1, .25
xd = np.arange(-0, 5, nx)
yd = np.arange(-3, 3, ny)
X, Y = np.meshgrid(xd, yd)

dy = (1 + X) * Y + 1 - 3 * X + X ** 2
dx = np.ones(dy.shape)
dyu = dy/np.sqrt(dx**2 + dy**2)
dxu = dx/np.sqrt(dx**2 + dy**2)


# plot solutions
plt.quiver(X, Y, dxu, dyu)
plt.plot(x, seul)
plt.ylim(-3, 3)
plt.xlabel("t")
plt.ylabel("x")
plt.title("simple Euler method")
plt.show()

plt.plot(x, ieul)
plt.quiver(X, Y, dxu, dyu)
plt.xlabel("t")
plt.ylabel("x")
plt.ylim(-3, 3)
plt.title("Improved Euler method")
plt.show()

plt.plot(x, ruku)
plt.quiver(X, Y, dxu, dyu)
plt.title("RK method")
plt.xlabel("t")
plt.ylabel("x")
plt.ylim(-3, 3)
plt.show()

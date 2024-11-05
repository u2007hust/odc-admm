import matplotlib.pyplot as plt

import numpy as np

textsize = 12
labelsize = 12
figuresize = (6, 3.5)

timeWindowLength = 12
x = np.linspace(1, timeWindowLength, timeWindowLength) * 1000
z = []
z.append(np.array([1.9318274044073784, 2.151089944389662, 4.09646961248367, 4.614679066956076]))
z.append(np.array([1.6554547972902391, 1.8513236191851874, 3.1495188342929463, 3.491129089631085]))
z.append(np.array([1.4759594539573904, 1.6512266294356568, 2.677134032459503, 2.89744113736448]))
z.append(np.array([1.323893769319072, 1.4539204276251074, 2.336196744705686, 2.513967873348932]))
z.append(np.array([1.239444435074271, 1.371631082214787, 2.144946978036838, 2.3266113303743206]))
z.append(np.array([1.1372426317424569, 1.2775137028139212, 1.9298262950640452, 2.0916453532694286]))
z.append(np.array([1.0500766510798913, 1.1665657912459726, 1.785830253439817, 1.9272006908234425]))
z.append(np.array([1.0242383466999307, 1.1348371689380206, 1.6962714709041584, 1.8279051642586446]))
z.append(np.array([0.9435149183880917, 1.0522167743105229, 1.5927142661660862, 1.7153887205817977]))
z.append(np.array([0.9341787550695584, 1.0350319340604375, 1.529568738681379, 1.641902665797869]))
z.append(np.array([0.8823916999075863, 0.9844077446932102, 1.4504877860174714, 1.5511760560257488]))
z.append(np.array([0.854958500029895, 0.9521645661241311, 1.3992076655620125, 1.5051362486418012]))


y1 = np.zeros(timeWindowLength)
y2 = np.zeros(timeWindowLength)
y3 = np.zeros(timeWindowLength)
y4 = np.zeros(timeWindowLength)
for i in range(timeWindowLength):
    y1[i] = z[i][0]
    y2[i] = z[i][1]
    y3[i] = z[i][2]
    y4[i] = z[i][3]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=figuresize)

plt.plot(x, y1, label="Complete graph", color="blue", linewidth=1.5, ls="--", marker='o', markersize=5)
plt.plot(x, y2, label="ER graph (p = 0.8)", color="red", linewidth=1.5, ls=":", marker='>', markersize=5)
plt.plot(x, y3, label="ER graph (p = 0.5)", color="green", linewidth=1.5, ls="-.", marker='D', markersize=5)
plt.plot(x, y4, label="Cycle graph", color="black", linewidth=1.5, ls="-", marker='s', markersize=5)

plt.xlabel(r"$T$", fontsize=labelsize)
plt.ylabel("Relative Regret (\%)", fontsize=labelsize)
# plt.title("PyPlot First Example")
plt.ylim(0, 5)
plt.xlim(1000, 12000)
plt.xticks(x, fontsize=textsize)
plt.yticks(np.linspace(0, 5, 6), fontsize=textsize)
plt.legend(fontsize=textsize, frameon=True, loc='upper right')
plt.grid(ls=':',
         color='blue')
# plt.ylim(-1.5,1.5)
# plt.legend(fontsize=textsize, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.04))  # 显示左下角的图例

foo_fig = plt.gcf()
foo_fig.savefig('expr2_regret.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.1)



z = []
z.append(np.array([0.8690362840609971, 0.8659002555105204, 0.6518999533827909, 0.5902381586522087]))
z.append(np.array([0.5231165997756108, 0.5236171760327258, 0.41413382580066826, 0.39106218716046887]))
z.append(np.array([0.41388474740829617, 0.41515546050232954, 0.34472522883954027, 0.3296581635766718]))
z.append(np.array([0.347549300654431, 0.35059645602878303, 0.29663366889556564, 0.2815140531761552]))
z.append(np.array([0.3012765877035676, 0.30199180121759167, 0.2592151624173541, 0.2471788509832506]))
z.append(np.array([0.2733724628379147, 0.2740401601055675, 0.23657687979463776, 0.22613174570628117]))
z.append(np.array([0.24830594633812647, 0.24802804459947758, 0.21864972916227593, 0.21087127984135465]))
z.append(np.array([0.23024005094876282, 0.23031801885497588, 0.20448259588168263, 0.19822458197517037]))
z.append(np.array([0.21958950357395465, 0.21950963256097644, 0.19545589426449747, 0.18999310883613202]))
z.append(np.array([0.20176309971296247, 0.2028260082780478, 0.18032420981420655, 0.17567869283720855]))
z.append(np.array([0.19267705369063187, 0.19327988902735346, 0.17276193311926685, 0.16829641825469407]))
z.append(np.array([0.1830582739771996, 0.1830438359715639, 0.16312214220919324, 0.1592636547115423]))



for i in range(timeWindowLength):
    y1[i] = z[i][0]
    y2[i] = z[i][1]
    y3[i] = z[i][2]
    y4[i] = z[i][3]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=figuresize)

plt.plot(x, y1, label="Complete graph", color="blue", linewidth=1.5, ls="--", marker='o', markersize=5)
plt.plot(x, y2, label="ER graph (p = 0.8)", color="red", linewidth=1.5, ls=":", marker='>', markersize=5)
plt.plot(x, y3, label="ER graph (p = 0.5)", color="green", linewidth=1.5, ls="-.", marker='D', markersize=5)
plt.plot(x, y4, label="Cycle graph", color="black", linewidth=1.5, ls="-", marker='s', markersize=5)

plt.xlabel(r"$T$", fontsize=labelsize)
plt.ylabel("Relative Constraint Violation (\%)", fontsize=labelsize)
# plt.title("PyPlot First Example")
plt.ylim(0, 1)
plt.xlim(1000, 12000)
plt.xticks(x, fontsize=textsize)
plt.yticks(np.linspace(0, 1, 6), fontsize=textsize)
plt.legend(fontsize=textsize, frameon=True, loc='upper right')
plt.grid(ls=':',
         color='blue')
# plt.ylim(-1.5,1.5)
# plt.legend(fontsize=textsize, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.04))  # 显示左下角的图例

foo_fig = plt.gcf()
foo_fig.savefig('expr2_vio.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.1)

plt.show()

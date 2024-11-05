import matplotlib.pyplot as plt

import numpy as np

textsize = 12
labelsize = 12
figuresize = (6, 3.5)

timeWindowLength = 16
x = np.linspace(1, timeWindowLength, timeWindowLength) * 10000
z = []
z.append(np.array([133541451.90843844, 74816205.82149744, 0.11105489109969047, 0.06221817623860515]))
z.append(np.array([249939163.29781723, 126701335.44405842, .10354856424608916, 0.05249173919039429]))
z.append(np.array([366874525.68885756, 170751819.40940523, 0.10125545956683668, 0.04712661342105817]))
z.append(np.array([483311616.968812, 211422996.50825596,  0.09999954117434773, 0.043744453686275886]))
z.append(np.array([591253230.0086298, 247840106.22869968, 0.09776043984226801, 0.040978985933182166]))
z.append(np.array([699366077.4786043, 282258841.2761154, 0.09635160339287019, 0.038886775902566395]))
z.append(np.array([810391534.6193695, 314969511.6780453, 0.09571768974866535, 0.03720196067108228]))
z.append(np.array([919071395.133276, 345535090.776523, 0.09501464324834319, 0.03572180959364357]))
z.append(np.array([1026187748.3495369, 375490507.6992321, 0.09431645462540428, 0.03451116375989094]))
z.append(np.array([1133718080.7191963, 403899318.504282, 0.09383500353729879, 0.03342973409802141]))
z.append(np.array([1242438148.1274452, 430857965.36917305, 0.09353481376349726, 0.03243639903528717]))
z.append(np.array([1348598172.1827888, 457170489.0045624, 0.09308163472895448, 0.03155437797865579]))
z.append(np.array([1459842899.9052372, 482428291.09742165, 0.09302433580454261, 0.03074137042799019]))
z.append(np.array([1566732231.2670345, 508232150.5451069, 0.09270693760115334, 0.03007319650874951]))
z.append(np.array([1671953198.6848793, 530977338.2205429, 0.09229278690628795, 0.029310257229091752]))
z.append(np.array([1777626813.7496262, 555160567.198513, 0.09197513330631346, 0.02872423321899986]))

y1 = np.zeros(timeWindowLength)
y2 = np.zeros(timeWindowLength)
y3 = np.zeros(timeWindowLength)
y4 = np.zeros(timeWindowLength)
for i in range(timeWindowLength):
    y1[i] = z[i][0]
    y2[i] = z[i][1]
    y3[i] = z[i][2] * 100
    y4[i] = z[i][3] * 100

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=figuresize)

line1 = plt.plot(x, y2, label="ODC-ADMM regret", color="red", linewidth=1.5, ls="-", marker='o', markersize=5)
line2 = plt.plot(x, y1, label="MOSP regret", color="blue", linewidth=1.5, ls="-", marker='o', markersize=5)
# line1 = plt.plot(x, y1, label="MOSP (regret)", color="blue", linewidth=1.5, ls="-", marker='o', markersize=5)
# line2 = plt.plot(x, y2, label="ODC-ADMM (regret)", color="red", linewidth=1.5, ls="-", marker='o', markersize=5)
# plt.plot(x, y3, label="ER graph (p = 0.5)", color="green", linewidth=1.5, ls="-.", marker='D', markersize=5)
# plt.plot(x, y4, label="Cycle graph", color="black", linewidth=1.5, ls="-", marker='s', markersize=5)

plt.xlabel(r"$T$", fontsize=labelsize)
plt.ylabel("Regret", fontsize=labelsize)
# plt.title("PyPlot First Example")
plt.ylim(1e8, 18*1e8)
plt.xlim(10000, 10000 * timeWindowLength)
plt.xticks(np.linspace(1e4, 16*1e4, 16), fontsize=textsize)
plt.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
# plt.gca().get_xaxis().get_major_formatter().set_useOffset(True)
plt.yticks(np.linspace(0, 18*1e8, 7), fontsize=textsize)
# plt.legend(fontsize=textsize, frameon=True, loc='upper left')
plt.grid(ls=':',
         color='blue')
# plt.ylim(-1.5,1.5)
# plt.legend(fontsize=textsize, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.04))  # 显示左下角的图例

ax2 = plt.twinx()
line3 = ax2.plot(x, y4, label="ODC-ADMM relative regret", color="black", linewidth=1.5, ls="--", marker='D', markersize=5)
line4 = ax2.plot(x, y3, label="MOSP relative regret", color="green", linewidth=1.5, ls="--", marker='D', markersize=5)
# ax2.legend(fontsize=textsize, frameon=True, loc='upper right')
ax2.set_ylim(0, 0.18 * 100)
ax2.set_yticks(np.linspace(0, 0.18, 7) * 100)
ax2.set_ylabel("Relative Regret (\%)", fontsize=labelsize)

lines = line1 + line2 + line3 + line4
labs = [label.get_label() for label in lines]
plt.legend(lines, labs, frameon=True, loc='upper left')

foo_fig = plt.gcf()
foo_fig.savefig('network_problem_regret.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.1)




plt.show()

import matplotlib.pyplot as plt

import numpy as np

textsize = 12
labelsize = 12
figuresize = (6, 3.5)

timeWindowLength = 12
x = np.linspace(1, timeWindowLength, timeWindowLength) * 1000
z = []
z.append(np.array([0.2988821701017554, 0.3665737440883977, 0.9639474028287829, 1.176663973628366]))
z.append(np.array([0.2598324295864594, 0.33044041091217463, 0.7216713889702921, 0.84332730822762]))
z.append(np.array([0.22099135150822863, 0.26775215487081966, 0.5707810763905649, 0.6585444356374316]))
z.append(np.array([0.20585372878997413, 0.24839001569260216, 0.49611032911222364, 0.5708962298573426]))
z.append(np.array([0.18505223346362354, 0.21956896289829717, 0.44858055895654125, 0.5089078342002263]))
z.append(np.array([0.17331272236268166, 0.2104456986935009, 0.40408674175389275, 0.4485813378372156]))
z.append(np.array([0.15595940051548873, 0.1888389908725919, 0.37117931425755407, 0.4074448747351681]))
z.append(np.array([0.15066811257273005, 0.18361013229719472, 0.3420280687043873, 0.3809976532979194]))
z.append(np.array([0.14596184049717068, 0.17442378522614152, 0.3241756130002361, 0.3595694495138891]))
z.append(np.array([0.14110295302973824, 0.16892740333217246, 0.3143054116191287, 0.3480740658374061]))
z.append(np.array([0.13840550536274884, 0.16365985233763333, 0.2982734309487641, 0.3230672453285302]))
z.append(np.array([0.1275838848259989, 0.1535009223165685, 0.2838450458339787, 0.3116097840298404]))

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
plt.ylim(0.05, 0.3)
plt.xlim(1000, 1000 * timeWindowLength)
plt.xticks(x, fontsize=textsize)
plt.yticks(np.linspace(0, 1.2, 7), fontsize=textsize)
plt.legend(fontsize=textsize, frameon=True, loc='upper right')
plt.grid(ls=':',
         color='blue')
# plt.ylim(-1.5,1.5)
# plt.legend(fontsize=textsize, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.04))  # 显示左下角的图例

foo_fig = plt.gcf()
foo_fig.savefig('expr1_regret.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.1)



z = []
z.append(np.array([0.463730598253972, 0.45326631314456856, 0.2811444644729145, 0.2481486360880071]))
z.append(np.array([0.2972755283022305, 0.28477717283178405, 0.19499270720088302, 0.17171263732974168]))
z.append(np.array([0.24241696617429104, 0.23271145008278388, 0.16883778751893502, 0.15461403616965913]))
z.append(np.array([0.19646102149350125, 0.18943243981384428, 0.13322801048507563, 0.12246810452825152]))
z.append(np.array([0.17609904860290765, 0.1715523814974907, 0.1164900699838038, 0.10819467996619991]))
z.append(np.array([0.16319436861742695, 0.1580975220341871, 0.11701778199946071, 0.10806668011176448]))
z.append(np.array([0.15414941260172838, 0.14761525084249572, 0.10404026513703593, 0.09877259590540136]))
z.append(np.array([0.1441852483776564, 0.14028948254398793, 0.1061245183774304, 0.09910274698374975]))
z.append(np.array([0.13417856425357708, 0.131269191726379, 0.1017783443179622, 0.09196184476481152]))
z.append(np.array([0.12502947956200974, 0.12082203679770367, 0.09207643492878435, 0.0876594903707306]))
z.append(np.array([0.11786557183947703, 0.11591673420117668, 0.09011712101093051, 0.08366710577617989]))
z.append(np.array([0.11796265720729834, 0.1153006551097869, 0.09003699820811246, 0.08455373386212638]))


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
plt.ylim(0, 0.3)
plt.xlim(1000, 1000 * timeWindowLength)
plt.xticks(x, fontsize=textsize)
plt.yticks(np.linspace(0, 0.5, 6), fontsize=textsize)
plt.legend(fontsize=textsize, frameon=True, loc='upper right')
plt.grid(ls=':',
         color='blue')
# plt.ylim(-1.5,1.5)
# plt.legend(fontsize=textsize, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.04))  # 显示左下角的图例

foo_fig = plt.gcf()
foo_fig.savefig('expr1_vio.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.1)

plt.show()

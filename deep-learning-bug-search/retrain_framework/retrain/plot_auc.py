import matplotlib.pyplot as mplt

testroc = np.load('testroc.npz')
validroc = np.load('validroc.npz')

fpr = testroc['fpr']
tpr = testroc['tpr']

mplt.plot(fpr, tpr)
mplt.save('roc_curve_test.png')

fpr = validroc['fpr']
tpr = validroc['tpr']

mplt.plot(fpr, tpr)
mplt.save('roc_curve_valid.png')

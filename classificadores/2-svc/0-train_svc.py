# Copyright (c) 2024 Visao Robotica Imagem (VRI)
#  - Felipe Bombardelli <felipebombardelli@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

# =============================================================================
#  Header
# =============================================================================

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# =============================================================================
#  Main
# =============================================================================

# Le os dados X e Y
print("Lendo os Dados")
x_data = []
y_real = []
fd = open("gen_db_dados.csv")
raw = fd.read()
fd.close()
for _line in raw.split('\n'):
    line = _line.split(',')
    if len(line) < 50:
        continue
    x_linha = []
    for i in range(49):
        x_linha.append(int(line[i]))
    x_data.append( x_linha )
    y_real.append( int(line[50]) )

# Divide em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_data, y_real, test_size=0.30)
print(len(x_train), len(x_test))

# Cria o SVM, treina e classifica
print("Treinando")
clf_svc = svm.SVC(
    C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
    probability=False, tol=0.001, cache_size=500, class_weight=None, verbose=False, 
    max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None
    )

# clf_nn = MLPClassifier (
#    hidden_layer_sizes=(50,25,10,5,), activation='tanh', solver='adam', alpha=0.0001,
#    batch_size='auto', learning_rate='constant', learning_rate_init=0.00125,
#    power_t=0.5, max_iter=10000, shuffle=True, random_state=None, tol=0.0001,
#    verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#    epsilon=1e-05, n_iter_no_change=10, max_fun=15000
#    )

clf = clf_svc


clf.fit(x_train, y_train)

print("Testando")
y_pred = clf.predict(x_test)

# print(y_test)
for i in range(len(y_pred)):
    print(y_pred[i], end=", ")
print()

# Mostra o resultado
print(classification_report(y_test, y_pred))

cont = 1
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        cont += 1

print(f"Total: {len(y_pred)}, Acertos: {cont} ({cont/len(y_pred)}%)")

